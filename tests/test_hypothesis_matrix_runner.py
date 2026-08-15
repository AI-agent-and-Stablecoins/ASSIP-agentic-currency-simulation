import json
import re

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, CohortHoldingsRecord, IndifferencePointRecord, SimulationRunRecord
from src.currencies.currency import load_currency_universe
from src.economy.equivalence_framework import EQUIVALENCE_COMPARISONS
from src.economy.hypothesis_scenarios import (
    HYPOTHESIS_CHAIN_PINS,
    HYPOTHESIS_CURRENCIES,
    baseline_cell_keys,
    build_hypothesis_cell_specs,
)
from src.economy.shocks import load_scenario
from src.llm.llm_router import OPENROUTER_BASE_URL
from src.simulation.hypothesis_matrix_runner import _build_fresh_cell_environment, run_hypothesis_matrix
from src.simulation.matrix_runner import MASTER_SCENARIO_NAME

MODEL_ID = "vendor/model"


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _mock_client(
    decision_currency: str,
    decision_chain: str,
    model_id: str = MODEL_ID,
    switch_field: str | None = None,
    switch_threshold: float | None = None,
    switch_higher_is_better: bool | None = None,
) -> httpx.Client:
    """A combined mock OpenRouter client covering every endpoint
    run_hypothesis_matrix touches: the `/models` preflight check
    (`_resolve_available_models`), regular day-loop decisions
    (`decide_single_model`, via `mock_openrouter_client`'s shape), and --
    when `switch_field` is given -- the post-run switch-elicitation question
    (`cohort_indifference_points`, via `mock_switch_threshold_client`'s
    threshold-dependent shape). A single client must handle all three since
    run_hypothesis_matrix takes only one openrouter_client for the whole
    call, unlike the two separate helpers in tests/llm_test_helpers.py."""
    decision = {
        "action": "ACCEPT",
        "proposed_currency": decision_currency,
        "proposed_chain": decision_chain,
        "amount": 1.0,
        "price": 1.0,
        "reasoning": "test",
    }
    switch_pattern = (
        re.compile(rf"Coin B \([^)]*\):[^\n]*{re.escape(switch_field)}=(-?[\d.]+)") if switch_field else None
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model_id}]})
        assert request.url.path == "/api/v1/chat/completions"
        body = json.loads(request.content)
        if body["model"] != model_id:
            return httpx.Response(404, json={"error": f"no mocked response for model {body['model']!r}"})
        prompt = body["messages"][0]["content"]
        match = switch_pattern.search(prompt) if switch_pattern is not None else None
        if match is not None:
            varied_value = float(match.group(1))
            will_switch = varied_value >= switch_threshold if switch_higher_is_better else varied_value <= switch_threshold
            content = json.dumps({"will_switch": will_switch, "reasoning": "test"})
        else:
            content = json.dumps(decision)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))


def test_run_hypothesis_matrix_rejects_none_openrouter_client():
    with pytest.raises(ValueError):
        run_hypothesis_matrix(
            model_candidates=[MODEL_ID],
            seeds=[0],
            num_days=1,
            openrouter_client=None,
            session=_session(),
            matrix_run_id="none-client-test",
        )


def test_h1_cell_persists_cohort_holdings_with_sane_pct_values():
    session = _session()
    client = _mock_client("USDC", "ethereum")

    results, failures = run_hypothesis_matrix(
        model_candidates=[MODEL_ID],
        seeds=[0],
        num_days=3,
        openrouter_client=client,
        session=session,
        matrix_run_id="test-h1",
        hypotheses=["H1"],
        utility_types=["crra"],
    )

    assert failures == []
    assert len(results) == 4  # H1 baseline + cross-border + 2 event variants
    for result in results:
        assert result.hypothesis == "H1"
        assert result.cohort_holdings is not None
        assert result.cohort_indifference is None
        assert result.num_days_completed == 3

    rows = session.query(CohortHoldingsRecord).all()
    assert len(rows) > 0

    by_run_cohort: dict[tuple[str, float], float] = {}
    for row in rows:
        assert row.currency_symbol in HYPOTHESIS_CURRENCIES["H1"]
        assert 0.0 <= row.pct_of_wealth <= 1.0 + 1e-6
        key = (row.run_id, row.risk_aversion_cohort)
        by_run_cohort[key] = by_run_cohort.get(key, 0.0) + row.pct_of_wealth

    # Every agent's wallet is restricted to exactly H1's 3 currencies, so a
    # cohort's %-of-wealth values across those currencies must sum to ~1 --
    # a direct check that these are percentages, not raw balances/counts.
    assert by_run_cohort
    for total in by_run_cohort.values():
        assert total == pytest.approx(1.0, abs=1e-4)

    # Persisted rows must agree with the in-memory HypothesisCellResult they
    # came from -- a transposition bug between the two (e.g. swapping which
    # dict is cohort-keyed vs. currency-keyed) would pass every check above
    # but fail this one.
    for result in results:
        persisted_for_run = {
            (row.risk_aversion_cohort, row.currency_symbol): row.pct_of_wealth
            for row in rows
            if row.run_id == result.run_id
        }
        in_memory = {
            (cohort, symbol): pct
            for cohort, per_symbol in result.cohort_holdings.items()
            for symbol, pct in per_symbol.items()
        }
        assert persisted_for_run == in_memory


def test_h3_cell_persists_indifference_points_tagged_correctly():
    session = _session()
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    client = _mock_client(
        decision_currency=next(iter(HYPOTHESIS_CURRENCIES["H3"])),
        decision_chain="ethereum",
        switch_field=comparison.varied_field,
        switch_threshold=0.5,
        switch_higher_is_better=True,
    )

    results, failures = run_hypothesis_matrix(
        model_candidates=[MODEL_ID],
        seeds=[0],
        num_days=3,
        openrouter_client=client,
        session=session,
        matrix_run_id="test-h3",
        hypotheses=["H3"],
        utility_types=["crra"],
    )

    assert failures == []
    assert len(results) == 1
    assert results[0].hypothesis == "H3"
    assert results[0].cohort_indifference is not None
    assert results[0].cohort_holdings is None
    assert set(results[0].cohort_indifference.keys()) == {comparison.varied_currency}

    rows = session.query(IndifferencePointRecord).all()
    assert len(rows) > 0
    for row in rows:
        assert row.hypothesis == "H3"
        assert row.fixed_currency == comparison.fixed_currency
        assert row.varied_currency == comparison.varied_currency
        assert row.varied_field == comparison.varied_field


def test_chain_pins_are_wired_into_env_for_a_gas_fee_hypothesis():
    """Regression test for the exact gap docs/superpowers/specs/
    2026-08-14-runner-wiring-design.md Sec 4 closes: without
    `env.currency_chain_pins = spec.chain_pins or {}`, H5/H8/H10/H11's
    chain-pinning silently never takes effect."""
    spec = next(
        s for s in build_hypothesis_cell_specs() if s.hypothesis == "H5" and not s.cross_border and s.event_shock is None
    )
    real_currency_universe = load_currency_universe()
    base_scenario = load_scenario(MASTER_SCENARIO_NAME)

    env, population = _build_fresh_cell_environment(
        spec, "crra", 0, [MODEL_ID], real_currency_universe, base_scenario
    )

    assert env.currency_chain_pins == HYPOTHESIS_CHAIN_PINS["H5"]
    assert env.currency_chain_pins != {}
    assert len(population) > 0


def test_resume_continues_from_last_persisted_day(tmp_path, monkeypatch):
    """Interrupt via a flaky run_timestep (mirroring
    tests/test_matrix_runner.py's own resume test): the failure must land
    BEFORE persist_full_timestep commits that day, or the injected failure
    doesn't test anything real -- a failure AFTER commit would make the next
    call's re-simulation of that same day collide on already-persisted rows,
    which is a test-authoring bug, not a resume-worthiness signal."""
    import src.simulation.hypothesis_matrix_runner as runner_module

    session = _session()
    h3_comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    client = _mock_client(
        next(iter(HYPOTHESIS_CURRENCIES["H3"])),
        "ethereum",
        switch_field=h3_comparison.varied_field,
        switch_threshold=0.5,
        switch_higher_is_better=True,
    )
    matrix_run_id = "resume-test"

    original_run_timestep = runner_module.run_timestep
    call_count = {"n": 0}

    def flaky_run_timestep(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash")
        return original_run_timestep(*args, **kwargs)

    monkeypatch.setattr(runner_module, "run_timestep", flaky_run_timestep)

    first_results, first_failures = run_hypothesis_matrix(
        model_candidates=[MODEL_ID],
        seeds=[0],
        num_days=4,
        openrouter_client=client,
        session=session,
        matrix_run_id=matrix_run_id,
        hypotheses=["H3"],
        utility_types=["crra"],
        checkpoint_dir=tmp_path,
    )

    assert first_results == []
    assert len(first_failures) == 1
    checkpoint_files = list(tmp_path.glob("*.pkl"))
    assert len(checkpoint_files) == 1

    monkeypatch.setattr(runner_module, "run_timestep", original_run_timestep)

    second_results, second_failures = run_hypothesis_matrix(
        model_candidates=[MODEL_ID],
        seeds=[0],
        num_days=4,
        openrouter_client=client,
        session=session,
        matrix_run_id=matrix_run_id,
        hypotheses=["H3"],
        utility_types=["crra"],
        checkpoint_dir=tmp_path,
    )

    assert second_failures == []
    assert len(second_results) == 1
    assert second_results[0].num_days_completed == 4
    assert list(tmp_path.glob("*.pkl")) == []


def test_resume_retries_the_analysis_phase_after_a_crash_there_without_losing_it(tmp_path):
    """Regression test for the bug the Task 6 review found: the day-loop's
    checkpoint used to be deleted immediately after the day loop, before the
    post-run analysis phase ran. A crash in that phase then left a committed
    SimulationRunRecord with no checkpoint -- exactly what the skip-check
    reads as "fully done" -- silently and permanently losing that cell's
    CohortHoldingsRecord/IndifferencePointRecord rows on every later retry.
    Fixed by deleting the checkpoint only after the analysis phase's own
    commit succeeds; this proves a crash there instead resumes into an
    already-exhausted day range and retries the analysis phase."""
    import src.simulation.hypothesis_matrix_runner as runner_module

    session = _session()
    h3_comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    client = _mock_client(
        next(iter(HYPOTHESIS_CURRENCIES["H3"])),
        "ethereum",
        switch_field=h3_comparison.varied_field,
        switch_threshold=0.5,
        switch_higher_is_better=True,
    )
    matrix_run_id = "resume-analysis-phase-test"

    original_cohort_indifference_points = runner_module.cohort_indifference_points
    call_count = {"n": 0}

    def flaky_cohort_indifference_points(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated crash during analysis phase")
        return original_cohort_indifference_points(*args, **kwargs)

    monkeypatch_target = "cohort_indifference_points"
    setattr(runner_module, monkeypatch_target, flaky_cohort_indifference_points)
    try:
        crash_results, crash_failures = run_hypothesis_matrix(
            model_candidates=[MODEL_ID],
            seeds=[0],
            num_days=2,
            openrouter_client=client,
            session=session,
            matrix_run_id=matrix_run_id,
            hypotheses=["H3"],
            utility_types=["crra"],
            checkpoint_dir=tmp_path,
        )
    finally:
        setattr(runner_module, monkeypatch_target, original_cohort_indifference_points)

    assert crash_results == []
    assert len(crash_failures) == 1
    checkpoint_files = list(tmp_path.glob("*.pkl"))
    assert len(checkpoint_files) == 1  # NOT deleted -- the day loop's checkpoint must survive an analysis-phase crash

    second_results, second_failures = run_hypothesis_matrix(
        model_candidates=[MODEL_ID],
        seeds=[0],
        num_days=2,
        openrouter_client=client,
        session=session,
        matrix_run_id=matrix_run_id,
        hypotheses=["H3"],
        utility_types=["crra"],
        checkpoint_dir=tmp_path,
    )

    assert second_failures == []
    assert len(second_results) == 1
    assert second_results[0].num_days_completed == 2
    assert list(tmp_path.glob("*.pkl")) == []

    rows = session.query(IndifferencePointRecord).filter(
        IndifferencePointRecord.run_id.like(f"{matrix_run_id}%")
    ).all()
    assert len(rows) > 0


def test_run_id_contains_utility_type_component():
    session = _session()
    client = _mock_client("USDC", "ethereum")

    results, failures = run_hypothesis_matrix(
        model_candidates=[MODEL_ID],
        seeds=[0],
        num_days=2,
        openrouter_client=client,
        session=session,
        matrix_run_id="utility-check",
        hypotheses=["H1"],
        utility_types=["crra"],
    )

    assert failures == []
    row = session.query(SimulationRunRecord).filter(SimulationRunRecord.run_id.like("%-H1-%")).one()
    assert "crra" in row.run_id


def test_cell_keys_restricts_to_baseline_only_excluding_cross_border_and_event():
    session = _session()
    client = _mock_client("USDC", "ethereum")

    results, failures = run_hypothesis_matrix(
        model_candidates=[MODEL_ID],
        seeds=[0],
        num_days=2,
        openrouter_client=client,
        session=session,
        matrix_run_id="baseline-only-test",
        hypotheses=["H1"],
        cell_keys=baseline_cell_keys(),
        utility_types=["crra"],
    )

    assert failures == []
    assert len(results) == 1  # H1's baseline only -- not H1_cb/H1_depeg_event/H1_bank_failure
    assert results[0].cell_key == "H1"


def test_cell_keys_rejects_unknown_key():
    session = _session()
    client = _mock_client("USDC", "ethereum")

    with pytest.raises(ValueError):
        run_hypothesis_matrix(
            model_candidates=[MODEL_ID],
            seeds=[0],
            num_days=2,
            openrouter_client=client,
            session=session,
            matrix_run_id="bad-cell-key-test",
            cell_keys=["H1_not_a_real_variant"],
        )
