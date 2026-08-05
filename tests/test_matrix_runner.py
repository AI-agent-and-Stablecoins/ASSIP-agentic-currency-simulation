"""Tests for the Phase 3 Plan 4 Task 13 matrix runner (`run_matrix`).

Every test here uses `dry_run=True` (the default) and passes its own
throwaway in-memory session, matching the existing
`tests/test_*_persistence.py` convention -- see
`src/simulation/matrix_runner.py`'s module docstring for why `dry_run=True`
never requires (and, in most tests here, never receives) a real
`httpx.Client`: the deterministic rule-based day loop runs instead, so
these tests hit zero real network endpoints while still exercising
population generation, environment construction, cross-border pairing, and
full persistence.

Kept fast: tiny `num_days` (1-2), a single fake model candidate, and the
smallest population Plan 3 defines (`generate_agent_population` always
builds the fixed 100-agent roster -- there is no smaller-population knob --
so "fast" here means few days/cells, not few agents).

`run_matrix` returns `(results, failures)` (a 2-tuple) -- every call site
below unpacks both, and asserts `failures == []` unless the test is
specifically about the per-cell/seed error-recovery fix. Tests that need
`MatrixCellResult.daily_results` populated (the pre-fix, full-retention
default) now pass `keep_daily_results=True` explicitly, since
`keep_daily_results=False` is the new default.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import (
    AgentRecord,
    Base,
    LLMDecisionRecord,
    SimulationRunRecord,
    TimestepLogRecord,
    TransactionRecord,
)
from src.agents.population import ROLE_COUNTS, generate_agent_population
from src.currencies.currency import load_currency_universe
from src.currencies.exchange_rates import ExchangeRateTable
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.economy.macro_state import MacroState
from src.economy.sandbox_scenarios import build_sandbox_scenario
from src.economy.shocks import ShockType, load_scenario
from src.simulation.environment import Environment
from src.simulation.matrix_runner import MASTER_SCENARIO_NAME, _build_cell_specs, _seed_sandbox_wallets, run_matrix
from tests.llm_test_helpers import mock_openrouter_client

MODEL_CANDIDATES = ["vendor/fake-model"]


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _by_cell_key(results, cell_key: str, seed: int = 0):
    return next(r for r in results if r.cell_key == cell_key and r.seed == seed)


def test_run_matrix_with_dry_run_true_does_not_require_real_clients():
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )
    assert len(results) == 13  # 1 master + 6 domestic + 6 cross-border, 1 seed
    assert failures == []


def test_run_matrix_produces_13_cells_per_seed():
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0, 1], num_days=1, dry_run=True, session=_session()
    )
    assert failures == []
    assert len(results) == 26  # 13 cells x 2 seeds
    cell_keys = {r.cell_key for r in results}
    assert len(cell_keys) == 13
    for cell_key in cell_keys:
        seeds_seen = {r.seed for r in results if r.cell_key == cell_key}
        assert seeds_seen == {0, 1}


def test_cell_keys_restricts_which_cells_run():
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=2,
        dry_run=True,
        session=_session(),
        cell_keys=["master", "liquidity_vs_governance_domestic"],
    )
    assert failures == []
    assert {r.cell_key for r in results} == {"master", "liquidity_vs_governance_domestic"}


def test_run_matrix_refuses_dry_run_false_without_any_real_clients():
    with pytest.raises(ValueError):
        run_matrix(model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=False, session=_session())


def test_run_matrix_refuses_dry_run_false_with_only_one_real_client(monkeypatch):
    fake_openrouter = mock_openrouter_client({})
    with pytest.raises(ValueError):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=1,
            dry_run=False,
            openrouter_client=fake_openrouter,
            polygon_client=None,
            session=_session(),
        )


def test_master_cell_uses_the_full_nine_currency_universe():
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )
    assert failures == []
    master = _by_cell_key(results, "master")
    assert master.num_currencies == 9
    assert not master.is_cross_border


def test_sandbox_cells_use_exactly_two_currencies_domestic_and_cross_border():
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )
    assert failures == []
    domestic = _by_cell_key(results, "liquidity_vs_governance_domestic")
    cross_border = _by_cell_key(results, "liquidity_vs_governance_cross_border")

    assert domestic.num_currencies == 2
    assert not domestic.is_cross_border
    assert cross_border.num_currencies == 2
    assert cross_border.is_cross_border


def test_every_sandbox_gets_a_domestic_and_a_cross_border_cell():
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )
    assert failures == []
    cell_keys = {r.cell_key for r in results}
    expected_sandboxes = {
        "liquidity_vs_governance",
        "governance_vs_stability",
        "liquidity_vs_stability",
        "asset_backing_vs_liquidity",
        "asset_backing_vs_stability",
        "asset_backing_vs_governance",
    }
    for sandbox in expected_sandboxes:
        assert f"{sandbox}_domestic" in cell_keys
        assert f"{sandbox}_cross_border" in cell_keys
    assert "master" in cell_keys


def test_cross_border_cells_only_settle_transactions_between_zone_mismatched_agents():
    """The observable side effect of `CrossZoneMarketplace` (see
    matrix_runner's module docstring): every transaction settled in a
    cross-border cell must be between a buyer and seller whose
    `currency_zone`s differ. Agent IDs/zones are deterministic per seed
    (`generate_agent_population`), so this test independently regenerates
    the same population to build a zone lookup rather than reaching into
    `run_matrix`'s internals.
    """
    seed = 0
    population = generate_agent_population(seed, MODEL_CANDIDATES)
    zone_by_id = {agent.agent_id: agent.currency_zone for agent in population}

    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[seed],
        num_days=2,
        dry_run=True,
        session=_session(),
        keep_daily_results=True,
    )
    assert failures == []
    cross_border = _by_cell_key(results, "liquidity_vs_governance_cross_border", seed=seed)

    all_transactions = [tx for day in cross_border.daily_results for tx in day.transactions]
    assert len(all_transactions) > 0  # the sandbox is large enough (35 buyers x 35 sellers) to settle at least one
    for tx in all_transactions:
        assert zone_by_id[tx.buyer_id] is not None
        assert zone_by_id[tx.seller_id] is not None
        assert zone_by_id[tx.buyer_id] != zone_by_id[tx.seller_id]


def test_domestic_cells_are_not_forced_cross_zone():
    """Contrast case for the above: the domestic sandbox cell (no
    CrossZoneMarketplace swapped in) should be able to settle at least one
    same-zone transaction, confirming the cross-border filter is a real,
    cell-specific behavior difference and not just a universal side effect
    of this codebase's negotiation mechanics."""
    seed = 0
    population = generate_agent_population(seed, MODEL_CANDIDATES)
    zone_by_id = {agent.agent_id: agent.currency_zone for agent in population}

    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[seed],
        num_days=2,
        dry_run=True,
        session=_session(),
        keep_daily_results=True,
    )
    assert failures == []
    domestic = _by_cell_key(results, "liquidity_vs_governance_domestic", seed=seed)

    all_transactions = [tx for day in domestic.daily_results for tx in day.transactions]
    assert len(all_transactions) > 0
    same_zone = [tx for tx in all_transactions if zone_by_id[tx.buyer_id] == zone_by_id[tx.seller_id]]
    assert len(same_zone) > 0


def test_run_matrix_persists_provenance_agent_and_transaction_rows():
    session = _session()
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=session
    )
    assert failures == []

    run_rows = session.query(SimulationRunRecord).all()
    assert len(run_rows) == 13
    run_ids = {row.run_id for row in run_rows}
    assert run_ids == {r.run_id for r in results}
    assert all(row.scenario_name == MASTER_SCENARIO_NAME for row in run_rows)

    # Every cell/seed ran exactly 1 day -> exactly 13 TimestepLogRecords.
    assert session.query(TimestepLogRecord).count() == 13

    # Every cell built its own 100-agent population. Deterministic agent_ids
    # repeat across cells for the same seed, but `agents` is keyed
    # `(run_id, id)` now -- so each of the 13 cells gets its OWN 100 rows.
    #
    # This assertion used to read `count() <= 100`, encoding the old bare-`id`
    # primary key's behavior: 12 of the 13 cells' agent rows never existed at
    # all, because each cell's upsert found the previous cell's row for the
    # same id and left it alone (and `_sync_wallet` then overwrote that row's
    # wallet mirror). That was the bug, not the contract -- see
    # `AgentRecord`'s docstring and tests/test_agent_persistence.py's
    # run-scoping regression tests.
    assert session.query(AgentRecord).count() == 13 * sum(ROLE_COUNTS.values())
    assert {row.run_id for row in session.query(AgentRecord).all()} == run_ids

    # `total_transactions` is a cheap per-cell aggregate populated
    # regardless of `keep_daily_results` (default False here, so
    # `daily_results` itself stays empty -- see the two dedicated
    # keep_daily_results tests below for that behavior).
    assert all(r.daily_results == [] for r in results)
    total_transactions_returned = sum(r.total_transactions for r in results)
    assert session.query(TransactionRecord).count() == total_transactions_returned


def test_run_matrix_refuses_any_externally_supplied_client_under_dry_run_true():
    """`dry_run=True` must guarantee no real network call is possible --
    `run_matrix` cannot distinguish a real `httpx.Client` from a test-only
    mock one by inspecting the object, so under `dry_run=True` it now
    refuses ANY externally-supplied client (real or mock) rather than
    trusting the caller. Use `exercise_llm_path=True` (optionally with
    `mock_llm_decision`) to exercise the LLM path under dry_run instead."""
    fake_client = mock_openrouter_client({})

    with pytest.raises(ValueError):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=1,
            dry_run=True,
            openrouter_client=fake_client,
            session=_session(),
        )

    with pytest.raises(ValueError):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=1,
            dry_run=True,
            polygon_client=fake_client,
            session=_session(),
        )


def test_exercise_llm_path_accepts_a_custom_mock_llm_decision_under_dry_run():
    """A caller wanting a specific canned LLM decision under `dry_run=True`
    (e.g. to exercise a particular proposed currency/price) supplies
    `mock_llm_decision` -- a plain response dict, not a client object --
    which `run_matrix` uses to build its own guaranteed-mock internal
    client. There is still no way to pass an actual `httpx.Client` under
    `dry_run=True` (see the refusal test above)."""
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        exercise_llm_path=True,
        mock_llm_decision={
            "action": "ACCEPT",
            "proposed_currency": "SBX1_HILIQ_LOGOV",
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 90.0,
            "reasoning": "test reasoning",
        },
        session=_session(),
        keep_daily_results=True,
    )

    assert failures == []
    assert len(results) == 13
    master = _by_cell_key(results, "master")
    assert master.daily_results[0].llm_decisions  # LLM path actually ran


def test_mock_llm_decision_rejected_outside_exercise_llm_path():
    """`mock_llm_decision` only means anything alongside `exercise_llm_path
    =True` -- passing it without `exercise_llm_path` (or under `dry_run
    =False`) is a caller error, not a silently-ignored no-op."""
    with pytest.raises(ValueError):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=1,
            dry_run=True,
            mock_llm_decision={"action": "ACCEPT"},
            session=_session(),
        )


def test_seed_sandbox_wallets_splits_by_usd_value_not_raw_units():
    """Regression test for the Task 13 review's Critical Fix 1:
    `_seed_sandbox_wallets` must split an agent's wealth EVENLY BY USD VALUE
    across the two sandbox symbols, not by raw unit count. The
    `asset_backing_vs_liquidity` sandbox pairs a gold-pegged symbol
    (peg="XAU", ~2400 USD/unit) against a stablecoin symbol (peg="USD", 1
    USD/unit) -- assigning both the SAME NUMBER of units (the pre-fix
    behavior) would leave the gold-pegged symbol holding ~2400x the actual
    USD value of its stablecoin counterpart."""
    seed = 0
    population = generate_agent_population(seed, MODEL_CANDIDATES)
    agents = {agent.agent_id: agent for agent in population}

    real_currency_universe = load_currency_universe()
    peg_reference_rates = MacroState().peg_reference_rates
    real_rates = ExchangeRateTable(real_currency_universe, peg_reference_rates)

    # Snapshot each agent's correctly-computed original USD value (summing
    # DIFFERENT real currencies' raw units directly, as the pre-fix code
    # did, is itself wrong -- Wallet.total_value_usd is the correct way to
    # total a wallet holding several distinct real currencies).
    original_usd_value = {
        agent_id: agent.wallet.total_value_usd(real_rates) for agent_id, agent in agents.items()
    }

    option_a, option_b = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_liquidity"]
    sandbox_currencies = {option_a.symbol: option_a, option_b.symbol: option_b}
    assert option_a.peg == "XAU"
    assert option_b.peg == "USD"

    _seed_sandbox_wallets(agents, sandbox_currencies, real_currency_universe, peg_reference_rates)

    sandbox_rates = ExchangeRateTable(sandbox_currencies, peg_reference_rates)
    for agent_id, agent in agents.items():
        assert set(agent.wallet.balances.keys()) == {option_a.symbol, option_b.symbol}
        value_a = sandbox_rates.convert(agent.wallet.balances[option_a.symbol], option_a.symbol, "USD")
        value_b = sandbox_rates.convert(agent.wallet.balances[option_b.symbol], option_b.symbol, "USD")

        # Both sandbox currencies hold approximately EQUAL USD value.
        assert value_a == pytest.approx(value_b, rel=0.01)

        # The total (sum of both currencies' USD value) approximately
        # equals the agent's original pre-sandbox wallet USD value.
        expected_total = original_usd_value[agent_id]
        if expected_total <= 0:
            expected_total = 1000.0  # matches _seed_sandbox_wallets' safe floor
        assert (value_a + value_b) == pytest.approx(expected_total, rel=0.01)


def test_run_matrix_can_be_called_twice_against_the_same_database_without_colliding():
    """Regression test for the Task 13 review's Critical Fix 2: two separate
    `run_matrix` calls (e.g. a pilot run followed by the real run, or a
    restart after a partial failure) against the SAME database must not
    collide on `run_id`, since `SimulationRunRecord.run_id` is a primary key
    with no upsert. Without an explicit `matrix_run_id`, each call generates
    its own fresh, unique prefix, so the second call must not raise
    `IntegrityError`."""
    session = _session()

    first, first_failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=session
    )
    second, second_failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=session
    )

    assert first_failures == []
    assert second_failures == []
    assert len(first) == 13
    assert len(second) == 13
    first_run_ids = {r.run_id for r in first}
    second_run_ids = {r.run_id for r in second}
    assert first_run_ids.isdisjoint(second_run_ids)

    assert session.query(SimulationRunRecord).count() == 26


def test_cell_specs_tag_sandbox_cells_with_their_sandbox_key_and_master_with_none():
    """Regression test for the sandbox-scenario wiring fix: _build_cell_specs
    must identify which of the 12 sandbox cells belongs to which sandbox (so
    run_matrix can look up that sandbox's own build_sandbox_scenario result),
    while the master cell must NOT be tagged (it keeps master_simulation.yaml
    unmodified)."""
    specs = {spec.key: spec for spec in _build_cell_specs()}

    assert specs["master"].sandbox_key is None
    for sandbox_name in SANDBOX_CURRENCY_PAIRS:
        assert specs[f"{sandbox_name}_domestic"].sandbox_key == sandbox_name
        assert specs[f"{sandbox_name}_cross_border"].sandbox_key == sandbox_name


def test_sandbox_cells_are_constructed_with_their_own_sandbox_scenario_not_the_raw_master_one():
    """Confirms the actual cell-construction path (Environment.build_from_
    population, called the same way run_matrix's cell loop calls it) wires a
    sandbox cell to build_sandbox_scenario's ScenarioConfig, not
    master_simulation.yaml's shock schedule verbatim. No simulated days are
    run here -- this is a construction-time check, kept fast."""
    specs = {spec.key: spec for spec in _build_cell_specs()}
    master_spec = specs["master"]
    sandbox_spec = specs["liquidity_vs_governance_domestic"]
    assert sandbox_spec.sandbox_key == "liquidity_vs_governance"

    base_scenario = load_scenario(MASTER_SCENARIO_NAME)
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_governance"]
    sandbox_scenario = build_sandbox_scenario("liquidity_vs_governance", option_a, option_b, base_scenario)

    population = generate_agent_population(0, MODEL_CANDIDATES)
    master_env = Environment.build_from_population(
        MASTER_SCENARIO_NAME, population, currencies=master_spec.currencies, scenario=base_scenario
    )
    sandbox_env = Environment.build_from_population(
        MASTER_SCENARIO_NAME, population, currencies=sandbox_spec.currencies, scenario=sandbox_scenario
    )

    assert master_env.scenario.name == "master_simulation"
    assert sandbox_env.scenario.name == "liquidity_vs_governance_sandbox"

    sandbox_symbols = {option_a.symbol, option_b.symbol}
    currency_targeted = [s for s in sandbox_env.scenario.shocks if s.target_currency is not None]
    assert len(currency_targeted) == 3
    assert all(s.target_currency in sandbox_symbols for s in currency_targeted)
    assert {s.type for s in currency_targeted} >= {ShockType.CRISIS_WARNING, ShockType.DEPEG_EVENT}

    # The master cell's own scenario must remain untouched (still names only
    # real-universe symbols from master_simulation.yaml).
    master_currency_targeted = {s.target_currency for s in master_env.scenario.shocks if s.target_currency is not None}
    assert master_currency_targeted.isdisjoint(sandbox_symbols)


def test_run_matrix_with_the_same_explicit_matrix_run_id_twice_does_collide():
    """Contrast case for the above: an explicitly-supplied `matrix_run_id`
    is a deliberate "resume/retry under this exact id" request, so reusing
    the SAME `matrix_run_id` against the SAME database must still collide
    (documented as intentional behavior on the `matrix_run_id` parameter)."""
    session = _session()

    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        session=session,
        matrix_run_id="fixed-run-id",
    )

    with pytest.raises(IntegrityError):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=1,
            dry_run=True,
            session=session,
            matrix_run_id="fixed-run-id",
        )


# --- Regression tests for the memory/progress/error-recovery fix ----------


def test_keep_daily_results_false_default_does_not_retain_full_results_but_still_persists():
    """`keep_daily_results=False` (the new default) must not hold onto full
    day-by-day `TimestepResult`s in memory, but the run must still complete
    normally and persist every day's data -- checked via DB row counts, not
    in-memory result completeness."""
    session = _session()
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=2, dry_run=True, session=session
    )

    assert failures == []
    assert len(results) == 13
    assert all(r.daily_results == [] for r in results)
    assert all(r.num_days_completed == 2 for r in results)

    assert session.query(TimestepLogRecord).count() == 13 * 2
    total_transactions_returned = sum(r.total_transactions for r in results)
    assert session.query(TransactionRecord).count() == total_transactions_returned


def test_keep_daily_results_true_preserves_full_retention_behavior():
    """`keep_daily_results=True` is the opt-in escape hatch that restores the
    original, pre-fix full-retention behavior: every simulated day's full
    `TimestepResult` stays in `MatrixCellResult.daily_results`."""
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=2,
        dry_run=True,
        session=_session(),
        keep_daily_results=True,
    )

    assert failures == []
    assert len(results) == 13
    for r in results:
        assert r.num_days_completed == 2
        assert len(r.daily_results) == 2
        assert [day_result.day for day_result in r.daily_results] == [0, 1]


def test_progress_callback_is_called_once_per_cell_seed_day():
    calls: list[tuple[str, int, int]] = []

    def progress_callback(cell_key: str, seed: int, day: int) -> None:
        calls.append((cell_key, seed, day))

    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0, 1],
        num_days=2,
        dry_run=True,
        session=_session(),
        progress_callback=progress_callback,
    )

    assert failures == []
    assert len(results) == 26  # 13 cells x 2 seeds
    assert len(calls) == 13 * 2 * 2  # cells x seeds x days

    cell_keys = {r.cell_key for r in results}
    for cell_key in cell_keys:
        for seed in (0, 1):
            for day in (0, 1):
                assert (cell_key, seed, day) in calls


def test_run_matrix_records_a_failed_cell_seed_and_still_completes_the_rest(monkeypatch):
    """A single cell/seed's exception (injected here via monkeypatch on
    `run_timestep`) must abort only that cell/seed's remaining days, not the
    whole matrix -- every other cell/seed must still complete, and the
    failure must show up in the returned `failures` list."""
    import src.simulation.matrix_runner as matrix_runner_module

    original_run_timestep = matrix_runner_module.run_timestep
    call_count = {"n": 0}

    def flaky_run_timestep(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("injected failure for testing")
        return original_run_timestep(*args, **kwargs)

    monkeypatch.setattr(matrix_runner_module, "run_timestep", flaky_run_timestep)

    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )

    # _build_cell_specs() yields "master" first, so the first (and, per
    # call_count, only) injected failure lands on the master cell.
    assert len(failures) == 1
    failed_cell_key, failed_seed, exc = failures[0]
    assert failed_cell_key == "master"
    assert failed_seed == 0
    assert isinstance(exc, RuntimeError)

    # The other 12 cells for this seed still ran to completion.
    assert len(results) == 12
    assert "master" not in {r.cell_key for r in results}


# --- Regression test for exercising the LLM path under dry_run ------------


def test_exercise_llm_path_under_dry_run_persists_llm_decisions_without_real_network():
    """`exercise_llm_path=True` (with `dry_run=True`) must make `run_matrix`
    build mock OpenRouter/Polygon clients internally and run every cell with
    `use_llm=True`, so `run_matrix`'s own test suite genuinely exercises the
    LLM-decision + LLM-decision-persistence path -- not just lower-level
    unit tests of `persist_full_timestep` in isolation. No real network call
    happens: both clients are `httpx.MockTransport`-backed fakes built by
    `tests/llm_test_helpers.py`, and this test sandbox has no real network
    access, so a real call would raise rather than silently succeed.

    A prior version of this test only asserted an aggregate
    `LLMDecisionRecord.count() > 0` across all 13 cells -- a check that would
    still pass even if every decision failed `adapt_decision`'s
    currency-validity check and produced a synthetic `WALK_AWAY` (this
    actually happened: the mock's canned `proposed_currency` was a hardcoded
    "USDC", which only the master cell's real 9-currency universe supports;
    all 12 sandbox cells use synthetic 2-currency `SBX*` universes that never
    include "USDC", so every one of their decisions failed validation and
    only the master cell's decisions ever reached ACCEPT). Fixed both here
    (per cell/seed, not in aggregate) and in `run_matrix` (the mock's
    proposed currency is now built per cell from that cell's own supported
    symbols -- see `src/simulation/matrix_runner.py`'s docstring). This test
    now asserts, for every one of the 13 cells, that at least one
    `LLMDecisionRecord` row for that cell's own `run_id` actually reached
    `action == "ACCEPT"` and that cell produced at least one settled
    transaction -- so a regression back to a cell-invalid hardcoded currency
    (or any other change that silently breaks a subset of cells) cannot pass
    this test merely because SOME cell (e.g. just the master cell) still
    succeeds."""
    session = _session()
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
    )

    assert failures == []
    assert len(results) == 13
    assert session.query(LLMDecisionRecord).count() > 0

    cell_keys = {r.cell_key for r in results}
    assert cell_keys == {"master"} | {
        f"{name}_{suffix}" for name in SANDBOX_CURRENCY_PAIRS for suffix in ("domestic", "cross_border")
    }

    for result in results:
        # Each cell's decisions are scoped by its own run_id (LLMDecisionRecord
        # .simulation_id == run_id, per persist_full_timestep), so this proves
        # the accept/settle path was exercised in THIS cell specifically, not
        # merely somewhere in the aggregate.
        cell_decisions = (
            session.query(LLMDecisionRecord).filter(LLMDecisionRecord.simulation_id == result.run_id).all()
        )
        assert cell_decisions, f"cell {result.cell_key!r} (run_id={result.run_id!r}) has no LLM decisions at all"
        assert any(row.action == "ACCEPT" for row in cell_decisions), (
            f"cell {result.cell_key!r} (run_id={result.run_id!r}) never reached action == 'ACCEPT' -- "
            f"actions seen: {sorted({row.action for row in cell_decisions})}"
        )
        assert result.total_transactions > 0, (
            f"cell {result.cell_key!r} (run_id={result.run_id!r}) settled zero transactions"
        )


# --- Regression tests for checkpoint/resume (checkpoint_dir) ---------------


def test_checkpoint_dir_none_default_leaves_no_new_behavior(tmp_path):
    """Zero behavior change for every existing caller: without
    `checkpoint_dir`, run_matrix never touches `tmp_path` at all."""
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=2, dry_run=True, session=_session()
    )
    assert failures == []
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_files_are_cleaned_up_after_a_fully_successful_run(tmp_path):
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=2,
        dry_run=True,
        session=_session(),
        checkpoint_dir=tmp_path,
    )
    assert failures == []
    assert len(results) == 13
    assert list(tmp_path.glob("*.pkl")) == []


def test_a_failed_cell_seed_leaves_a_checkpoint_reflecting_its_last_persisted_day(tmp_path, monkeypatch):
    """`_build_cell_specs()` yields "master" first, so an injected failure on
    the second `run_timestep` call lands on master's SECOND day (day=1) of a
    3-day run -- day 0 must already be checkpointed (next_day=1) by the time
    the failure aborts the rest of master's days."""
    import src.simulation.matrix_runner as matrix_runner_module

    original_run_timestep = matrix_runner_module.run_timestep
    call_count = {"n": 0}

    def flaky_run_timestep(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("injected failure for checkpoint test")
        return original_run_timestep(*args, **kwargs)

    monkeypatch.setattr(matrix_runner_module, "run_timestep", flaky_run_timestep)

    matrix_run_id = "checkpoint-fail-test"
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=3,
        dry_run=True,
        session=_session(),
        checkpoint_dir=tmp_path,
        matrix_run_id=matrix_run_id,
    )

    assert len(failures) == 1
    assert failures[0][0] == "master"
    assert len(results) == 12
    assert "master" not in {r.cell_key for r in results}

    checkpoint_file = tmp_path / f"{matrix_run_id}-master-seed0.pkl"
    assert checkpoint_file.exists()

    import pickle

    with open(checkpoint_file, "rb") as f:
        checkpoint = pickle.load(f)
    assert checkpoint.next_day == 1
    assert checkpoint.num_days_completed == 1

    # Every other (successful) cell's checkpoint was cleaned up.
    assert {p.name for p in tmp_path.glob("*.pkl")} == {f"{matrix_run_id}-master-seed0.pkl"}


def test_resuming_the_whole_matrix_only_redoes_the_failed_cell_seed(tmp_path, monkeypatch):
    """The realistic recovery flow: after a crash, the caller re-invokes
    run_matrix with the SAME matrix_run_id/database/checkpoint_dir. Already-
    complete cells must be skipped (not re-registered, not re-simulated);
    the interrupted cell must resume from its last persisted day and finish;
    the final `failures` list must be empty."""
    import src.simulation.matrix_runner as matrix_runner_module

    original_run_timestep = matrix_runner_module.run_timestep
    call_count = {"n": 0}

    def flaky_run_timestep(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("injected failure for resume test")
        return original_run_timestep(*args, **kwargs)

    monkeypatch.setattr(matrix_runner_module, "run_timestep", flaky_run_timestep)

    session = _session()
    matrix_run_id = "checkpoint-resume-test"

    first_results, first_failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=3,
        dry_run=True,
        session=session,
        checkpoint_dir=tmp_path,
        matrix_run_id=matrix_run_id,
    )
    assert len(first_failures) == 1
    assert len(first_results) == 12

    monkeypatch.setattr(matrix_runner_module, "run_timestep", original_run_timestep)

    second_results, second_failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=3,
        dry_run=True,
        session=session,
        checkpoint_dir=tmp_path,
        matrix_run_id=matrix_run_id,
        keep_daily_results=True,
    )

    assert second_failures == []
    # Only the previously-failed cell (master) is re-run/completed here --
    # the other 12 were skipped as already-complete.
    assert {r.cell_key for r in second_results} == {"master"}
    master = second_results[0]
    assert master.num_days_completed == 3
    # Resumed from day 1, not re-run from day 0: only days 1 and 2 appear in
    # this call's own daily_results (day 0 ran in the FIRST call).
    assert len(master.daily_results) == 2

    assert list(tmp_path.glob("*.pkl")) == []

    # The master cell's full 3 days are all in the database (day 0 from the
    # first call, days 1-2 from the second), scoped under one run_id.
    master_run_id = f"{matrix_run_id}-master-seed0"
    assert session.query(TimestepLogRecord).filter(TimestepLogRecord.run_id == master_run_id).count() == 3
