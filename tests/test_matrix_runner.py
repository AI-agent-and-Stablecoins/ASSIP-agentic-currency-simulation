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
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import AgentRecord, Base, SimulationRunRecord, TimestepLogRecord, TransactionRecord
from src.agents.population import generate_agent_population
from src.simulation.matrix_runner import MASTER_SCENARIO_NAME, run_matrix
from tests.llm_test_helpers import mock_openrouter_client

MODEL_CANDIDATES = ["vendor/fake-model"]


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _by_cell_key(results, cell_key: str, seed: int = 0):
    return next(r for r in results if r.cell_key == cell_key and r.seed == seed)


def test_run_matrix_with_dry_run_true_does_not_require_real_clients():
    results = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )
    assert len(results) == 13  # 1 master + 6 domestic + 6 cross-border, 1 seed


def test_run_matrix_produces_13_cells_per_seed():
    results = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0, 1], num_days=1, dry_run=True, session=_session()
    )
    assert len(results) == 26  # 13 cells x 2 seeds
    cell_keys = {r.cell_key for r in results}
    assert len(cell_keys) == 13
    for cell_key in cell_keys:
        seeds_seen = {r.seed for r in results if r.cell_key == cell_key}
        assert seeds_seen == {0, 1}


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
    results = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )
    master = _by_cell_key(results, "master")
    assert master.num_currencies == 9
    assert not master.is_cross_border


def test_sandbox_cells_use_exactly_two_currencies_domestic_and_cross_border():
    results = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )
    domestic = _by_cell_key(results, "liquidity_vs_governance_domestic")
    cross_border = _by_cell_key(results, "liquidity_vs_governance_cross_border")

    assert domestic.num_currencies == 2
    assert not domestic.is_cross_border
    assert cross_border.num_currencies == 2
    assert cross_border.is_cross_border


def test_every_sandbox_gets_a_domestic_and_a_cross_border_cell():
    results = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=_session()
    )
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

    results = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[seed], num_days=2, dry_run=True, session=_session()
    )
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

    results = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[seed], num_days=2, dry_run=True, session=_session()
    )
    domestic = _by_cell_key(results, "liquidity_vs_governance_domestic", seed=seed)

    all_transactions = [tx for day in domestic.daily_results for tx in day.transactions]
    assert len(all_transactions) > 0
    same_zone = [tx for tx in all_transactions if zone_by_id[tx.buyer_id] == zone_by_id[tx.seller_id]]
    assert len(same_zone) > 0


def test_run_matrix_persists_provenance_agent_and_transaction_rows():
    session = _session()
    results = run_matrix(
        model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=session
    )

    run_rows = session.query(SimulationRunRecord).all()
    assert len(run_rows) == 13
    run_ids = {row.run_id for row in run_rows}
    assert run_ids == {r.run_id for r in results}
    assert all(row.scenario_name == MASTER_SCENARIO_NAME for row in run_rows)

    # Every cell/seed ran exactly 1 day -> exactly 13 TimestepLogRecords.
    assert session.query(TimestepLogRecord).count() == 13

    # Every cell built its own 100-agent population; AgentRepository upserts
    # by agent_id, and deterministic agent_ids repeat across cells for the
    # same seed, so the total row count is at most 100 (never more).
    assert session.query(AgentRecord).count() <= 100
    assert session.query(AgentRecord).count() > 0

    total_transactions_returned = sum(len(day.transactions) for r in results for day in r.daily_results)
    assert session.query(TransactionRecord).count() == total_transactions_returned


def test_run_matrix_works_with_a_supplied_mock_openrouter_client_under_dry_run():
    """`dry_run=True` doesn't forbid a caller-supplied client -- it only
    means a real one is never required. Supplying a mock client should
    switch the day loop onto the LLM-driven path (`use_llm=True`
    internally) without hitting any real network endpoint."""
    client = mock_openrouter_client({"vendor/fake-model": {
        "action": "ACCEPT",
        "proposed_currency": "SBX1_HILIQ_LOGOV",
        "proposed_chain": "ethereum",
        "amount": 1.0,
        "price": 90.0,
        "reasoning": "test reasoning",
    }})

    results = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        openrouter_client=client,
        session=_session(),
    )

    assert len(results) == 13
    master = _by_cell_key(results, "master")
    assert master.daily_results[0].llm_decisions  # LLM path actually ran
