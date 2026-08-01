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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import AgentRecord, Base, SimulationRunRecord, TimestepLogRecord, TransactionRecord
from src.agents.population import generate_agent_population
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

    first = run_matrix(model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=session)
    second = run_matrix(model_candidates=MODEL_CANDIDATES, seeds=[0], num_days=1, dry_run=True, session=session)

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
