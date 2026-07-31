import random

import pytest

from src.blockchain.routing_engine import generate_candidates
from src.economy.shocks import ShockEvent, ShockType
from src.economy.trust import TrustLedger
from src.simulation.environment import Environment
from src.simulation.event_queue import EventQueue
from src.simulation.simulation_runner import SimulationConfig, SimulationRunner
from src.simulation.timestep import run_timestep
from src.transactions.transaction import TransactionStatus


def _build_env_with_shocks(shocks: list[ShockEvent], agent_mix: dict[str, int]) -> Environment:
    """Build an Environment with custom shocks by reusing Environment.build
    and reassigning its event_queue with the given shocks.
    """
    env = Environment.build("baseline", agent_mix)
    env.event_queue = EventQueue(shocks)
    return env


def test_simulation_runs_end_to_end_without_errors():
    config = SimulationConfig(
        agent_mix={"consumer": 3, "merchant": 2},
        num_days=5,
        scenario="baseline",
        random_seed=42,
    )

    result = SimulationRunner().run(config)

    assert len(result.timesteps) == 5
    all_transactions = [tx for step in result.timesteps for tx in step.transactions]
    assert len(all_transactions) > 0
    settled = [tx for tx in all_transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0


def test_simulation_conserves_total_currency_balances():
    env = Environment.build("baseline", {"consumer": 3, "merchant": 2})

    def total_balances() -> dict[str, float]:
        totals: dict[str, float] = {}
        for agent in env.agents.values():
            for symbol, amount in agent.wallet.balances.items():
                totals[symbol] = totals.get(symbol, 0.0) + amount
        return totals

    before = total_balances()

    rng = random.Random(42)
    for day in range(5):
        run_timestep(env, day, rng)

    after = total_balances()

    for symbol, amount in before.items():
        assert after.get(symbol, 0.0) == pytest.approx(amount, rel=1e-6)


def test_run_timestep_reports_fired_shocks_on_the_day_they_fire():
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.INFLATION, magnitude=0.02)],
        {"consumer": 2, "merchant": 2},
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    assert len(result.fired_shocks) == 1
    assert result.fired_shocks[0].type == ShockType.INFLATION


def test_run_timestep_reports_no_fired_shocks_on_a_quiet_day():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    assert result.fired_shocks == []


def test_run_timestep_records_narrative_memory_for_agents_holding_a_shocked_currency():
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")],
        {"consumer": 2, "merchant": 2},
    )
    consumer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    consumer.wallet.balances["USDT"] = 100.0
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    matching = [e for e in result.memory_events if e[0] == consumer.agent_id]
    assert len(matching) == 1
    assert matching[0][1] == "Depeg"
    assert "USDT" in matching[0][2]
    assert matching[0][2] in consumer.memory.narrative_events


def test_environment_starts_with_price_index_of_one():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    assert env.price_index == 1.0


def test_environment_build_constructs_a_trust_ledger():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})

    assert isinstance(env.trust_ledger, TrustLedger)
    # Sanity: it should be initialized from the same currencies the
    # environment holds, not some detached default instance.
    for symbol in env.currencies:
        assert env.trust_ledger.trust_score(symbol) == pytest.approx(env.currencies[symbol].governance_score)


def test_run_timestep_advances_trust_ledger_so_depeg_offset_decays_across_days():
    """The review's core finding: TrustLedger must actually be driven by the
    real simulation loop (run_timestep), not just be constructible/callable
    in isolation. A depeg_event on day 0 should spike USDT's peg_error
    offset via env.trust_ledger; subsequent quiet days should decay that
    offset -- and generate_candidates() called through the production path
    (i.e. via env.trust_ledger) must reflect a shrinking, non-static
    peg_error day over day.
    """
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")],
        {"consumer": 2, "merchant": 2},
    )
    rng = random.Random(0)

    run_timestep(env, day=0, rng=rng)
    day0_offset = env.trust_ledger.peg_error_offset("USDT")
    day0_effective = env.trust_ledger.effective_peg_error("USDT", env.currencies["USDT"].peg_error)
    assert day0_offset == pytest.approx(0.08)

    run_timestep(env, day=1, rng=rng)
    day1_offset = env.trust_ledger.peg_error_offset("USDT")
    day1_effective = env.trust_ledger.effective_peg_error("USDT", env.currencies["USDT"].peg_error)

    # The production call site of generate_candidates (inside run_timestep)
    # is wired to the same ledger: re-deriving candidates directly with
    # env.currencies/env.chains/env.trust_ledger, at this point in time,
    # reproduces the same shrinking peg_error the loop itself is using.
    candidates_day1 = generate_candidates(
        {"USDT": 1.0}, env.currencies, env.chains, env.liquidity_pools, trust_ledger=env.trust_ledger
    )
    usdt_candidate = next(c for c in candidates_day1 if c.currency_symbol == "USDT")
    assert usdt_candidate.peg_error == pytest.approx(day1_effective)

    run_timestep(env, day=2, rng=rng)
    day2_offset = env.trust_ledger.peg_error_offset("USDT")

    # Present but shrinking -- proves the ledger is being genuinely advanced
    # once per day by run_timestep, not left static/dead.
    assert 0.0 < day1_offset < day0_offset
    assert 0.0 < day2_offset < day1_offset
    assert day1_effective < day0_effective
