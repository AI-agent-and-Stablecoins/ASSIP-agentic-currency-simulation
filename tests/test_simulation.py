import random

import pytest

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.blockchain.chain import load_chain_universe
from src.currencies.currency import load_currency_universe
from src.economy.shocks import ShockEvent, ShockType, load_scenario
from src.simulation.environment import Environment
from src.simulation.simulation_runner import SimulationConfig, SimulationRunner
from src.simulation.timestep import run_timestep
from src.transactions.transaction import TransactionStatus


def _build_env_with_shocks(shocks: list[ShockEvent], agent_mix: dict[str, int]) -> Environment:
    """Environment.build has no way to inject extra shocks, and EventQueue
    only exposes pop_due (populated from ScenarioConfig.shocks at
    construction) -- there is no schedule()/add() method to append a shock
    after the fact. So tests that need a shock to fire construct the
    Environment directly with a ScenarioConfig whose shocks list already
    includes it, mirroring Environment.build's own body.
    """
    currencies = load_currency_universe()
    chains = load_chain_universe()
    scenario = load_scenario("baseline").model_copy(update={"shocks": shocks})

    profiles = load_agent_profiles()
    agents = []
    for profile_name, count in agent_mix.items():
        profile = profiles[profile_name]
        agents.extend(build_agent(profile) for _ in range(count))

    return Environment(currencies=currencies, chains=chains, scenario=scenario, agents=agents)


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
