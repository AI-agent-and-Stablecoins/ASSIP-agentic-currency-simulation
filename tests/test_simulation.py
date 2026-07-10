import random

import pytest

from src.simulation.environment import Environment
from src.simulation.simulation_runner import SimulationConfig, SimulationRunner
from src.simulation.timestep import run_timestep
from src.transactions.transaction import TransactionStatus


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
