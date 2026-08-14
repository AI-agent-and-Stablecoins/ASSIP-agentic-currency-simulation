import random

from src.agents.population import generate_hypothesis_population
from src.currencies.currency import load_currency_universe
from src.economy.hypothesis_scenarios import HYPOTHESIS_CURRENCIES
from src.economy.macro_state import MacroState
from src.economy.wallet_seeding import seed_restricted_wallets
from src.simulation.environment import Environment
from src.simulation.timestep import run_timestep
from src.transactions.transaction import TransactionStatus


def _build_hypothesis_env(hypothesis: str, utility_type: str, seed: int) -> Environment:
    real_currencies = load_currency_universe()
    restricted = {symbol: real_currencies[symbol] for symbol in HYPOTHESIS_CURRENCIES[hypothesis]}
    population = generate_hypothesis_population(seed, ["vendor/model"], utility_type)
    env = Environment.build_from_population("baseline", population, currencies=restricted)
    seed_restricted_wallets(env.agents, restricted, real_currencies, MacroState().peg_reference_rates)
    return env


def test_h3_produces_settled_transactions_once_wallets_are_seeded():
    """H3 (TDUSD vs USDT) is disjoint from every profile's default
    initial_wallet (USDC/EURC) -- without seed_restricted_wallets, no agent
    holds a symbol generate_candidates recognizes, so zero transactions can
    ever settle. This is the direct regression test for that gap."""
    env = _build_hypothesis_env("H3", "crra", seed=0)
    rng = random.Random(0)

    settled = []
    for day in range(5):
        result = run_timestep(env, day, rng)
        settled.extend(tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED)

    assert len(settled) > 0
    assert all(tx.currency_symbol in HYPOTHESIS_CURRENCIES["H3"] for tx in settled)
