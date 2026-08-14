import pytest

from src.agents.population import generate_hypothesis_population
from src.currencies.currency import load_currency_universe
from src.economy.macro_state import MacroState
from src.economy.wallet_seeding import seed_restricted_wallets


def test_seed_restricted_wallets_splits_evenly_by_usd_value():
    real_currencies = load_currency_universe()
    peg_reference_rates = MacroState().peg_reference_rates
    population = generate_hypothesis_population(seed=0, model_candidates=["vendor/model"], utility_type="crra")
    agents = {a.agent_id: a for a in population}
    restricted = {"TDUSD": real_currencies["TDUSD"], "USDT": real_currencies["USDT"]}

    seed_restricted_wallets(agents, restricted, real_currencies, peg_reference_rates)

    for agent in agents.values():
        assert set(agent.wallet.balances.keys()) == {"TDUSD", "USDT"}


def test_seed_restricted_wallets_gives_a_safe_floor_to_an_empty_wallet():
    real_currencies = load_currency_universe()
    peg_reference_rates = MacroState().peg_reference_rates
    population = generate_hypothesis_population(seed=0, model_candidates=["vendor/model"], utility_type="crra")
    agent = next(a for a in population if a.profile_name == "consumer")
    agent.wallet.balances = {}
    agents = {agent.agent_id: agent}
    restricted = {"USDC": real_currencies["USDC"]}

    seed_restricted_wallets(agents, restricted, real_currencies, peg_reference_rates)

    assert agent.wallet.balances["USDC"] == pytest.approx(1000.0)
