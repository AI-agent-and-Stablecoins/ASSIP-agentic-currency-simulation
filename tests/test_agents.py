from src.agents.agent_factory import build_agent, load_agent_profiles
from src.blockchain.chain import load_chain_universe
from src.blockchain.routing_engine import generate_candidates
from src.currencies.currency import load_currency_universe


def test_agent_only_selects_known_currencies_and_chains():
    currencies = load_currency_universe()
    chains = load_chain_universe()
    profiles = load_agent_profiles()
    agent = build_agent(profiles["consumer"])

    candidates = generate_candidates(agent.wallet.balances, currencies, chains)
    chosen = agent.choose_currency_and_chain(candidates)

    assert chosen.currency_symbol in currencies
    assert chosen.chain_name in chains


def test_wallet_withdraw_never_goes_negative():
    profiles = load_agent_profiles()
    agent = build_agent(profiles["consumer"])
    balance = agent.wallet.balances["USDC"]

    assert agent.wallet.withdraw("USDC", balance + 1) is False
    assert agent.wallet.balances["USDC"] == balance


def test_preferences_move_toward_positive_outcome():
    profiles = load_agent_profiles()
    agent = build_agent(profiles["consumer"])

    before = agent.preferences.score("USDC")
    agent.update_memory("USDC", success=True)
    after = agent.preferences.score("USDC")

    assert after > before
