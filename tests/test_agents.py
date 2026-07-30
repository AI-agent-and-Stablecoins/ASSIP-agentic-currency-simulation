from src.agents.agent_factory import build_agent, load_agent_profiles
from src.agents.memory import AgentMemory
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


def test_record_narrative_appends_events():
    memory = AgentMemory()

    memory.record_narrative("On day 5 I held USDC through a banking crisis and lost nothing.")

    assert memory.narrative_events == ["On day 5 I held USDC through a banking crisis and lost nothing."]


def test_record_narrative_caps_at_max_events():
    memory = AgentMemory()

    for day in range(15):
        memory.record_narrative(f"Event on day {day}")

    assert len(memory.narrative_events) == 10
    assert memory.narrative_events[0] == "Event on day 5"  # oldest 5 dropped
    assert memory.narrative_events[-1] == "Event on day 14"
