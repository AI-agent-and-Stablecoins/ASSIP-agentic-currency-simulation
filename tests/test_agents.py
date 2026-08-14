from src.agents.agent_factory import build_agent, load_agent_profiles
from src.agents.memory import AgentMemory
from src.blockchain.chain import load_chain_universe
from src.blockchain.routing_engine import generate_candidates
from src.currencies.currency import load_currency_universe
from src.utility.cara import CARAUtility
from src.utility.risk_neutral import RiskNeutralUtility


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


def test_base_agent_defaults_new_population_fields_to_none():
    agent = build_agent(load_agent_profiles()["consumer"])

    assert agent.currency_zone is None
    assert agent.assigned_model is None
    assert agent.cara_coefficient is None


def test_base_agent_accepts_population_fields():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)
    agent.currency_zone = "EUR"
    agent.assigned_model = "anthropic/claude-sonnet-5"
    agent.cara_coefficient = 1.5

    assert agent.currency_zone == "EUR"
    assert agent.assigned_model == "anthropic/claude-sonnet-5"
    assert agent.cara_coefficient == 1.5


def test_build_llm_context_carries_population_fields():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)
    agent.currency_zone = "USD"
    agent.assigned_model = "openai/gpt-5"
    agent.cara_coefficient = 0.5

    context = agent.build_llm_context()

    assert context.currency_zone == "USD"
    assert context.assigned_model == "openai/gpt-5"
    assert context.cara_coefficient == 0.5


def test_build_llm_context_defaults_population_fields_to_none():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)

    context = agent.build_llm_context()

    assert context.currency_zone is None
    assert context.assigned_model is None
    assert context.cara_coefficient is None


def test_build_agent_with_no_overrides_behaves_exactly_as_before():
    profile = load_agent_profiles()["consumer"]

    agent = build_agent(profile)

    assert agent.currency_zone is None
    assert agent.assigned_model is None
    assert agent.cara_coefficient is None
    assert agent.utility_type == profile.utility_type
    assert agent.risk_aversion == profile.risk_aversion


def test_build_agent_accepts_currency_zone_and_assigned_model():
    profile = load_agent_profiles()["consumer"]

    agent = build_agent(profile, currency_zone="EUR", assigned_model="openai/gpt-5")

    assert agent.currency_zone == "EUR"
    assert agent.assigned_model == "openai/gpt-5"


def test_build_agent_cara_override_supersedes_profile_utility():
    profile = load_agent_profiles()["consumer"]  # profile.utility_type == "crra"

    agent = build_agent(profile, cara_override=("cara", 1.5))

    assert agent.utility_type == "cara"
    assert agent.risk_aversion == 1.5
    assert agent.cara_coefficient == 1.5
    assert isinstance(agent.utility_fn, CARAUtility)


def test_build_agent_cara_override_risk_neutral_branch():
    profile = load_agent_profiles()["bank"]  # profile.utility_type == "cara"

    agent = build_agent(profile, cara_override=("risk_neutral", None))

    assert agent.utility_type == "risk_neutral"
    assert isinstance(agent.utility_fn, RiskNeutralUtility)
    assert agent.cara_coefficient == 0.0  # nominal a is still recorded even though utility_type switched


def test_consumer_profile_has_income_fields():
    profile = load_agent_profiles()["consumer"]

    assert profile.income_per_period == 250.0
    assert profile.income_period_days == 7


def test_non_buyer_profile_has_no_income_fields():
    profile = load_agent_profiles()["merchant"]

    assert profile.income_per_period is None
    assert profile.income_period_days is None


def test_build_agent_carries_income_fields_onto_agent():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)

    assert agent.income_per_period == 250.0
    assert agent.income_period_days == 7


def test_build_agent_leaves_income_fields_none_for_a_profile_without_them():
    profile = load_agent_profiles()["merchant"]
    agent = build_agent(profile)

    assert agent.income_per_period is None
    assert agent.income_period_days is None
