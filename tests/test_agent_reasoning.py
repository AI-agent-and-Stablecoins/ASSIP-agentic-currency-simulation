from src.agents.agent_factory import build_agent, load_agent_profiles
from src.llm.agent_reasoning import AgentUtilityContext


def test_build_llm_context_surfaces_crra_agent_parameters():
    profiles = load_agent_profiles()
    agent = build_agent(profiles["consumer"])  # utility_type: crra, risk_aversion: 3.0

    context = agent.build_llm_context()

    assert isinstance(context, AgentUtilityContext)
    assert context.agent_id == agent.agent_id
    assert context.utility_type == "crra"
    assert context.risk_aversion == 3.0
    assert context.eis is None
    assert context.wallet_balances == agent.wallet.balances


def test_build_llm_context_surfaces_multi_attribute_agent_weights():
    profiles = load_agent_profiles()
    agent = build_agent(profiles["merchant"])  # utility_type: multi_attribute

    context = agent.build_llm_context()

    assert context.utility_type == "multi_attribute"
    assert context.multi_attribute_weights is not None
    assert context.multi_attribute_weights.liquidity_weight == 0.35
