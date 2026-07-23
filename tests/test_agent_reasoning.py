from src.agents.agent_factory import build_agent, load_agent_profiles
from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.agent_reasoning import AgentDecisionContext, AgentUtilityContext, TransactionContext, build_decision_context
from src.llm.market_intelligence import load_currency_profile


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


def _option(**overrides) -> CurrencyChainOption:
    defaults = dict(
        currency_symbol="USDC",
        chain_name="ethereum",
        governance_score=0.95,
        liquidity_score=0.97,
        peg_error=0.0003,
        gas_fee=2.5,
        finality_seconds=12.0,
        genius_compliant=True,
    )
    defaults.update(overrides)
    return CurrencyChainOption(**defaults)


def test_build_decision_context_filters_profiles_to_candidate_currencies_only():
    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC"), _option(currency_symbol="EURC", governance_score=0.90)]
    profiles = {"USDC": load_currency_profile("USDC"), "USDT": load_currency_profile("USDT")}
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)

    context = build_decision_context(agent_context, candidates, profiles, macro, macro, txn_context)

    assert isinstance(context, AgentDecisionContext)
    # USDT was in the profile corpus passed in, but no candidate proposes USDT --
    # it must not leak into the context (keeps the prompt focused, per the
    # hypothesis -> context-field traceability rule in the design doc).
    assert set(context.currency_profiles.keys()) == {"USDC"}
    assert context.transaction_context.is_cross_border is False
    assert context.conversation_history == []
    assert context.governance_prompt_enabled is False
    assert context.opponent_offer is None
