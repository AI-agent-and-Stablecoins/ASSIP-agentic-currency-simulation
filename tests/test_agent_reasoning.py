from src.agents.agent_factory import build_agent, load_agent_profiles
from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.agent_reasoning import AgentDecisionContext, AgentUtilityContext, TransactionContext, build_decision_context, prompt_version_for, render_prompt
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


def test_render_prompt_includes_all_context_sections_and_respects_governance_flag():
    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    profiles = {"USDC": load_currency_profile("USDC")}
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)

    baseline_context = build_decision_context(
        agent_context, candidates, profiles, macro, macro, txn_context, governance_prompt_enabled=False
    )
    governance_context = build_decision_context(
        agent_context, candidates, profiles, macro, macro, txn_context, governance_prompt_enabled=True
    )

    baseline_prompt = render_prompt("buyer", baseline_context, "{}")
    governance_prompt = render_prompt("buyer", governance_context, "{}")

    assert "USDC" in baseline_prompt
    assert "Risk aversion" in baseline_prompt
    assert "Governance emphasis" not in baseline_prompt
    assert "Governance emphasis" in governance_prompt


def test_render_prompt_works_for_all_four_agent_classes():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="seller",
        risk_profile="medium",
        utility_type="multi_attribute",
        wallet_balances={"USDC": 500.0},
    )
    candidates = [_option()]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    context = build_decision_context(agent_context, candidates, {}, macro, macro, txn_context)

    for agent_class in ["buyer", "seller", "investor", "bank"]:
        prompt = render_prompt(agent_class, context, "{}")
        assert "USDC" in prompt


def test_prompt_version_for_returns_stable_identifier():
    assert prompt_version_for("buyer") == "buyer_prompt@v1"
