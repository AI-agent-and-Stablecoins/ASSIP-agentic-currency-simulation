import json as _json
from datetime import datetime, timezone

import httpx

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.agent_reasoning import AgentDecisionContext, AgentUtilityContext, TransactionContext, build_decision_context, prompt_version_for, render_prompt
from src.llm.agent_reasoning import LLMDecisionOutcome, decide
from src.llm.agent_reasoning import CurrencyHistory, MacroHistory
from src.llm.decision_adapter import NegotiationAction
from src.llm.decision_schema import DecisionAction
from src.llm.llm_router import OPENROUTER_BASE_URL, RetryConfig, load_model_roster
from src.llm.market_intelligence import load_currency_profile, LivePriceSnapshot


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


def test_agent_utility_context_defaults_population_fields_to_none():
    context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        wallet_balances={"USDC": 1000.0},
    )

    assert context.currency_zone is None
    assert context.assigned_model is None
    assert context.cara_coefficient is None


def test_agent_utility_context_accepts_population_fields():
    context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        wallet_balances={"USDC": 1000.0},
        currency_zone="EUR",
        assigned_model="anthropic/claude-sonnet-5",
        cara_coefficient=1.5,
    )

    assert context.currency_zone == "EUR"
    assert context.assigned_model == "anthropic/claude-sonnet-5"
    assert context.cara_coefficient == 1.5


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


def _decision_json(action: str = "OFFER", currency: str = "USDC", price: float = 100.0) -> str:
    return _json.dumps(
        {
            "action": action,
            "proposed_currency": currency,
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": price,
            "reasoning": "test",
        }
    )


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _base_decision_context() -> AgentDecisionContext:
    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    return build_decision_context(agent_context, candidates, {}, macro, macro, txn_context)


def test_decide_returns_valid_negotiation_action_on_first_try():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()

    outcome = decide(
        "buyer",
        _base_decision_context(),
        roster,
        client,
        {"USDC"},
        {"ethereum"},
        retry_config=RetryConfig(sleep_fn=lambda s: None),
    )

    assert isinstance(outcome, LLMDecisionOutcome)
    assert outcome.used_deterministic_fallback is False
    assert outcome.negotiation_action.currency_symbol == "USDC"
    assert outcome.correction_attempts == 0
    assert outcome.call_result.actual_model == "anthropic/claude-sonnet-5"


def test_decide_corrects_economically_invalid_decision_with_the_same_model():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, json=_chat_response(_decision_json(currency="NOTACOIN")))
        return httpx.Response(200, json=_chat_response(_decision_json(currency="USDC")))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()

    outcome = decide(
        "buyer",
        _base_decision_context(),
        roster,
        client,
        {"USDC"},
        {"ethereum"},
        retry_config=RetryConfig(sleep_fn=lambda s: None),
    )

    assert outcome.used_deterministic_fallback is False
    assert outcome.correction_attempts == 1
    assert outcome.negotiation_action.currency_symbol == "USDC"


def test_decide_falls_back_deterministically_after_exhausting_correction_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(_decision_json(currency="NOTACOIN")))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()
    fallback_action = NegotiationAction(
        action=DecisionAction.WALK_AWAY,
        price=0.0,
        amount=0.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        reasoning="deterministic fallback",
    )

    outcome = decide(
        "buyer",
        _base_decision_context(),
        roster,
        client,
        {"USDC"},
        {"ethereum"},
        retry_config=RetryConfig(sleep_fn=lambda s: None),
        max_correction_attempts=2,
        deterministic_fallback=lambda: fallback_action,
    )

    assert outcome.used_deterministic_fallback is True
    assert outcome.correction_attempts == 2
    assert outcome.negotiation_action is fallback_action


def test_decide_falls_back_when_every_model_in_the_chain_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()
    fallback_action = NegotiationAction(
        action=DecisionAction.WALK_AWAY,
        price=0.0,
        amount=0.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        reasoning="deterministic fallback",
    )

    outcome = decide(
        "buyer",
        _base_decision_context(),
        roster,
        client,
        {"USDC"},
        {"ethereum"},
        retry_config=RetryConfig(max_retries=1, sleep_fn=lambda s: None),
        deterministic_fallback=lambda: fallback_action,
    )

    assert outcome.used_deterministic_fallback is True
    assert outcome.call_result is None
    assert outcome.negotiation_action is fallback_action


def test_build_decision_context_filters_live_price_snapshots_to_candidates_only():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    snapshots = {
        "USDC": LivePriceSnapshot(ticker="X:USDCUSD", price=1.0001, retrieval_timestamp=datetime.now(timezone.utc)),
        "USDT": LivePriceSnapshot(ticker="X:USDTUSD", price=0.9998, retrieval_timestamp=datetime.now(timezone.utc)),
    }

    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, txn_context, live_price_snapshots=snapshots
    )

    assert set(context.live_price_snapshots.keys()) == {"USDC"}


def test_build_decision_context_filters_currency_history_to_candidates_only():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    currency_history = {
        "USDC": CurrencyHistory(
            trust_now=0.95,
            trust_30d_ago=0.93,
            trust_min_90d=0.91,
            trend="stable",
            depeg_events_90d=0,
            last_event_days_ago=None,
        ),
        "USDT": CurrencyHistory(
            trust_now=0.41,
            trust_30d_ago=0.55,
            trust_min_90d=0.38,
            trend="declining",
            depeg_events_90d=2,
            last_event_days_ago=6,
        ),
    }

    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, txn_context, currency_history=currency_history
    )

    assert set(context.currency_history.keys()) == {"USDC"}


def test_render_prompt_includes_live_price_block():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    snapshots = {"USDC": LivePriceSnapshot(ticker="X:USDCUSD", price=1.0001, retrieval_timestamp=datetime.now(timezone.utc))}
    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, txn_context, live_price_snapshots=snapshots
    )

    prompt = render_prompt("buyer", context, "{}")

    assert "1.0001" in prompt


def test_render_prompt_reports_unavailable_live_price_explicitly_not_silently():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    snapshots = {
        "USDC": LivePriceSnapshot(
            ticker="X:USDCUSD",
            price=None,
            retrieval_timestamp=datetime.now(timezone.utc),
            unavailable_reason="no data returned for this ticker",
        )
    }
    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, txn_context, live_price_snapshots=snapshots
    )

    prompt = render_prompt("buyer", context, "{}")

    assert "unavailable" in prompt


def test_currency_history_renders_into_the_prompt():
    context = _base_decision_context()
    context.currency_history = {
        "USDT": CurrencyHistory(
            trust_now=0.41,
            trust_30d_ago=0.55,
            trust_min_90d=0.38,
            trend="declining",
            depeg_events_90d=2,
            last_event_days_ago=6,
            recent_events=["Day 44: brief 1.8% depeg, recovered in 2 days"],
        )
    }
    context.macro_history = MacroHistory(
        confidence_now=0.9, confidence_30d_ago=0.95, days_since_last_shock=6, last_shock_type="depeg_event"
    )
    schema_json = "{}"

    prompt = render_prompt("buyer", context, schema_json)

    assert "declining" in prompt
    assert "Day 44: brief 1.8% depeg" in prompt
    assert "days_since_last_shock" in prompt or "6" in prompt


def test_currency_history_defaults_to_empty_and_still_renders():
    context = _base_decision_context()
    schema_json = "{}"

    prompt = render_prompt("buyer", context, schema_json)

    assert "History" in prompt
