"""Assembles the context an LLM needs to make an economically meaningful
decision, and (in Task 15) drives the actual LLM call.

AgentUtilityContext is the agent-side slice only (identity, risk profile,
utility parameters, wallet) -- everything an agent knows about itself.
Environment-level context (currency governance, market intelligence, macro
state, opponent offers) is assembled separately in Task 13's
build_decision_context, which takes plain values rather than Environment/
BaseAgent objects, matching this codebase's existing layering convention
(e.g. src.blockchain.routing_engine.generate_candidates takes plain
balances, not a Wallet).
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel

from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.decision_adapter import NegotiationAction
from src.llm.market_intelligence import CurrencyProfile
from src.utility.multi_attribute import MultiAttributeWeights

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

PROMPT_VERSIONS: dict[str, str] = {
    "buyer": "buyer_prompt@v1",
    "seller": "seller_prompt@v1",
    "investor": "investor_prompt@v1",
    "bank": "bank_prompt@v1",
}

_GOVERNANCE_EMPHASIS_BLOCK = (
    "# Governance emphasis\n"
    "Pay particular attention to each currency's governance quality: reserve "
    "composition, transparency, issuer risk, and GENIUS Act compliance status "
    "above. These factors should weigh heavily in your decision.\n"
)


class AgentUtilityContext(BaseModel):
    agent_id: str
    agent_class: str
    risk_profile: str
    utility_type: str
    risk_aversion: float | None = None
    eis: float | None = None
    multi_attribute_weights: MultiAttributeWeights | None = None
    wallet_balances: dict[str, float] = {}


class TransactionContext(BaseModel):
    is_cross_border: bool
    origin_currency: str | None = None
    destination_currency: str | None = None
    exchange_rate: float | None = None
    exchange_rate_volatility: float | None = None


class AgentDecisionContext(BaseModel):
    agent: AgentUtilityContext
    candidates: list[CurrencyChainOption]
    currency_profiles: dict[str, CurrencyProfile] = {}
    objective_macro_state: MacroState
    perceived_macro_state: MacroState
    transaction_context: TransactionContext
    opponent_offer: NegotiationAction | None = None
    conversation_history: list[str] = []
    governance_prompt_enabled: bool = False


def build_decision_context(
    agent_context: AgentUtilityContext,
    candidates: list[CurrencyChainOption],
    currency_profiles: dict[str, CurrencyProfile],
    objective_macro_state: MacroState,
    perceived_macro_state: MacroState,
    transaction_context: TransactionContext,
    opponent_offer: NegotiationAction | None = None,
    conversation_history: list[str] | None = None,
    governance_prompt_enabled: bool = False,
) -> AgentDecisionContext:
    candidate_symbols = {candidate.currency_symbol for candidate in candidates}
    relevant_profiles = {
        symbol: profile for symbol, profile in currency_profiles.items() if symbol in candidate_symbols
    }
    return AgentDecisionContext(
        agent=agent_context,
        candidates=candidates,
        currency_profiles=relevant_profiles,
        objective_macro_state=objective_macro_state,
        perceived_macro_state=perceived_macro_state,
        transaction_context=transaction_context,
        opponent_offer=opponent_offer,
        conversation_history=conversation_history or [],
        governance_prompt_enabled=governance_prompt_enabled,
    )


def prompt_version_for(agent_class: str) -> str:
    return PROMPT_VERSIONS[agent_class]


def hash_rendered_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _format_utility_context_block(agent: AgentUtilityContext) -> str:
    parts = [f"Risk profile: {agent.risk_profile}", f"Utility type: {agent.utility_type}"]
    if agent.risk_aversion is not None:
        parts.append(f"Risk aversion (CRRA/CARA-style gamma): {agent.risk_aversion}")
    if agent.eis is not None:
        parts.append(f"EIS-inspired fee-sensitivity parameter: {agent.eis}")
    if agent.multi_attribute_weights is not None:
        w = agent.multi_attribute_weights
        parts.append(
            f"Multi-attribute weights: governance={w.governance_weight}, liquidity={w.liquidity_weight}, "
            f"gas_fee={w.gas_fee_weight}, volatility={w.volatility_weight}, compliance={w.compliance_weight}"
        )
    return "\n".join(parts)


def _format_candidates_block(candidates: list[CurrencyChainOption]) -> str:
    if not candidates:
        return "(no candidates available)"
    lines = [
        f"- {option.currency_symbol} on {option.chain_name}: governance_score={option.governance_score}, "
        f"liquidity_score={option.liquidity_score}, peg_error={option.peg_error}, gas_fee={option.gas_fee}, "
        f"finality_seconds={option.finality_seconds}, genius_compliant={option.genius_compliant}"
        for option in candidates
    ]
    return "\n".join(lines)


def _format_currency_profiles_block(profiles: dict[str, CurrencyProfile]) -> str:
    if not profiles:
        return "(no background information available for these currencies)"
    sections = [
        f"## {symbol}\n{profile.executive_summary}\nGovernance: {profile.governance}\n"
        f"Reserves/transparency: {profile.reserves_and_transparency}"
        for symbol, profile in profiles.items()
    ]
    return "\n\n".join(sections)


def _format_macro_block(objective: MacroState, perceived: MacroState) -> str:
    return (
        f"Objective state: inflation={objective.inflation}, interest_rate={objective.interest_rate}, "
        f"gold_price={objective.gold_price}, confidence_index={objective.confidence_index}\n"
        f"Your perceived state (may differ from objective): inflation={perceived.inflation}, "
        f"interest_rate={perceived.interest_rate}, gold_price={perceived.gold_price}, "
        f"confidence_index={perceived.confidence_index}"
    )


def _format_transaction_block(txn: TransactionContext) -> str:
    if not txn.is_cross_border:
        return "Domestic transaction."
    return (
        f"Cross-border transaction: {txn.origin_currency} -> {txn.destination_currency}, "
        f"exchange_rate={txn.exchange_rate}, exchange_rate_volatility={txn.exchange_rate_volatility}"
    )


def _format_conversation_block(history: list[str]) -> str:
    return "\n".join(history) if history else "(negotiation has not started yet)"


def render_prompt(agent_class: str, context: AgentDecisionContext, schema_json: str) -> str:
    template = (PROMPTS_DIR / f"{agent_class}_prompt.txt").read_text(encoding="utf-8")
    fields = {
        "utility_context_block": _format_utility_context_block(context.agent),
        "candidates_block": _format_candidates_block(context.candidates),
        "currency_profiles_block": _format_currency_profiles_block(context.currency_profiles),
        "macro_block": _format_macro_block(context.objective_macro_state, context.perceived_macro_state),
        "transaction_block": _format_transaction_block(context.transaction_context),
        "governance_block": _GOVERNANCE_EMPHASIS_BLOCK if context.governance_prompt_enabled else "",
        "conversation_block": _format_conversation_block(context.conversation_history),
        "schema_block": schema_json,
    }
    return template.format(**fields)
