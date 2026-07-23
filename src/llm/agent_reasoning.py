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

from pydantic import BaseModel

from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.decision_adapter import NegotiationAction
from src.llm.market_intelligence import CurrencyProfile
from src.utility.multi_attribute import MultiAttributeWeights


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
