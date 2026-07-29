"""Structured LLM decision output and its economic-validity check.

The LLM proposes; it never mutates state. Decision is the schema every model
must fill in (phase_2_instructions_v2.md §4C). validate_decision is the
"economically invalid" tier of the three-tier failure handling in
llm_router.py -- distinct from JSON/schema malformation, which llm_router
itself repairs before a Decision object exists at all.
"""

from enum import Enum

from pydantic import BaseModel


class DecisionAction(str, Enum):
    OFFER = "OFFER"
    COUNTER_OFFER = "COUNTER_OFFER"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    WALK_AWAY = "WALK_AWAY"


class Decision(BaseModel):
    action: DecisionAction
    proposed_currency: str
    proposed_chain: str
    amount: float
    price: float
    reasoning: str
    confidence: float | None = None
    utility_estimate: float | None = None
    risk_assessment: str | None = None
    preferred_alternative_currency: str | None = None
    preferred_alternative_chain: str | None = None


class DecisionValidationResult(BaseModel):
    is_valid: bool
    reason: str | None = None


def validate_decision(
    decision: Decision,
    supported_currencies: set[str],
    supported_chains: set[str],
    wallet_balances: dict[str, float],
) -> DecisionValidationResult:
    if decision.action in (DecisionAction.REJECT, DecisionAction.WALK_AWAY):
        return DecisionValidationResult(is_valid=True)
    if decision.proposed_currency not in supported_currencies:
        return DecisionValidationResult(is_valid=False, reason=f"Unsupported currency: {decision.proposed_currency}")
    if decision.proposed_chain not in supported_chains:
        return DecisionValidationResult(is_valid=False, reason=f"Unsupported chain: {decision.proposed_chain}")
    if decision.amount <= 0:
        return DecisionValidationResult(is_valid=False, reason="Amount must be positive")
    if decision.price <= 0:
        return DecisionValidationResult(is_valid=False, reason="Price must be positive")
    if decision.action == DecisionAction.ACCEPT:
        available = wallet_balances.get(decision.proposed_currency, 0.0)
        if available < decision.price:
            return DecisionValidationResult(is_valid=False, reason="Insufficient funds")
    return DecisionValidationResult(is_valid=True)
