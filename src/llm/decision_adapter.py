"""Sits between raw LLM output and the negotiation engine.

Converts a schema-valid Decision into the negotiation engine's internal
action type after checking economic validity (currency/chain support,
positive amount/price, sufficient funds for ACCEPT) -- the "economically
invalid" tier of the three-tier failure handling described in
llm_router.py. Keeping this check here, not in llm_negotiation_engine.py,
means the negotiation state machine never has to know about the LLM-specific
Decision schema.
"""

from pydantic import BaseModel

from src.llm.decision_schema import Decision, DecisionAction, validate_decision


class NegotiationAction(BaseModel):
    action: DecisionAction
    price: float
    amount: float
    currency_symbol: str
    chain_name: str
    reasoning: str


class DecisionValidationError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def adapt_decision(
    decision: Decision,
    supported_currencies: set[str],
    supported_chains: set[str],
    wallet_balances: dict[str, float],
) -> NegotiationAction:
    result = validate_decision(decision, supported_currencies, supported_chains, wallet_balances)
    if not result.is_valid:
        raise DecisionValidationError(result.reason or "invalid decision")

    return NegotiationAction(
        action=decision.action,
        price=decision.price,
        amount=decision.amount,
        currency_symbol=decision.proposed_currency,
        chain_name=decision.proposed_chain,
        reasoning=decision.reasoning,
    )
