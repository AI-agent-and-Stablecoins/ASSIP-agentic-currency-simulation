"""Multi-round LLM-driven negotiation state machine.

Additive alongside src.negotiation.negotiation_engine's rule-based
negotiate() -- that function and its tests are untouched; this is a
separate path used only when a caller opts into it (see
experiments/experiment_007_governance_prompting.py). A hard max_rounds cap
guarantees termination, the same guarantee the rule-based engine already
provides.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from src.llm.decision_adapter import NegotiationAction
from src.llm.decision_schema import DecisionAction
from src.negotiation.llm_offer import LLMOffer
from src.utils.helpers import generate_id


class NegotiationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WALKED_AWAY = "walked_away"
    MAX_ROUNDS_REACHED = "max_rounds_reached"


class NegotiationSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    negotiation_id: str = Field(default_factory=lambda: generate_id("neg"))
    buyer_id: str
    seller_id: str
    current_round: int = 0
    max_rounds: int
    status: NegotiationStatus = NegotiationStatus.IN_PROGRESS
    initial_offer: LLMOffer | None = None
    current_offer: LLMOffer | None = None
    current_currency: str | None = None
    current_blockchain: str | None = None
    conversation_history: list[LLMOffer] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def record_offer(self, agent_id: str, action: NegotiationAction) -> LLMOffer:
        offer = LLMOffer(
            negotiation_id=self.negotiation_id,
            previous_offer_id=self.current_offer.offer_id if self.current_offer else None,
            agent_id=agent_id,
            action=action.action,
            price=action.price,
            currency_symbol=action.currency_symbol,
            chain_name=action.chain_name,
            reasoning=action.reasoning,
            round=self.current_round,
        )
        self.conversation_history.append(offer)
        if self.initial_offer is None:
            self.initial_offer = offer
        self.current_offer = offer
        self.current_currency = offer.currency_symbol
        self.current_blockchain = offer.chain_name
        return offer

    def finalize(self, status: NegotiationStatus) -> None:
        self.status = status
        self.completed_at = datetime.now(timezone.utc)


def run_llm_negotiation(
    buyer_id: str,
    seller_id: str,
    buyer_decide: Callable[[NegotiationSession], NegotiationAction],
    seller_decide: Callable[[NegotiationSession], NegotiationAction],
    max_rounds: int = 10,
) -> NegotiationSession:
    """Alternates buyer_decide/seller_decide turns (buyer opens) until one
    side ACCEPTs, REJECTs, or WALKs_AWAY, or max_rounds is hit.

    buyer_decide/seller_decide are injected callables (typically closures
    around src.llm.agent_reasoning.decide) so this module has no direct
    dependency on the LLM router -- it only knows about NegotiationAction.
    """
    session = NegotiationSession(buyer_id=buyer_id, seller_id=seller_id, max_rounds=max_rounds)
    turns = [(buyer_id, buyer_decide), (seller_id, seller_decide)]

    for round_number in range(max_rounds):
        session.current_round = round_number
        agent_id, decide_fn = turns[round_number % 2]
        action = decide_fn(session)
        session.record_offer(agent_id, action)

        if action.action == DecisionAction.ACCEPT:
            session.finalize(NegotiationStatus.ACCEPTED)
            return session
        if action.action == DecisionAction.REJECT:
            session.finalize(NegotiationStatus.REJECTED)
            return session
        if action.action == DecisionAction.WALK_AWAY:
            session.finalize(NegotiationStatus.WALKED_AWAY)
            return session
        # OFFER / COUNTER_OFFER: continue to the next round.

    session.finalize(NegotiationStatus.MAX_ROUNDS_REACHED)
    return session
