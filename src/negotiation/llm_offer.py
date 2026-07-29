"""Immutable offer record for the LLM-driven negotiation engine.

Distinct from src.negotiation.offer.Offer (the rule-based engine's offer
type) so nothing here can break Phase 1's tested rule-based negotiation path
-- see llm_negotiation_engine.py. Counter-offers must create new LLMOffer
instances referencing previous_offer_id rather than mutating an existing one.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.llm.decision_schema import DecisionAction
from src.utils.helpers import generate_id


class LLMOffer(BaseModel):
    offer_id: str = Field(default_factory=lambda: generate_id("offer"))
    negotiation_id: str
    previous_offer_id: str | None = None
    agent_id: str
    action: DecisionAction
    price: float
    currency_symbol: str
    chain_name: str
    reasoning: str
    round: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
