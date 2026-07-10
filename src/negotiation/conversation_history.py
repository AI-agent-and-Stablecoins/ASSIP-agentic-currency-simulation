"""Stores the ordered log of offers/counter-offers exchanged in a negotiation."""

from pydantic import BaseModel, Field

from src.negotiation.offer import Offer


class ConversationLog(BaseModel):
    offers: list[Offer] = Field(default_factory=list)
    outcome: str | None = None

    def add(self, offer: Offer) -> None:
        self.offers.append(offer)

    def finalize(self, outcome: str) -> None:
        self.outcome = outcome
