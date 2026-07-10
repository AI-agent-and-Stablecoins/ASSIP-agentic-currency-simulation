"""A single offer within a negotiation."""

from pydantic import BaseModel, Field


class Offer(BaseModel):
    price: float
    currency_symbol: str
    chain_name: str
    round: int = Field(ge=0)

    def is_valid(self, supported_currencies: set[str]) -> bool:
        return self.price > 0 and self.currency_symbol in supported_currencies
