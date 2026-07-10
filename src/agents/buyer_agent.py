"""Buyer-specific behavior: decide what to buy, negotiate, select currency."""

from src.agents.base_agent import BaseAgent


class BuyerAgent(BaseAgent):
    def opening_offer_price(self, true_price: float, markup_tolerance: float = 0.05) -> float:
        """Rule-based starting offer: slightly below the good's true price."""
        return true_price * (1.0 - markup_tolerance)
