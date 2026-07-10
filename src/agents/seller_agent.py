"""Seller-specific behavior: set prices, accept/reject offers."""

from src.agents.base_agent import BaseAgent


class SellerAgent(BaseAgent):
    def asking_price(self, true_price: float, markup: float = 0.10) -> float:
        """Rule-based opening ask: the good's true price plus a markup."""
        return true_price * (1.0 + markup)
