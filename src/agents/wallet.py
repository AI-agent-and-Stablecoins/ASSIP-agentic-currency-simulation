"""Tracks per-currency balances for an agent."""

from pydantic import BaseModel, Field

from src.currencies.exchange_rates import ExchangeRateTable
from src.utils.helpers import round_currency


class Wallet(BaseModel):
    balances: dict[str, float] = Field(default_factory=dict)

    def deposit(self, symbol: str, amount: float) -> None:
        if amount < 0:
            raise ValueError("Cannot deposit a negative amount")
        self.balances[symbol] = round_currency(self.balances.get(symbol, 0.0) + amount)

    def withdraw(self, symbol: str, amount: float) -> bool:
        """Return False (leaving the wallet untouched) if funds are insufficient."""
        if amount < 0:
            raise ValueError("Cannot withdraw a negative amount")
        current = self.balances.get(symbol, 0.0)
        if current < amount:
            return False
        self.balances[symbol] = round_currency(current - amount)
        return True

    def total_value_usd(self, rates: ExchangeRateTable) -> float:
        return sum(rates.convert(amount, symbol, "USD") for symbol, amount in self.balances.items() if amount)
