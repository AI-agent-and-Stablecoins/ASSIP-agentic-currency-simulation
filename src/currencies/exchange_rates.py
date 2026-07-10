"""Currency conversion derived from each currency's peg, not a hardcoded rate matrix.

Reference rates between the underlying pegs (e.g. USD, EUR, XAU) are supplied
by the caller -- typically sourced from the active scenario's macro state
(see src/economy/macro_state.py) -- rather than hardcoded here, so shocks in
Phase 3 can move them without touching this module.
"""

from src.currencies.currency import CurrencyConfig


class ExchangeRateTable:
    def __init__(self, currencies: dict[str, CurrencyConfig], peg_reference_rates: dict[str, float]):
        self._currencies = currencies
        self._peg_reference_rates = peg_reference_rates

    def _peg_value(self, symbol_or_peg: str) -> float:
        """Resolve a currency symbol (via its peg) or a raw peg unit (e.g. "USD") to a reference value."""
        currency = self._currencies.get(symbol_or_peg)
        if currency is not None:
            return self._peg_reference_rates[currency.peg or "USD"]
        return self._peg_reference_rates[symbol_or_peg]

    def get_rate(self, from_symbol: str, to_symbol: str) -> float:
        """Units of to_symbol received per one unit of from_symbol."""
        if from_symbol == to_symbol:
            return 1.0
        return self._peg_value(from_symbol) / self._peg_value(to_symbol)

    def convert(self, amount: float, from_symbol: str, to_symbol: str) -> float:
        return amount * self.get_rate(from_symbol, to_symbol)
