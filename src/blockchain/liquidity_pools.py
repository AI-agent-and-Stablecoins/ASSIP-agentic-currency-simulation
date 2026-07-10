"""Per-(currency, chain) liquidity overrides.

A currency's base liquidity_score (configs/currencies/*.yaml) is the
default; this registry lets specific chains be flagged as thinner or
deeper than that default without mutating the currency config itself.
Dependency-injected rather than a module-level dict, per the "no global
state" standard -- callers construct one and pass it where needed.
"""

from src.currencies.currency import CurrencyConfig


class LiquidityPoolRegistry:
    def __init__(self, overrides: dict[tuple[str, str], float] | None = None):
        self._overrides = dict(overrides or {})

    def register(self, currency_symbol: str, chain_name: str, liquidity_score: float) -> None:
        self._overrides[(currency_symbol, chain_name)] = liquidity_score

    def get_liquidity(self, currency: CurrencyConfig, chain_name: str) -> float:
        return self._overrides.get((currency.symbol, chain_name), currency.liquidity_score)
