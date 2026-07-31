"""Real (inflation-adjusted) purchasing power helpers.

A wallet's nominal USD value doesn't reflect what it can actually buy once
prices rise; dividing by a running price index yields the real value. This
is the foundation for loss-driven CARA risk-aversion adaptation (Phase 3
Plan 4, Task 7): agents react to changes in real purchasing power, not raw
nominal balances.
"""

from src.agents.wallet import Wallet
from src.currencies.exchange_rates import ExchangeRateTable


def real_purchasing_power(wallet: Wallet, rates: ExchangeRateTable, price_index: float) -> float:
    """Nominal USD wallet value deflated by the current price index."""
    return wallet.total_value_usd(rates) / price_index


def advance_price_index(price_index: float, inflation_rate: float) -> float:
    """Compound the price index forward by one day's inflation rate."""
    return price_index * (1 + inflation_rate)
