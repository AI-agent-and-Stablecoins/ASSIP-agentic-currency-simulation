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


def advance_price_index(price_index: float, annual_inflation_rate: float) -> float:
    """Compound the price index forward by one day's inflation.

    ``annual_inflation_rate`` is an ANNUAL rate (e.g. 0.02 for 2%/year, the
    convention used by scenario configs' ``inflation`` field). It is
    converted to a daily-equivalent rate before compounding so that a
    365-day run reconstructs the annual rate rather than compounding the
    annual rate once per day (which would turn a 2%/year "baseline" into
    ~1349x hyperinflation over a year).
    """
    daily_rate = (1 + annual_inflation_rate) ** (1 / 365) - 1
    return price_index * (1 + daily_rate)
