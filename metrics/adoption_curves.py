"""Market share trajectory over time -- does USD dominate, does gold gain
share during inflation, how long until preferences stabilize."""

from metrics.currency_usage import market_share
from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def adoption_curve_up_to_day(ledger: Ledger, up_to_day: int) -> dict[str, float]:
    filtered = Ledger()
    for tx in ledger.history():
        if tx.timestep <= up_to_day and tx.status == TransactionStatus.SETTLED:
            filtered.record(tx)
    return market_share(filtered)


def adoption_curve_series(ledger: Ledger, num_days: int) -> dict[int, dict[str, float]]:
    return {day: adoption_curve_up_to_day(ledger, day) for day in range(num_days)}
