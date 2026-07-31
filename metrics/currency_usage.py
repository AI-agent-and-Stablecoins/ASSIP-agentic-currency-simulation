"""Tracks currency market share by settled transaction volume."""

from collections import Counter

from src.currencies.exchange_rates import ExchangeRateTable
from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def market_share(ledger: Ledger, exchange_rates: ExchangeRateTable) -> dict[str, float]:
    """Volume-weighted share of settled transactions per currency.

    tx.paid_value is native units of tx.currency_symbol (see the
    settlement-currency-conversion fix, d8f6568), so summing/comparing it
    directly across DIFFERENT currencies without conversion would let a
    unit-scale artifact (e.g. ~2400x for a gold-pegged currency vs. a
    USD-pegged one) masquerade as an adoption signal. Each transaction's
    volume is converted to USD before being weighted/summed.
    """
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    volumes_usd: Counter[str] = Counter()
    for tx in settled:
        volumes_usd[tx.currency_symbol] += exchange_rates.convert(tx.paid_value, tx.currency_symbol, "USD")
    total = sum(volumes_usd.values())
    if total == 0:
        return {}
    return {symbol: volume / total for symbol, volume in volumes_usd.items()}
