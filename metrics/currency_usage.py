"""Tracks currency market share by settled transaction volume."""

from collections import Counter

from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def market_share(ledger: Ledger) -> dict[str, float]:
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    total = sum(tx.paid_value for tx in settled)
    if total == 0:
        return {}
    volumes: Counter[str] = Counter()
    for tx in settled:
        volumes[tx.currency_symbol] += tx.paid_value
    return {symbol: volume / total for symbol, volume in volumes.items()}
