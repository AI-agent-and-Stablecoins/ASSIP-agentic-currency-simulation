"""Measures: does governance quality affect agents' currency choices?"""

from src.currencies.currency import CurrencyConfig
from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def governance_preference(ledger: Ledger, currencies: dict[str, CurrencyConfig]) -> float:
    """Volume-weighted average governance_score across settled transactions."""
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    total = sum(tx.paid_value for tx in settled)
    if total == 0:
        return 0.0
    return sum(tx.paid_value * currencies[tx.currency_symbol].governance_score for tx in settled) / total
