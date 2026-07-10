"""Tests: does GENIUS Act compliance affect adoption?"""

from src.currencies.currency import CurrencyConfig
from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def compliance_adoption_share(ledger: Ledger, currencies: dict[str, CurrencyConfig]) -> float:
    """Fraction of settled transaction volume routed through GENIUS-Act-compliant currencies."""
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    total = sum(tx.paid_value for tx in settled)
    if total == 0:
        return 0.0
    compliant_volume = sum(tx.paid_value for tx in settled if currencies[tx.currency_symbol].genius_compliant)
    return compliant_volume / total
