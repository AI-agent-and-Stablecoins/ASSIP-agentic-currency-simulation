"""Tests: do agents sacrifice gas savings for liquidity?"""

from src.currencies.currency import CurrencyConfig
from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def liquidity_sensitivity(ledger: Ledger, currencies: dict[str, CurrencyConfig], threshold: float = 0.7) -> float:
    """Fraction of settled transaction volume that used a high-liquidity currency."""
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    if not settled:
        return 0.0
    high_liquidity = sum(1 for tx in settled if currencies[tx.currency_symbol].liquidity_score >= threshold)
    return high_liquidity / len(settled)
