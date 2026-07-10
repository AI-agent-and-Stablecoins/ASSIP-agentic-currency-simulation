"""Measures how much agents care about transaction fees."""

from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def average_gas_fee_paid(ledger: Ledger) -> float:
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    if not settled:
        return 0.0
    return sum(tx.gas_fee for tx in settled) / len(settled)
