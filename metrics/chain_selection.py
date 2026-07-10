"""Tracks blockchain usage share (Ethereum, Base, Solana, Arbitrum, ...)."""

from collections import Counter

from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def chain_usage_share(ledger: Ledger) -> dict[str, float]:
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    if not settled:
        return {}
    counts = Counter(tx.chain_name for tx in settled)
    total = len(settled)
    return {chain: count / total for chain, count in counts.items()}
