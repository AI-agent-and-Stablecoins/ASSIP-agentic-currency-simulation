"""Append-only audit trail of every transaction -- the blockchain-equivalent record."""

from src.transactions.transaction import Transaction, TransactionStatus


class Ledger:
    def __init__(self):
        self._records: list[Transaction] = []

    def record(self, tx: Transaction) -> None:
        self._records.append(tx)

    def history(self, agent_id: str | None = None) -> list[Transaction]:
        if agent_id is None:
            return list(self._records)
        return [tx for tx in self._records if tx.buyer_id == agent_id or tx.seller_id == agent_id]

    def total_settled_volume(self) -> float:
        return sum(tx.paid_value for tx in self._records if tx.status == TransactionStatus.SETTLED)
