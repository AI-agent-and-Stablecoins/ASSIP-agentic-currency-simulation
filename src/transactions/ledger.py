"""Append-only audit trail of every transaction -- the blockchain-equivalent record."""

from src.currencies.exchange_rates import ExchangeRateTable
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

    def total_settled_volume(self, exchange_rates: ExchangeRateTable) -> float:
        """Total settled volume, in USD.

        tx.paid_value is native units of tx.currency_symbol, so each
        transaction is converted to USD before summing -- otherwise summing
        across different currencies would mix unit scales (e.g. gold-pegged
        vs. USD-pegged) into a meaningless number.
        """
        return sum(
            exchange_rates.convert(tx.paid_value, tx.currency_symbol, "USD")
            for tx in self._records
            if tx.status == TransactionStatus.SETTLED
        )
