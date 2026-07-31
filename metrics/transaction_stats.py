"""Transaction volume/count/success-rate and negotiation-length statistics."""

from src.currencies.exchange_rates import ExchangeRateTable
from src.negotiation.conversation_history import ConversationLog
from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def transaction_success_rate(ledger: Ledger) -> float:
    records = ledger.history()
    if not records:
        return 0.0
    settled = sum(1 for tx in records if tx.status == TransactionStatus.SETTLED)
    return settled / len(records)


def average_transaction_value(ledger: Ledger, exchange_rates: ExchangeRateTable) -> float:
    """Mean settled transaction volume, in USD.

    tx.paid_value is native units of tx.currency_symbol, so each
    transaction is converted to USD before averaging -- otherwise averaging
    across different currencies would mix unit scales (e.g. gold-pegged vs.
    USD-pegged) into a meaningless number.
    """
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    if not settled:
        return 0.0
    return sum(exchange_rates.convert(tx.paid_value, tx.currency_symbol, "USD") for tx in settled) / len(settled)


def negotiation_length(logs: list[ConversationLog]) -> dict[str, float]:
    if not logs:
        return {"average_rounds": 0.0, "max_rounds": 0.0}
    lengths = [len(log.offers) for log in logs]
    return {"average_rounds": sum(lengths) / len(lengths), "max_rounds": float(max(lengths))}
