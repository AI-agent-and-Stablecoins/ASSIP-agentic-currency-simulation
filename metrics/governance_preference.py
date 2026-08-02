"""Measures: does governance quality affect agents' currency choices?"""

from src.currencies.currency import CurrencyConfig
from src.currencies.exchange_rates import ExchangeRateTable
from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def governance_preference(
    ledger: Ledger, currencies: dict[str, CurrencyConfig], exchange_rates: ExchangeRateTable
) -> float:
    """Volume-weighted average governance_score across settled transactions.

    tx.paid_value is native units of tx.currency_symbol, so each
    transaction's volume is converted to USD before being used as a weight --
    otherwise transactions in different currencies would be weighted by
    unit-scale artifacts rather than actual economic volume.
    """
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    volumes_usd = [exchange_rates.convert(tx.paid_value, tx.currency_symbol, "USD") for tx in settled]
    total = sum(volumes_usd)
    if total == 0:
        return 0.0
    return (
        sum(volume_usd * currencies[tx.currency_symbol].governance_score for tx, volume_usd in zip(settled, volumes_usd))
        / total
    )
