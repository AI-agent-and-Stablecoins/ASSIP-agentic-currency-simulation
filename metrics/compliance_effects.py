"""Tests: does GENIUS Act compliance affect adoption?"""

from src.currencies.currency import CurrencyConfig
from src.currencies.exchange_rates import ExchangeRateTable
from src.transactions.ledger import Ledger
from src.transactions.transaction import TransactionStatus


def compliance_adoption_share(
    ledger: Ledger, currencies: dict[str, CurrencyConfig], exchange_rates: ExchangeRateTable
) -> float:
    """Fraction of settled transaction volume routed through GENIUS-Act-compliant currencies.

    tx.paid_value is native units of tx.currency_symbol, so each
    transaction's volume is converted to USD before being summed --
    otherwise volume across different currencies (e.g. a gold-pegged
    currency vs. a USD-pegged one) would be mixed at wildly different unit
    scales rather than compared on an equal economic footing.
    """
    settled = [tx for tx in ledger.history() if tx.status == TransactionStatus.SETTLED]
    volumes_usd = [exchange_rates.convert(tx.paid_value, tx.currency_symbol, "USD") for tx in settled]
    total = sum(volumes_usd)
    if total == 0:
        return 0.0
    compliant_volume = sum(
        volume_usd
        for tx, volume_usd in zip(settled, volumes_usd)
        if currencies[tx.currency_symbol].genius_compliant
    )
    return compliant_volume / total
