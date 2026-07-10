"""Pre-settlement checks: sufficient funds, accepted currency, valid transaction."""

from pydantic import BaseModel

from src.agents.wallet import Wallet
from src.currencies.currency import CurrencyConfig
from src.transactions.transaction import Transaction


class TransactionValidationResult(BaseModel):
    is_valid: bool
    reason: str | None = None


def validate_transaction(
    tx: Transaction,
    buyer_wallet: Wallet,
    currencies: dict[str, CurrencyConfig],
) -> TransactionValidationResult:
    if tx.currency_symbol not in currencies:
        return TransactionValidationResult(is_valid=False, reason=f"Unsupported currency: {tx.currency_symbol}")
    if tx.paid_value <= 0:
        return TransactionValidationResult(is_valid=False, reason="Paid value must be positive")
    balance = buyer_wallet.balances.get(tx.currency_symbol, 0.0)
    if balance < tx.paid_value:
        return TransactionValidationResult(is_valid=False, reason="Insufficient funds")
    return TransactionValidationResult(is_valid=True)
