"""Actually moves money: debits the buyer, credits the seller, never creates or
destroys value. Callers must validate (src/transactions/validation.py) first."""

from src.agents.wallet import Wallet
from src.transactions.transaction import Transaction, TransactionStatus


def settle(tx: Transaction, buyer_wallet: Wallet, seller_wallet: Wallet) -> Transaction:
    withdrawn = buyer_wallet.withdraw(tx.currency_symbol, tx.paid_value)
    if not withdrawn:
        tx.status = TransactionStatus.FAILED
        return tx
    seller_wallet.deposit(tx.currency_symbol, tx.paid_value)
    tx.status = TransactionStatus.SETTLED
    return tx
