"""Actually moves money: debits the buyer, credits the seller, never creates or
destroys value (beyond the FX tax, which is a genuine leak out of the
buyer/seller pair -- see tx.fx_tax_paid below). Callers must validate
(src/transactions/validation.py) first."""

from src.agents.wallet import Wallet
from src.transactions.transaction import Transaction, TransactionStatus


def settle(tx: Transaction, buyer_wallet: Wallet, seller_wallet: Wallet) -> Transaction:
    # The buyer is debited the price AND the FX conversion tax (tx.fx_tax_paid,
    # 0.0 unless the settlement currency's zone differs from the buyer's --
    # see src/economy/fx_tax.py) in a single atomic withdraw call: either both
    # amounts are covered or the wallet is left completely untouched. Only the
    # price is credited to the seller -- the tax is not conserved between the
    # two parties, it leaves the transaction entirely (paid to an unmodeled
    # tax authority).
    total_debit = tx.paid_value + tx.fx_tax_paid
    withdrawn = buyer_wallet.withdraw(tx.currency_symbol, total_debit)
    if not withdrawn:
        tx.status = TransactionStatus.FAILED
        return tx
    seller_wallet.deposit(tx.currency_symbol, tx.paid_value)
    tx.status = TransactionStatus.SETTLED
    return tx
