import pytest

from src.agents.wallet import Wallet
from src.currencies.currency import load_currency_universe
from src.transactions.settlement import settle
from src.transactions.transaction import Transaction, TransactionStatus
from src.transactions.validation import validate_transaction


def _make_tx(currency_symbol: str = "USDC", paid_value: float = 100.0) -> Transaction:
    return Transaction(
        buyer_id="buyer-1",
        seller_id="seller-1",
        good_name="cloud_compute",
        currency_symbol=currency_symbol,
        chain_name="ethereum",
        gas_fee=0.5,
        expected_value=100.0,
        paid_value=paid_value,
        timestep=0,
    )


def test_settlement_moves_exact_amount_and_conserves_value():
    buyer_wallet = Wallet(balances={"USDC": 1000.0})
    seller_wallet = Wallet(balances={"USDC": 0.0})
    tx = _make_tx()

    settle(tx, buyer_wallet, seller_wallet)

    assert tx.status == TransactionStatus.SETTLED
    assert buyer_wallet.balances["USDC"] == 900.0
    assert seller_wallet.balances["USDC"] == 100.0


def test_settlement_fails_gracefully_on_insufficient_funds():
    buyer_wallet = Wallet(balances={"USDC": 50.0})
    seller_wallet = Wallet(balances={})
    tx = _make_tx()

    settle(tx, buyer_wallet, seller_wallet)

    assert tx.status == TransactionStatus.FAILED
    assert buyer_wallet.balances["USDC"] == 50.0
    assert seller_wallet.balances.get("USDC", 0.0) == 0.0


def test_validation_rejects_insufficient_funds():
    currencies = load_currency_universe()
    buyer_wallet = Wallet(balances={"USDC": 50.0})
    tx = _make_tx()

    result = validate_transaction(tx, buyer_wallet, currencies)

    assert result.is_valid is False


def test_validation_rejects_unsupported_currency():
    currencies = load_currency_universe()
    buyer_wallet = Wallet(balances={"NOTACOIN": 1000.0})
    tx = _make_tx(currency_symbol="NOTACOIN")

    result = validate_transaction(tx, buyer_wallet, currencies)

    assert result.is_valid is False


def test_validation_accepts_well_formed_transaction():
    currencies = load_currency_universe()
    buyer_wallet = Wallet(balances={"USDC": 1000.0})
    tx = _make_tx()

    result = validate_transaction(tx, buyer_wallet, currencies)

    assert result.is_valid is True


def test_transaction_defaults_fx_tax_paid_to_zero():
    tx = _make_tx()

    assert tx.fx_tax_paid == 0.0


def test_transaction_accepts_explicit_fx_tax_paid():
    tx = Transaction(
        buyer_id="buyer-1",
        seller_id="seller-1",
        good_name="cloud_compute",
        currency_symbol="USDC",
        chain_name="ethereum",
        gas_fee=0.5,
        expected_value=100.0,
        paid_value=100.0,
        timestep=0,
        fx_tax_paid=2.5,
    )

    assert tx.fx_tax_paid == 2.5


def test_settle_debits_fx_tax_paid_in_addition_to_paid_value():
    # Construct a Transaction with a nonzero fx_tax_paid and confirm the buyer's
    # wallet balance drops by paid_value + fx_tax_paid, not just paid_value.
    buyer_wallet = Wallet(balances={"USDC": 1000.0})
    seller_wallet = Wallet(balances={"USDC": 0.0})
    tx = _make_tx(paid_value=100.0)
    tx.fx_tax_paid = 2.5

    settle(tx, buyer_wallet, seller_wallet)

    assert tx.status == TransactionStatus.SETTLED
    assert buyer_wallet.balances["USDC"] == pytest.approx(1000.0 - 100.0 - 2.5)
    # Only the price (not the tax) is credited to the seller -- the tax
    # leaves the buyer/seller pair entirely, it is not conserved between them.
    assert seller_wallet.balances["USDC"] == pytest.approx(100.0)


def test_settle_fails_gracefully_when_funds_cover_price_but_not_price_plus_fx_tax():
    buyer_wallet = Wallet(balances={"USDC": 100.0})
    seller_wallet = Wallet(balances={})
    tx = _make_tx(paid_value=100.0)
    tx.fx_tax_paid = 2.5

    settle(tx, buyer_wallet, seller_wallet)

    assert tx.status == TransactionStatus.FAILED
    # Withdrawal is atomic: an insufficient-funds failure must leave the
    # buyer's wallet completely untouched, not partially debited.
    assert buyer_wallet.balances["USDC"] == 100.0
    assert seller_wallet.balances.get("USDC", 0.0) == 0.0
