"""Regression tests: post-settlement-fix (d8f6568), Transaction.paid_value is
native units of tx.currency_symbol, not USD. Five aggregators used to
sum/weight paid_value directly across DIFFERENT currencies with no
conversion -- for a gold-pegged currency (~2400 USD/unit) vs. a USD-pegged
one, this produced a ~2400x unit-scale artifact masquerading as an economic
signal (e.g. two economically-identical $100 trades, one per currency,
would show market_share ~= {"USDC": 0.9996, "XAUT": 0.00042} instead of the
correct {"USDC": 0.5, "XAUT": 0.5}).

Each test below constructs two settled transactions with equal USD value
(100.0 each) in USDC (USD-pegged, genius_compliant=True,
governance_score=0.95) and XAUT (gold-pegged, genius_compliant=False,
governance_score=0.65) and confirms the now-currency-neutral aggregate.
"""

import pytest

from metrics.compliance_effects import compliance_adoption_share
from metrics.currency_usage import market_share
from metrics.governance_preference import governance_preference
from metrics.transaction_stats import average_transaction_value
from src.currencies.currency import load_currency_universe
from src.currencies.exchange_rates import ExchangeRateTable
from src.transactions.ledger import Ledger
from src.transactions.transaction import Transaction, TransactionStatus

PEG_RATES = {"USD": 1.0, "XAU": 2400.0}


def _rates() -> ExchangeRateTable:
    return ExchangeRateTable(load_currency_universe(), PEG_RATES)


def _equal_usd_value_ledger() -> Ledger:
    """One $100 trade in USDC (USD-pegged), one economically-identical $100
    trade in XAUT (gold-pegged, ~2400 USD/unit) -- 100.0 native USDC units
    vs. ~0.041667 native XAUT units for the same USD value.
    """
    ledger = Ledger()
    ledger.record(
        Transaction(
            buyer_id="buyer-1",
            seller_id="seller-1",
            good_name="good",
            currency_symbol="USDC",
            chain_name="ethereum",
            gas_fee=0.5,
            expected_value=100.0,
            paid_value=100.0,
            timestep=0,
            status=TransactionStatus.SETTLED,
        )
    )
    ledger.record(
        Transaction(
            buyer_id="buyer-2",
            seller_id="seller-2",
            good_name="good",
            currency_symbol="XAUT",
            chain_name="ethereum",
            gas_fee=0.5,
            expected_value=100.0,
            paid_value=100.0 / 2400.0,
            timestep=0,
            status=TransactionStatus.SETTLED,
        )
    )
    return ledger


def test_market_share_is_currency_neutral_across_gold_and_usd_pegged_currencies():
    ledger = _equal_usd_value_ledger()

    shares = market_share(ledger, _rates())

    assert shares["USDC"] == pytest.approx(0.5, rel=1e-6)
    assert shares["XAUT"] == pytest.approx(0.5, rel=1e-6)


def test_compliance_adoption_share_is_currency_neutral_across_gold_and_usd_pegged_currencies():
    ledger = _equal_usd_value_ledger()
    currencies = load_currency_universe()

    # USDC is genius_compliant=True, XAUT is genius_compliant=False, and each
    # contributes an equal 100 USD of volume -- the compliant share of total
    # USD volume must be 0.5, not ~0.9996 (native-unit-skewed toward USDC).
    share = compliance_adoption_share(ledger, currencies, _rates())

    assert share == pytest.approx(0.5, rel=1e-6)


def test_governance_preference_is_currency_neutral_across_gold_and_usd_pegged_currencies():
    ledger = _equal_usd_value_ledger()
    currencies = load_currency_universe()

    # Equal USD-value weighting -> simple average of the two governance
    # scores (0.95, 0.65), not a native-unit-skewed value dominated by USDC.
    result = governance_preference(ledger, currencies, _rates())

    assert result == pytest.approx((0.95 + 0.65) / 2, rel=1e-6)


def test_average_transaction_value_is_currency_neutral_across_gold_and_usd_pegged_currencies():
    ledger = _equal_usd_value_ledger()

    result = average_transaction_value(ledger, _rates())

    assert result == pytest.approx(100.0, rel=1e-6)


def test_ledger_total_settled_volume_is_currency_neutral_across_gold_and_usd_pegged_currencies():
    ledger = _equal_usd_value_ledger()

    result = ledger.total_settled_volume(_rates())

    assert result == pytest.approx(200.0, rel=1e-6)
