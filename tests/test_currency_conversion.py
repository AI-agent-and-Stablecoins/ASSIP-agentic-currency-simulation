import pytest

from src.currencies.currency import load_currency_universe
from src.currencies.exchange_rates import ExchangeRateTable

PEG_RATES = {"USD": 1.0, "EUR": 1.08, "XAU": 2400.0}


def test_round_trip_conversion_preserves_value():
    currencies = load_currency_universe()
    rates = ExchangeRateTable(currencies, PEG_RATES)

    usd_amount = 100.0
    eur_amount = rates.convert(usd_amount, "USDC", "EURC")
    back_to_usd = rates.convert(eur_amount, "EURC", "USDC")

    assert back_to_usd == pytest.approx(usd_amount, rel=1e-9)


def test_conversion_through_third_asset_matches_direct_conversion():
    currencies = load_currency_universe()
    rates = ExchangeRateTable(currencies, PEG_RATES)

    direct = rates.convert(100.0, "USDC", "EURC")
    via_gold = rates.convert(rates.convert(100.0, "USDC", "PAXG"), "PAXG", "EURC")

    assert via_gold == pytest.approx(direct, rel=1e-9)


def test_same_currency_conversion_is_identity():
    currencies = load_currency_universe()
    rates = ExchangeRateTable(currencies, PEG_RATES)

    assert rates.convert(50.0, "USDC", "USDC") == pytest.approx(50.0)
