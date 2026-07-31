import pytest

from src.currencies.currency import load_currency_universe
from src.economy.fx_tax import FxParams, compute_fx_tax, currency_zone_of, load_fx_params


def test_load_fx_params_reads_the_real_config():
    params = load_fx_params()
    assert params.fx_tax_rate == 0.0002


def test_currency_zone_of_maps_usd_pegged_currencies():
    currencies = load_currency_universe()
    assert currency_zone_of(currencies["USDC"]) == "USD"


def test_currency_zone_of_maps_eur_pegged_currencies():
    currencies = load_currency_universe()
    assert currency_zone_of(currencies["EURC"]) == "EUR"


def test_currency_zone_of_is_none_for_gold_backed():
    currencies = load_currency_universe()
    assert currency_zone_of(currencies["PAXG"]) is None
    assert currency_zone_of(currencies["XAUT"]) is None


def test_compute_fx_tax_applies_when_buyer_zone_differs_from_currency_zone():
    currencies = load_currency_universe()
    tax = compute_fx_tax(paid_value=1000.0, currency=currencies["EURC"], buyer_zone="USD", fx_tax_rate=0.0002)
    assert tax == pytest.approx(0.2)


def test_compute_fx_tax_is_zero_when_zones_match():
    currencies = load_currency_universe()
    tax = compute_fx_tax(paid_value=1000.0, currency=currencies["USDC"], buyer_zone="USD", fx_tax_rate=0.0002)
    assert tax == 0.0


def test_compute_fx_tax_is_zero_for_gold_backed_regardless_of_buyer_zone():
    currencies = load_currency_universe()
    tax = compute_fx_tax(paid_value=1000.0, currency=currencies["PAXG"], buyer_zone="EUR", fx_tax_rate=0.0002)
    assert tax == 0.0


def test_compute_fx_tax_is_zero_when_buyer_zone_is_none():
    # An agent with no currency_zone assigned (e.g. legacy single-profile construction) never pays FX tax.
    currencies = load_currency_universe()
    tax = compute_fx_tax(paid_value=1000.0, currency=currencies["EURC"], buyer_zone=None, fx_tax_rate=0.0002)
    assert tax == 0.0
