import random

import pytest

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.currencies.currency import load_currency_universe
from src.economy.fx_tax import (
    FxParams,
    compute_counterparty_zone_tax,
    compute_fx_tax,
    currency_zone_of,
    load_fx_params,
)
from src.market.goods import Good
from src.simulation.environment import Environment
from src.simulation.timestep import run_timestep
from src.transactions.transaction import TransactionStatus


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


def test_compute_counterparty_zone_tax_is_zero_when_zones_match():
    tax = compute_counterparty_zone_tax(paid_value=1000.0, buyer_zone="USD", seller_zone="USD", fx_tax_rate=0.0002)
    assert tax == 0.0


def test_compute_counterparty_zone_tax_applies_when_zones_differ():
    tax = compute_counterparty_zone_tax(paid_value=1000.0, buyer_zone="USD", seller_zone="EUR", fx_tax_rate=0.0002)
    assert tax == pytest.approx(0.2)


def test_compute_counterparty_zone_tax_is_zero_when_buyer_zone_is_none():
    tax = compute_counterparty_zone_tax(paid_value=1000.0, buyer_zone=None, seller_zone="EUR", fx_tax_rate=0.0002)
    assert tax == 0.0


def test_compute_counterparty_zone_tax_is_zero_when_seller_zone_is_none():
    tax = compute_counterparty_zone_tax(paid_value=1000.0, buyer_zone="USD", seller_zone=None, fx_tax_rate=0.0002)
    assert tax == 0.0


def test_compute_counterparty_zone_tax_fires_even_for_a_zone_neutral_currency():
    """Unlike compute_fx_tax, this friction is about the COUNTERPARTY zone
    mismatch, not the settlement currency's own zone -- it must fire
    regardless of whether the settling currency itself is zone-neutral
    (e.g. gold-backed). This function doesn't take a currency at all, so
    that's inherent, but the test documents the intent explicitly."""
    tax = compute_counterparty_zone_tax(paid_value=1000.0, buyer_zone="USD", seller_zone="EUR", fx_tax_rate=0.0002)
    assert tax > 0.0


def _build_single_pair_env(buyer_zone: str, seller_zone: str) -> Environment:
    """One buyer, one seller, one good, settling exclusively in USDC (zone
    USD) -- deliberately small and fully deterministic (negotiate() and
    generate_candidates() take no rng) so the domestic-cell and
    cross-border-cell scenarios below differ ONLY in currency_zone, not in
    price or settlement currency."""
    profiles = load_agent_profiles()
    buyer = build_agent(profiles["consumer"], currency_zone=buyer_zone, agent_id="fx-tax-buyer")
    seller = build_agent(profiles["merchant"], currency_zone=seller_zone, agent_id="fx-tax-seller")
    buyer.wallet.balances = {"USDC": 100_000.0}
    seller.wallet.balances = {"USDC": 100_000.0}

    good = Good(name="widget", category="cloud_compute", base_price_usd=100.0)
    return Environment.build_from_population("master_simulation", [buyer, seller], goods=[good])


def test_cross_zone_counterparty_transaction_pays_strictly_more_fx_tax_than_same_zone():
    """Regression test for the matrix-runner cross-border-friction fix: an
    economically identical transaction (same currency, same negotiated
    price) must incur MORE fx_tax_paid when buyer and seller are in
    different currency_zones than when they share a zone -- otherwise a
    cross-border cell adds no friction beyond what a same-currency-zone-
    mismatched buyer already pays in a domestic cell (the bug this fix
    addresses)."""
    same_zone_env = _build_single_pair_env(buyer_zone="USD", seller_zone="USD")
    cross_zone_env = _build_single_pair_env(buyer_zone="USD", seller_zone="EUR")

    same_zone_result = run_timestep(same_zone_env, day=0, rng=random.Random(0))
    cross_zone_result = run_timestep(cross_zone_env, day=0, rng=random.Random(0))

    same_zone_settled = [tx for tx in same_zone_result.transactions if tx.status == TransactionStatus.SETTLED]
    cross_zone_settled = [tx for tx in cross_zone_result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(same_zone_settled) == 1
    assert len(cross_zone_settled) == 1

    same_zone_tx = same_zone_settled[0]
    cross_zone_tx = cross_zone_settled[0]

    # Same currency, same negotiated/settled price -- the only difference
    # between the two scenarios is the buyer/seller zone pairing.
    assert same_zone_tx.currency_symbol == cross_zone_tx.currency_symbol == "USDC"
    assert same_zone_tx.paid_value == pytest.approx(cross_zone_tx.paid_value)

    # USDC's own settlement-currency zone (USD) matches the buyer's zone
    # (USD) in BOTH scenarios, so compute_fx_tax alone is zero for both --
    # this isolates compute_counterparty_zone_tax as the sole source of the
    # difference below.
    assert same_zone_tx.fx_tax_paid == 0.0
    assert cross_zone_tx.fx_tax_paid > same_zone_tx.fx_tax_paid
    assert cross_zone_tx.fx_tax_paid == pytest.approx(cross_zone_tx.paid_value * 0.0002)
