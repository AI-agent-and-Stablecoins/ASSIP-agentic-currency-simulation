import pytest

from src.currencies.currency import load_currency_universe
from src.economy.shocks import ShockEvent, ShockType
from src.economy.trust import TrustLedger, TrustParams, load_trust_params
from src.utils.helpers import clamp


def _params() -> TrustParams:
    return TrustParams(lambda_shock=0.5, lambda_recover=0.03, lambda_contagion=0.1, rolling_window_days=30)


def test_trust_ledger_initializes_at_governance_score():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())

    assert ledger.trust_score("USDC") == pytest.approx(currencies["USDC"].governance_score)
    assert ledger.trust_score("USDT") == pytest.approx(currencies["USDT"].governance_score)


def test_trust_ledger_quiet_day_recovers_toward_baseline():
    currencies = load_currency_universe()
    params = _params()
    ledger = TrustLedger(currencies, params)
    baseline = currencies["USDT"].governance_score

    # Manually depress USDT's trust via an event day, then let quiet days recover it.
    ledger.update([ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.8, target_currency="USDT")])
    depressed = ledger.trust_score("USDT")
    assert depressed < baseline

    for _ in range(5):
        ledger.update([])

    recovered = ledger.trust_score("USDT")
    assert recovered > depressed
    assert recovered < baseline  # partial recovery only, lambda_recover=0.03 is slow


def test_load_trust_params_reads_the_real_config():
    params = load_trust_params()

    assert params.lambda_shock == 0.5
    assert params.lambda_recover == 0.03
    assert params.lambda_contagion == 0.1
    assert params.rolling_window_days == 30


def test_depeg_event_spikes_and_decays_peg_error_offset():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())

    ledger.update([ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")])
    spiked = ledger.peg_error_offset("USDT")
    assert spiked == pytest.approx(0.08)
    assert ledger.effective_peg_error("USDT", currencies["USDT"].peg_error) == pytest.approx(
        currencies["USDT"].peg_error + spiked
    )

    for _ in range(10):
        ledger.update([])

    decayed = ledger.peg_error_offset("USDT")
    assert 0.0 < decayed < spiked


def test_liquidity_crunch_drops_and_recovers_liquidity_offset():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())

    ledger.update([ShockEvent(day=0, type=ShockType.LIQUIDITY_CRUNCH, magnitude=0.3, target_currency="USDC")])
    dropped = ledger.liquidity_offset("USDC")
    assert dropped < 0.0
    effective = ledger.effective_liquidity_score("USDC", currencies["USDC"].liquidity_score)
    assert effective == pytest.approx(clamp(currencies["USDC"].liquidity_score + dropped, 0.0, 1.0))

    for _ in range(10):
        ledger.update([])

    recovered = ledger.liquidity_offset("USDC")
    assert dropped < recovered < 0.0


def test_currencies_untouched_by_offset_shocks_have_zero_offset():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())

    ledger.update([ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")])

    assert ledger.peg_error_offset("USDC") == 0.0
    assert ledger.liquidity_offset("USDC") == 0.0
