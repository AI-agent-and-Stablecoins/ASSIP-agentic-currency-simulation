import pytest

from src.currencies.currency import load_currency_universe
from src.economy.macro_state import MacroState
from src.economy.shocks import ShockEvent, ShockType, apply_currency_shock, apply_shock


def test_shock_type_has_all_twelve_members():
    expected = {
        "inflation", "bank_failure", "gold_rally", "fee_spike",
        "regulatory_enforcement", "liquidity_crunch", "governance_downgrade",
        "depeg_event", "crisis_warning", "fx_volatility_shock", "fx_rate_shock",
        "capital_controls",
    }
    assert {member.value for member in ShockType} == expected


def test_shock_event_accepts_target_currency_and_issuer():
    shock = ShockEvent(day=5, type=ShockType.GOVERNANCE_DOWNGRADE, magnitude=0.2, target_currency="USDT")
    assert shock.target_currency == "USDT"
    assert shock.target_issuer is None
    assert shock.decay_days is None


def test_governance_downgrade_permanently_lowers_governance_score():
    currencies = load_currency_universe()
    original = currencies["USDT"].governance_score
    shock = ShockEvent(day=0, type=ShockType.GOVERNANCE_DOWNGRADE, magnitude=0.2, target_currency="USDT")

    updated = apply_currency_shock(currencies, shock)

    assert updated["USDT"].governance_score == pytest.approx(max(0.0, original - 0.2))
    assert updated["USDC"].governance_score == currencies["USDC"].governance_score  # untouched
    assert currencies["USDT"].governance_score == original  # original dict/config untouched


def test_governance_downgrade_clamps_at_zero():
    currencies = load_currency_universe()
    shock = ShockEvent(day=0, type=ShockType.GOVERNANCE_DOWNGRADE, magnitude=5.0, target_currency="USDT")

    updated = apply_currency_shock(currencies, shock)

    assert updated["USDT"].governance_score == 0.0


def test_regulatory_enforcement_spikes_issuer_risk_permanently():
    currencies = load_currency_universe()
    original = currencies["USDT"].issuer_risk
    shock = ShockEvent(day=0, type=ShockType.REGULATORY_ENFORCEMENT, magnitude=0.3, target_currency="USDT")

    updated = apply_currency_shock(currencies, shock)

    assert updated["USDT"].issuer_risk == pytest.approx(min(1.0, original + 0.3))


def test_apply_currency_shock_is_a_noop_for_non_currency_shocks():
    currencies = load_currency_universe()
    shock = ShockEvent(day=0, type=ShockType.INFLATION, magnitude=0.05)

    updated = apply_currency_shock(currencies, shock)

    assert updated == currencies


def test_crisis_warning_applies_a_small_confidence_dip():
    state = MacroState(confidence_index=1.0)
    shock = ShockEvent(day=0, type=ShockType.CRISIS_WARNING, magnitude=0.05)

    updated = apply_shock(state, shock)

    assert updated.confidence_index == pytest.approx(0.95)
    assert state.confidence_index == 1.0  # original untouched


def test_fx_rate_shock_moves_eur_reference_rate():
    state = MacroState()
    original_eur = state.peg_reference_rates["EUR"]
    shock = ShockEvent(day=0, type=ShockType.FX_RATE_SHOCK, magnitude=0.1)

    updated = apply_shock(state, shock)

    assert updated.peg_reference_rates["EUR"] == pytest.approx(original_eur * 1.1)


def test_bank_failure_still_drops_confidence_with_optional_target_issuer():
    state = MacroState(confidence_index=1.0)
    shock = ShockEvent(day=0, type=ShockType.BANK_FAILURE, magnitude=0.3, target_issuer="Circle")

    updated = apply_shock(state, shock)

    assert updated.confidence_index == pytest.approx(0.7)
