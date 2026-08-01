"""Tests for `build_sandbox_scenario` (src/economy/sandbox_scenarios.py) --
the per-sandbox synthetic shock schedule fix described in that module's
docstring: `master_simulation.yaml`'s currency-targeted shocks name only
real-universe symbols, so every one of them is silently inert for the 12
factor-isolation sandbox cells (whose currency universe is a synthetic
2-symbol pair). `build_sandbox_scenario` keeps the 5 macro-level shocks and
adds a crisis_warning/depeg_event H4 pair plus one additional
currency-targeted shock, both targeting one of that sandbox's own symbols.

Two different currency-mutation mechanisms are exercised deliberately
distinctly here, per `apply_currency_shock`'s own docstring
(src/economy/shocks.py): GOVERNANCE_DOWNGRADE/REGULATORY_ENFORCEMENT are
PERMANENT structural mutations applied via `apply_currency_shock` directly
on the `CurrencyConfig`; DEPEG_EVENT/CRISIS_WARNING/LIQUIDITY_CRUNCH are
TEMPORARY, decaying effects applied via `TrustLedger`'s offset channels
instead -- `apply_currency_shock` is a documented no-op for all three of
those (confirmed below). Tests for each additional_shock_type therefore
check the mechanism that shock type actually uses, not `apply_currency_shock`
unconditionally.
"""

import pytest

from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.economy.shocks import ShockType, apply_currency_shock, load_scenario
from src.economy.sandbox_scenarios import _SANDBOX_SHOCK_PLANS, build_sandbox_scenario
from src.economy.trust import TrustLedger, load_trust_params

_BASE_SCENARIO = load_scenario("master_simulation")


def _params():
    return load_trust_params()


def test_all_six_sandbox_plans_present():
    assert set(_SANDBOX_SHOCK_PLANS.keys()) == set(SANDBOX_CURRENCY_PAIRS.keys())


def test_build_sandbox_scenario_preserves_macro_backdrop_independently():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_governance"]
    scenario = build_sandbox_scenario("liquidity_vs_governance", option_a, option_b, _BASE_SCENARIO)

    assert scenario.duration_days == _BASE_SCENARIO.duration_days == 365
    assert scenario.initial_state == _BASE_SCENARIO.initial_state
    # model_copy(deep=True) must produce an independent initial_state object,
    # not an alias -- mutating one must never affect the other.
    assert scenario.initial_state is not _BASE_SCENARIO.initial_state
    scenario.initial_state.gold_price = 1.0
    assert _BASE_SCENARIO.initial_state.gold_price != 1.0


@pytest.mark.parametrize("sandbox_key", list(SANDBOX_CURRENCY_PAIRS.keys()))
def test_every_sandbox_builds_with_eight_shocks(sandbox_key):
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    scenario = build_sandbox_scenario(sandbox_key, option_a, option_b, _BASE_SCENARIO)

    # 5 macro-level shocks (no target_currency) + crisis_warning + depeg_event
    # + 1 additional currency-targeted shock == 8.
    assert len(scenario.shocks) == 8
    macro_shocks = [s for s in scenario.shocks if s.target_currency is None]
    assert len(macro_shocks) == 5


@pytest.mark.parametrize("sandbox_key", list(SANDBOX_CURRENCY_PAIRS.keys()))
def test_every_currency_targeted_shock_targets_one_of_the_sandboxs_own_symbols(sandbox_key):
    """No stray real-universe symbol (USDC, PAXG, ...) should ever appear --
    only this sandbox's own two synthetic symbols."""
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    own_symbols = {option_a.symbol, option_b.symbol}
    scenario = build_sandbox_scenario(sandbox_key, option_a, option_b, _BASE_SCENARIO)

    currency_targeted = [s for s in scenario.shocks if s.target_currency is not None]
    assert len(currency_targeted) == 3  # crisis_warning, depeg_event, additional
    for shock in currency_targeted:
        assert shock.target_currency in own_symbols


@pytest.mark.parametrize("sandbox_key", list(SANDBOX_CURRENCY_PAIRS.keys()))
def test_crisis_warning_depeg_pair_present_and_on_the_same_symbol(sandbox_key):
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    scenario = build_sandbox_scenario(sandbox_key, option_a, option_b, _BASE_SCENARIO)

    warnings = [s for s in scenario.shocks if s.type == ShockType.CRISIS_WARNING]
    depegs = [s for s in scenario.shocks if s.type == ShockType.DEPEG_EVENT]
    assert len(warnings) == 1
    assert len(depegs) == 1
    assert warnings[0].target_currency == depegs[0].target_currency
    assert depegs[0].day >= warnings[0].day


@pytest.mark.parametrize("sandbox_key", list(SANDBOX_CURRENCY_PAIRS.keys()))
def test_sandbox_shocks_are_non_confounding_with_master_macro_shocks(sandbox_key):
    """Every new shock day (crisis pair + additional) must be >=15 days clear
    of every macro-level shock day carried over from master_simulation.yaml
    (days 10/30/50/70/90), and of each other outside the deliberate
    crisis_warning/depeg_event pair itself."""
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    scenario = build_sandbox_scenario(sandbox_key, option_a, option_b, _BASE_SCENARIO)

    def is_paired(a, b):
        pair_types = {ShockType.CRISIS_WARNING, ShockType.DEPEG_EVENT}
        return {a.type, b.type} == pair_types and a.target_currency == b.target_currency

    shocks = scenario.shocks
    for i, a in enumerate(shocks):
        for b in shocks[i + 1 :]:
            if is_paired(a, b):
                continue
            assert abs(a.day - b.day) >= 15, f"{sandbox_key}: {a.type}@{a.day} and {b.type}@{b.day} too close"


@pytest.mark.parametrize("sandbox_key", list(SANDBOX_CURRENCY_PAIRS.keys()))
def test_crisis_warning_depeg_pair_actually_moves_trust_ledger_state(sandbox_key):
    """End-to-end proof the H4 pair is no longer inert: apply_currency_shock
    is a documented no-op for CRISIS_WARNING/DEPEG_EVENT (see
    src/economy/shocks.py's apply_currency_shock docstring -- these are
    TEMPORARY effects handled by TrustLedger's offset channels, not
    permanent CurrencyConfig mutations), so the real mutation mechanism to
    exercise here is TrustLedger.update, not apply_currency_shock."""
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    scenario = build_sandbox_scenario(sandbox_key, option_a, option_b, _BASE_SCENARIO)
    currencies = {option_a.symbol: option_a, option_b.symbol: option_b}

    depeg_shock = next(s for s in scenario.shocks if s.type == ShockType.DEPEG_EVENT)
    warning_shock = next(s for s in scenario.shocks if s.type == ShockType.CRISIS_WARNING)
    target = depeg_shock.target_currency

    # Confirm apply_currency_shock really is inert for these two types --
    # not just assumed. If this ever stops being true (the mechanism
    # changes), the test above documents exactly what changed.
    unchanged = apply_currency_shock(currencies, depeg_shock)
    assert unchanged[target] == currencies[target]

    ledger = TrustLedger(currencies, _params())
    baseline_peg_offset = ledger.peg_error_offset(target)
    ledger.update([warning_shock, depeg_shock])

    assert ledger.peg_error_offset(target) != baseline_peg_offset
    assert ledger.peg_error_offset(target) == pytest.approx(depeg_shock.magnitude)


@pytest.mark.parametrize("sandbox_key", list(SANDBOX_CURRENCY_PAIRS.keys()))
def test_additional_shock_genuinely_mutates_currency_or_trust_state(sandbox_key):
    """End-to-end proof the additional (second isolated-dimension) shock is
    no longer inert. GOVERNANCE_DOWNGRADE/REGULATORY_ENFORCEMENT are
    permanent structural mutations -> exercised via apply_currency_shock
    directly. LIQUIDITY_CRUNCH is a temporary/decaying effect ->
    apply_currency_shock is a documented no-op for it (confirmed below);
    the real mutation is TrustLedger's liquidity_offset channel instead."""
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    scenario = build_sandbox_scenario(sandbox_key, option_a, option_b, _BASE_SCENARIO)
    currencies = {option_a.symbol: option_a, option_b.symbol: option_b}
    plan = _SANDBOX_SHOCK_PLANS[sandbox_key]

    additional_shock = next(
        s
        for s in scenario.shocks
        if s.type not in {ShockType.CRISIS_WARNING, ShockType.DEPEG_EVENT} and s.target_currency is not None
    )
    target = additional_shock.target_currency

    if plan.additional_shock_type == ShockType.GOVERNANCE_DOWNGRADE:
        original = currencies[target].governance_score
        updated = apply_currency_shock(currencies, additional_shock)
        assert updated[target].governance_score < original
    elif plan.additional_shock_type == ShockType.REGULATORY_ENFORCEMENT:
        original = currencies[target].issuer_risk
        updated = apply_currency_shock(currencies, additional_shock)
        assert updated[target].issuer_risk > original
    elif plan.additional_shock_type == ShockType.LIQUIDITY_CRUNCH:
        # Confirm apply_currency_shock really is inert for this type.
        unchanged = apply_currency_shock(currencies, additional_shock)
        assert unchanged[target] == currencies[target]

        ledger = TrustLedger(currencies, _params())
        baseline_liquidity_offset = ledger.liquidity_offset(target)
        ledger.update([additional_shock])
        assert ledger.liquidity_offset(target) < baseline_liquidity_offset
    else:
        pytest.fail(f"Unexpected additional_shock_type for {sandbox_key}: {plan.additional_shock_type}")


def test_build_sandbox_scenario_raises_keyerror_for_unknown_sandbox_key():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_governance"]
    with pytest.raises(KeyError):
        build_sandbox_scenario("not_a_real_sandbox", option_a, option_b, _BASE_SCENARIO)
