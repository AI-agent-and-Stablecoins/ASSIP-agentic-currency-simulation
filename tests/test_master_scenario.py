"""Tests for the 365-day master simulation scenario (Phase 3 Plan 4 Task 12).

Covers: full-year duration, the H4 crisis-proximity sweep (crisis_warning ->
depeg_event pairs at 0/5/10/20-day gaps, each pair on a distinct currency),
and presence of all 12 ShockType members at least once.
"""

from src.economy.shocks import ShockType, load_scenario


def test_master_simulation_scenario_loads_and_spans_365_days():
    scenario = load_scenario("master_simulation")
    assert scenario.duration_days == 365


def test_master_simulation_has_h4_proximity_sweep_at_four_gap_values():
    scenario = load_scenario("master_simulation")
    warnings = [s for s in scenario.shocks if s.type == ShockType.CRISIS_WARNING]
    depegs = [s for s in scenario.shocks if s.type == ShockType.DEPEG_EVENT]
    # For each of the 4 gap values, confirm a crisis_warning/depeg_event pair
    # targeting the same currency exists at that exact day-gap.
    #
    # NOTE: the brief's illustrative version of this test used a strict
    # `d.day > warning.day` filter, which can never produce a gap of 0 (day
    # equality is excluded by construction -- d.day - warning.day == 0
    # contradicts d.day > warning.day). Using `>=` so the same-day (0-day
    # gap) pair is actually detected, matching the design spec's intent
    # (design spec Sec 9: "0, 5, 10, 20-day gaps").
    gaps_found = set()
    for warning in warnings:
        matching_depeg = next(
            (d for d in depegs if d.target_currency == warning.target_currency and d.day >= warning.day), None
        )
        if matching_depeg is not None:
            gaps_found.add(matching_depeg.day - warning.day)
    assert gaps_found == {0, 5, 10, 20}


def test_master_simulation_proximity_pairs_target_distinct_currencies():
    scenario = load_scenario("master_simulation")
    warnings = [s for s in scenario.shocks if s.type == ShockType.CRISIS_WARNING]
    currencies = {w.target_currency for w in warnings}
    assert len(currencies) == 4  # each of the 4 gap pairs uses a different currency


def test_master_simulation_includes_every_shock_type_at_least_once():
    scenario = load_scenario("master_simulation")
    types_present = {s.type for s in scenario.shocks}
    assert types_present == set(ShockType)


def test_master_simulation_shocks_are_non_confounding_in_time():
    """Every shock day must be >=15 days from every other shock's day,
    except within a deliberate crisis_warning/depeg_event proximity pair
    (matched by same target_currency), per design spec Sec 9."""
    scenario = load_scenario("master_simulation")
    shocks = scenario.shocks

    def is_paired(a, b):
        pair_types = {ShockType.CRISIS_WARNING, ShockType.DEPEG_EVENT}
        return (
            {a.type, b.type} == pair_types
            and a.target_currency == b.target_currency
            and a.target_currency is not None
        )

    for i, a in enumerate(shocks):
        for b in shocks[i + 1 :]:
            if is_paired(a, b):
                continue
            assert abs(a.day - b.day) >= 15, f"{a.type}@{a.day} and {b.type}@{b.day} are too close"
