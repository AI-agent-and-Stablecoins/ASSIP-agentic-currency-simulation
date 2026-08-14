from src.economy.hypothesis_scenarios import (
    CROSS_BORDER_HYPOTHESES,
    EVENT_BASED_HYPOTHESES,
    HYPOTHESIS_CURRENCIES,
    build_hypothesis_cell_specs,
)


def test_all_11_hypotheses_have_currency_definitions():
    assert set(HYPOTHESIS_CURRENCIES.keys()) == {f"H{i}" for i in range(1, 12)}


def test_h1_isolates_medium_of_exchange_alone_with_three_currencies():
    assert HYPOTHESIS_CURRENCIES["H1"] == ("USDC", "EURC", "PAXG")


def test_h2_isolates_governance_by_medium_of_exchange_with_six_currencies():
    assert set(HYPOTHESIS_CURRENCIES["H2"]) == {"USDC", "USDT", "EURC", "EURT", "PAXG", "XAUT"}


def test_two_currency_hypotheses_have_exactly_two_currencies():
    for hypothesis in ("H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"):
        assert len(HYPOTHESIS_CURRENCIES[hypothesis]) == 2


def test_total_cell_count_is_24():
    specs = build_hypothesis_cell_specs()
    assert len(specs) == 24


def test_11_baseline_cells_have_no_cross_border_or_event_shock():
    specs = build_hypothesis_cell_specs()
    baseline = [s for s in specs if not s.cross_border and s.event_shock is None]
    assert len(baseline) == 11
    assert {s.hypothesis for s in baseline} == {f"H{i}" for i in range(1, 12)}


def test_5_cross_border_cells_match_the_spec_priority_list():
    specs = build_hypothesis_cell_specs()
    cross_border = [s for s in specs if s.cross_border]
    assert len(cross_border) == 5
    assert {s.hypothesis for s in cross_border} == set(CROSS_BORDER_HYPOTHESES)
    assert set(CROSS_BORDER_HYPOTHESES) == {"H1", "H2", "H6", "H7", "H8"}


def test_8_event_based_cells_are_4_hypotheses_times_2_shocks():
    specs = build_hypothesis_cell_specs()
    event_based = [s for s in specs if s.event_shock is not None]
    assert len(event_based) == 8
    assert {s.hypothesis for s in event_based} == set(EVENT_BASED_HYPOTHESES)
    assert set(EVENT_BASED_HYPOTHESES) == {"H1", "H2", "H4", "H9"}
    shock_types = {s.event_shock for s in event_based}
    assert shock_types == {"depeg", "banking_crisis"}
    for hypothesis in EVENT_BASED_HYPOTHESES:
        this_hypothesis_shocks = {s.event_shock for s in event_based if s.hypothesis == hypothesis}
        assert this_hypothesis_shocks == {"depeg", "banking_crisis"}


def test_event_based_cells_target_a_currency_actually_in_that_hypothesis():
    specs = build_hypothesis_cell_specs()
    for spec in specs:
        if spec.event_shock is not None:
            assert spec.event_target_currency in spec.currencies


def test_gas_fee_hypotheses_have_chain_pins_covering_both_currencies():
    specs = build_hypothesis_cell_specs()
    for spec in specs:
        if spec.hypothesis in ("H5", "H8", "H10", "H11") and not spec.cross_border and spec.event_shock is None:
            assert spec.chain_pins is not None
            assert set(spec.chain_pins.keys()) == set(spec.currencies)
            assert set(spec.chain_pins.values()) == {"ethereum", "solana"}


def test_non_gas_fee_hypotheses_have_no_chain_pins():
    specs = build_hypothesis_cell_specs()
    for spec in specs:
        if spec.hypothesis not in ("H5", "H8", "H10", "H11"):
            assert spec.chain_pins is None
