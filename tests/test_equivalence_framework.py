from src.economy.equivalence_framework import EQUIVALENCE_COMPARISONS, EquivalenceComparison


def test_h2_has_exactly_two_comparisons():
    assert len(EQUIVALENCE_COMPARISONS["H2"]) == 2
    varied = {c.varied_currency for c in EQUIVALENCE_COMPARISONS["H2"]}
    assert varied == {"EURC", "PAXG"}


def test_every_other_hypothesis_has_exactly_one_comparison():
    for hypothesis in ("H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"):
        assert len(EQUIVALENCE_COMPARISONS[hypothesis]) == 1


def test_h1_has_no_comparisons():
    assert "H1" not in EQUIVALENCE_COMPARISONS


def test_h3_fixes_usdt_liquidity_and_varies_tdusd_liquidity():
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    assert comparison.fixed_currency == "USDT"
    assert comparison.varied_currency == "TDUSD"
    assert comparison.varied_field == "liquidity_score"
    assert comparison.bounds == (0.0, 1.0)


def test_gas_fee_comparisons_have_the_gas_fee_bounds():
    for hypothesis in ("H5", "H8", "H10", "H11"):
        comparison = EQUIVALENCE_COMPARISONS[hypothesis][0]
        assert comparison.varied_field == "gas_fee"
        assert comparison.bounds == (0.0, 5.0)


def test_h2_comparisons_vary_governance_score():
    for comparison in EQUIVALENCE_COMPARISONS["H2"]:
        assert comparison.varied_field == "governance_score"
        assert comparison.fixed_currency == "USDT"
