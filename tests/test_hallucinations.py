import pytest

from src.llm.hallucination_detector import (
    HallucinationDirection,
    detect_hallucination,
    overpayment_pct,
)


def test_overpayment_is_positive_percentage():
    assert overpayment_pct(100, 150) == pytest.approx(50.0)


def test_exact_payment_is_zero():
    assert overpayment_pct(100, 100) == pytest.approx(0.0)


def test_underpayment_is_negative_percentage():
    assert overpayment_pct(100, 80) == pytest.approx(-20.0)


def test_nonpositive_expected_value_raises():
    with pytest.raises(ValueError):
        overpayment_pct(0, 100)


def test_detect_hallucination_classifies_overpayment_above_threshold():
    result = detect_hallucination(100.0, 150.0, hallucination_threshold=0.20)

    assert result.direction == HallucinationDirection.OVERPAYMENT
    assert result.absolute_error == pytest.approx(50.0)
    assert result.percentage_error == pytest.approx(50.0)


def test_detect_hallucination_classifies_underpayment_above_threshold():
    result = detect_hallucination(100.0, 50.0, hallucination_threshold=0.20)

    assert result.direction == HallucinationDirection.UNDERPAYMENT


def test_detect_hallucination_classifies_small_deviation_as_accurate():
    result = detect_hallucination(100.0, 105.0, hallucination_threshold=0.20)

    assert result.direction == HallucinationDirection.ACCURATE


def test_detect_hallucination_threshold_is_configurable_not_hardcoded():
    lenient = detect_hallucination(100.0, 115.0, hallucination_threshold=0.20)
    strict = detect_hallucination(100.0, 115.0, hallucination_threshold=0.10)

    assert lenient.direction == HallucinationDirection.ACCURATE
    assert strict.direction == HallucinationDirection.OVERPAYMENT


def test_detect_hallucination_carries_correlated_fields():
    result = detect_hallucination(
        100.0,
        150.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        actual_model="anthropic/claude-sonnet-5",
        agent_type="buyer",
        risk_profile="low",
        economic_scenario="baseline",
    )

    assert result.currency_symbol == "USDC"
    assert result.actual_model == "anthropic/claude-sonnet-5"
