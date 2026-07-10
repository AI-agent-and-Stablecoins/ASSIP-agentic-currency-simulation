import pytest

from src.llm.hallucination_detector import overpayment_pct


def test_overpayment_is_positive_percentage():
    assert overpayment_pct(100, 150) == pytest.approx(50.0)


def test_exact_payment_is_zero():
    assert overpayment_pct(100, 100) == pytest.approx(0.0)


def test_underpayment_is_negative_percentage():
    assert overpayment_pct(100, 80) == pytest.approx(-20.0)


def test_nonpositive_expected_value_raises():
    with pytest.raises(ValueError):
        overpayment_pct(0, 100)
