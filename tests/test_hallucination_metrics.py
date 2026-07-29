"""Tests for hallucination metrics functions."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from database.repository import HallucinationLogEntry, HallucinationRepository
from metrics.hallucinations import hallucination_frequency, overpayment_by_currency


def _session() -> Session:
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_hallucination_frequency_with_no_records():
    """When there are no hallucination records, frequency should be 0."""
    session = _session()
    result = hallucination_frequency(session, total_transactions=100)
    assert result == 0.0


def test_hallucination_frequency_with_zero_transactions():
    """When total_transactions is 0, should return 0.0 to avoid division by zero."""
    session = _session()
    result = hallucination_frequency(session, total_transactions=0)
    assert result == 0.0


def test_hallucination_frequency_filters_by_is_hallucination_true():
    """hallucination_frequency should only count records where is_hallucination=True,
    not all records in the table."""
    session = _session()
    repo = HallucinationRepository(session)

    # Add 3 hallucination records (is_hallucination=True)
    for i in range(3):
        repo.record(
            HallucinationLogEntry(
                decision_id=f"dec-{i}",
                transaction_id=None,
                expected_price=100.0,
                paid_price=150.0,
                overpayment_pct=50.0,
                direction="OVERPAYMENT",
                is_hallucination=True,
                currency_symbol="USDC",
                model_name="anthropic/claude-sonnet-5",
            )
        )

    # Add 7 accurate decision records (is_hallucination=False)
    for i in range(3, 10):
        repo.record(
            HallucinationLogEntry(
                decision_id=f"dec-{i}",
                transaction_id=None,
                expected_price=100.0,
                paid_price=102.0,
                overpayment_pct=2.0,
                direction="ACCURATE",
                is_hallucination=False,
                currency_symbol="USDC",
                model_name="anthropic/claude-sonnet-5",
            )
        )

    session.commit()

    # Total records in table: 10 (3 hallucinations + 7 accurate)
    # But hallucination_frequency should only count the 3 true hallucinations
    result = hallucination_frequency(session, total_transactions=100)
    assert result == pytest.approx(0.03)  # 3 / 100


def test_hallucination_frequency_all_hallucinations():
    """When all records are hallucinations, frequency should be 1.0 (100%)."""
    session = _session()
    repo = HallucinationRepository(session)

    for i in range(5):
        repo.record(
            HallucinationLogEntry(
                decision_id=f"dec-{i}",
                transaction_id=None,
                expected_price=100.0,
                paid_price=200.0,
                overpayment_pct=100.0,
                direction="OVERPAYMENT",
                is_hallucination=True,
                currency_symbol="USDC",
                model_name="anthropic/claude-sonnet-5",
            )
        )

    session.commit()

    result = hallucination_frequency(session, total_transactions=5)
    assert result == pytest.approx(1.0)


def test_hallucination_frequency_no_hallucinations():
    """When all records are accurate (no hallucinations), frequency should be 0.0."""
    session = _session()
    repo = HallucinationRepository(session)

    for i in range(5):
        repo.record(
            HallucinationLogEntry(
                decision_id=f"dec-{i}",
                transaction_id=None,
                expected_price=100.0,
                paid_price=101.0,
                overpayment_pct=1.0,
                direction="ACCURATE",
                is_hallucination=False,
                currency_symbol="USDC",
                model_name="anthropic/claude-sonnet-5",
            )
        )

    session.commit()

    result = hallucination_frequency(session, total_transactions=100)
    assert result == pytest.approx(0.0)


def test_overpayment_by_currency_with_mixed_records():
    """overpayment_by_currency should average overpayment_pct across all records,
    not filtered by is_hallucination status."""
    session = _session()
    repo = HallucinationRepository(session)

    # Add records for USDC with different overpayment percentages
    repo.record(
        HallucinationLogEntry(
            decision_id="dec-1",
            transaction_id=None,
            expected_price=100.0,
            paid_price=120.0,
            overpayment_pct=20.0,
            direction="OVERPAYMENT",
            is_hallucination=True,
            currency_symbol="USDC",
            model_name="anthropic/claude-sonnet-5",
        )
    )

    repo.record(
        HallucinationLogEntry(
            decision_id="dec-2",
            transaction_id=None,
            expected_price=100.0,
            paid_price=110.0,
            overpayment_pct=10.0,
            direction="ACCURATE",
            is_hallucination=False,
            currency_symbol="USDC",
            model_name="anthropic/claude-sonnet-5",
        )
    )

    # Add record for EURC
    repo.record(
        HallucinationLogEntry(
            decision_id="dec-3",
            transaction_id=None,
            expected_price=100.0,
            paid_price=105.0,
            overpayment_pct=5.0,
            direction="ACCURATE",
            is_hallucination=False,
            currency_symbol="EURC",
            model_name="anthropic/claude-sonnet-5",
        )
    )

    session.commit()

    result = overpayment_by_currency(session)

    # USDC: average of (20.0 + 10.0) = 15.0
    # EURC: average of (5.0) = 5.0
    assert result["USDC"] == pytest.approx(15.0)
    assert result["EURC"] == pytest.approx(5.0)
