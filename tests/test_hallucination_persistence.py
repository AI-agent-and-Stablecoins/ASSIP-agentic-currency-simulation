from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, HallucinationRecord
from database.repository import HallucinationLogEntry, HallucinationRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_hallucination_repository_persists_without_a_transaction():
    """A hallucination can be detected on a raw LLM decision before any
    transaction settles -- transaction_id must be optional."""
    session = _session()
    repo = HallucinationRepository(session)
    entry = HallucinationLogEntry(
        decision_id="dec-1",
        transaction_id=None,
        expected_price=100.0,
        paid_price=150.0,
        overpayment_pct=50.0,
        direction="OVERPAYMENT",
        is_hallucination=True,
        currency_symbol="USDC",
        model_name="anthropic/claude-sonnet-5",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(HallucinationRecord).all()
    assert len(rows) == 1
    assert rows[0].transaction_id is None
    assert rows[0].decision_id == "dec-1"
    assert rows[0].direction == "OVERPAYMENT"
    assert rows[0].is_hallucination is True


def test_hallucination_repository_persists_accurate_decisions_too():
    """Accurate (non-hallucinated) decisions are recorded too, with
    is_hallucination=False -- the table is a complete telemetry record, not
    just a log of failures."""
    session = _session()
    repo = HallucinationRepository(session)
    entry = HallucinationLogEntry(
        decision_id="dec-2",
        transaction_id="tx-1",
        expected_price=100.0,
        paid_price=102.0,
        overpayment_pct=2.0,
        direction="ACCURATE",
        is_hallucination=False,
        currency_symbol="EURC",
        model_name="openai/gpt-5.6-luna",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(HallucinationRecord).all()
    assert rows[0].is_hallucination is False
    assert rows[0].transaction_id == "tx-1"
