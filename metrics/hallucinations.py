"""Hallucination frequency/magnitude metrics.

Rule-based Phase 1 agents compute optimal choices deterministically, so
there is no "mistake" signal yet -- these query the (empty) hallucinations
table and return 0 / empty results. This becomes live once Phase 2's LLM
decisions start writing rows via src/llm/hallucination_detector.py.
"""

from sqlalchemy.orm import Session

from database.models import HallucinationRecord


def hallucination_frequency(session: Session, total_transactions: int) -> float:
    if total_transactions == 0:
        return 0.0
    hallucination_count = session.query(HallucinationRecord).count()
    return hallucination_count / total_transactions


def overpayment_by_currency(session: Session) -> dict[str, float]:
    records = session.query(HallucinationRecord).all()
    if not records:
        return {}
    totals: dict[str, list[float]] = {}
    for record in records:
        totals.setdefault(record.currency_symbol, []).append(record.overpayment_pct)
    return {symbol: sum(values) / len(values) for symbol, values in totals.items()}
