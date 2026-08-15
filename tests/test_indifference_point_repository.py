import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import Base, IndifferencePointRecord
from database.repository import (
    IndifferencePointLogEntry,
    IndifferencePointRepository,
    SimulationRunLogEntry,
    SimulationRunRepository,
)


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_run(session: Session, run_id: str = "run-h3-seed0") -> None:
    SimulationRunRepository(session).record(
        SimulationRunLogEntry(
            run_id=run_id,
            scenario_name="master_simulation",
            research_mode="factual",
            random_seed=0,
            model_roster_summary="100 agents across 90 OpenRouter models",
            prompt_version_hash="deadbeef",
            git_commit_hash="abc1234",
            config_hash="feedface",
        )
    )
    session.commit()


def test_indifference_point_repository_persists_and_round_trips():
    session = _session()
    _seed_run(session)
    repo = IndifferencePointRepository(session)
    entry = IndifferencePointLogEntry(
        run_id="run-h3-seed0",
        hypothesis="H3",
        fixed_currency="USDC",
        varied_currency="EURC",
        varied_field="peg_stability",
        risk_aversion_cohort=0.5,
        compensation=0.03,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(IndifferencePointRecord).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == "run-h3-seed0"
    assert row.hypothesis == "H3"
    assert row.fixed_currency == "USDC"
    assert row.varied_currency == "EURC"
    assert row.varied_field == "peg_stability"
    assert row.risk_aversion_cohort == 0.5
    assert row.compensation == 0.03


def test_indifference_point_rejects_duplicate_composite_key():
    session = _session()
    _seed_run(session)
    repo = IndifferencePointRepository(session)
    repo.record(
        IndifferencePointLogEntry(
            run_id="run-h3-seed0",
            hypothesis="H3",
            fixed_currency="USDC",
            varied_currency="EURC",
            varied_field="peg_stability",
            risk_aversion_cohort=0.5,
            compensation=0.03,
        )
    )
    session.commit()

    repo.record(
        IndifferencePointLogEntry(
            run_id="run-h3-seed0",
            hypothesis="H3",
            fixed_currency="USDC",
            varied_currency="EURC",
            varied_field="peg_stability",
            risk_aversion_cohort=0.5,
            compensation=0.99,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
