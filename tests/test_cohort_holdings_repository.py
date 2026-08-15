import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import Base, CohortHoldingsRecord
from database.repository import CohortHoldingsLogEntry, CohortHoldingsRepository, SimulationRunLogEntry, SimulationRunRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_run(session: Session, run_id: str = "run-h1-seed0") -> None:
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


def test_cohort_holdings_repository_persists_and_round_trips():
    session = _session()
    _seed_run(session)
    repo = CohortHoldingsRepository(session)
    entry = CohortHoldingsLogEntry(
        run_id="run-h1-seed0",
        risk_aversion_cohort=0.5,
        currency_symbol="USDC",
        pct_of_wealth=0.42,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(CohortHoldingsRecord).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == "run-h1-seed0"
    assert row.risk_aversion_cohort == 0.5
    assert row.currency_symbol == "USDC"
    assert row.pct_of_wealth == 0.42


def test_cohort_holdings_rejects_duplicate_composite_key():
    session = _session()
    _seed_run(session)
    repo = CohortHoldingsRepository(session)
    repo.record(
        CohortHoldingsLogEntry(
            run_id="run-h1-seed0",
            risk_aversion_cohort=0.5,
            currency_symbol="USDC",
            pct_of_wealth=0.42,
        )
    )
    session.commit()

    repo.record(
        CohortHoldingsLogEntry(
            run_id="run-h1-seed0",
            risk_aversion_cohort=0.5,
            currency_symbol="USDC",
            pct_of_wealth=0.99,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
