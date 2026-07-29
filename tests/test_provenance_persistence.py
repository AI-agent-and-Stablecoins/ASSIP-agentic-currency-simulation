from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, SimulationRunRecord, InterventionLogRecord
from database.repository import SimulationRunLogEntry, SimulationRunRepository, InterventionLogEntry, InterventionLogRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_simulation_run_repository_persists_provenance():
    session = _session()
    repo = SimulationRunRepository(session)
    entry = SimulationRunLogEntry(
        run_id="run-master-seed-0",
        scenario_name="master_simulation",
        research_mode="factual",
        random_seed=0,
        model_roster_summary="100 agents across 90 OpenRouter models",
        prompt_version_hash="deadbeef",
        git_commit_hash="abc1234",
        config_hash="feedface",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(SimulationRunRecord).all()
    assert len(rows) == 1
    assert rows[0].run_id == "run-master-seed-0"
    assert rows[0].research_mode == "factual"
    assert rows[0].random_seed == 0


def test_intervention_log_repository_persists_shock_event():
    session = _session()
    repo = InterventionLogRepository(session)
    entry = InterventionLogEntry(
        run_id="run-master-seed-0",
        timestep=212,
        shock_type="inflation",
        target_currency=None,
        target_issuer=None,
        magnitude=0.085,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(InterventionLogRecord).all()
    assert len(rows) == 1
    assert rows[0].shock_type == "inflation"
    assert rows[0].timestep == 212


def test_intervention_log_repository_persists_targeted_shock():
    session = _session()
    repo = InterventionLogRepository(session)
    entry = InterventionLogEntry(
        run_id="run-master-seed-0",
        timestep=610,
        shock_type="depeg_event",
        target_currency="USDT",
        target_issuer=None,
        magnitude=0.08,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(InterventionLogRecord).all()
    assert rows[0].target_currency == "USDT"
