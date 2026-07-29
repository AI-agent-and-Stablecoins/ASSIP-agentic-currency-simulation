from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, SimulationRunRecord
from database.repository import SimulationRunLogEntry, SimulationRunRepository


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
