from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, AgentStateRecord
from database.repository import AgentStateLogEntry, AgentStateRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_agent_state_repository_persists_full_wallet_snapshot():
    session = _session()
    repo = AgentStateRepository(session)
    entry = AgentStateLogEntry(
        run_id="run-master-seed-0",
        timestep=10,
        agent_id="buyer-1",
        risk_profile="low",
        crra_sigma=1.5,
        real_purchasing_power=987.3,
        wallet_balances={"USDC": 800.0, "EURC": 200.0, "PAXG": 1.5},
        utility_score=0.72,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(AgentStateRecord).all()
    assert len(rows) == 1
    assert rows[0].wallet_balances == {"USDC": 800.0, "EURC": 200.0, "PAXG": 1.5}
    assert rows[0].crra_sigma == 1.5


def test_agent_state_primary_key_is_run_timestep_agent():
    session = _session()
    repo = AgentStateRepository(session)
    repo.record(
        AgentStateLogEntry(
            run_id="run-a",
            timestep=1,
            agent_id="buyer-1",
            risk_profile="low",
            crra_sigma=0.0,
            real_purchasing_power=1000.0,
            wallet_balances={"USDC": 1000.0},
            utility_score=1.0,
        )
    )
    repo.record(
        AgentStateLogEntry(
            run_id="run-a",
            timestep=2,
            agent_id="buyer-1",
            risk_profile="low",
            crra_sigma=0.0,
            real_purchasing_power=990.0,
            wallet_balances={"USDC": 990.0},
            utility_score=0.99,
        )
    )
    session.commit()

    rows = session.query(AgentStateRecord).order_by(AgentStateRecord.timestep).all()
    assert len(rows) == 2
    assert [r.timestep for r in rows] == [1, 2]
