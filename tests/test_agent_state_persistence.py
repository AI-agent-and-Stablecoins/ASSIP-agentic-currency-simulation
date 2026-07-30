from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, AgentStateRecord, AgentMemoryLogRecord
from database.repository import AgentStateLogEntry, AgentStateRepository, AgentMemoryLogEntry, AgentMemoryLogRepository


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
        cara_coefficient=1.5,
        real_purchasing_power=987.3,
        wallet_balances={"USDC": 800.0, "EURC": 200.0, "PAXG": 1.5},
        utility_score=0.72,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(AgentStateRecord).all()
    assert len(rows) == 1
    assert rows[0].wallet_balances == {"USDC": 800.0, "EURC": 200.0, "PAXG": 1.5}
    assert rows[0].cara_coefficient == 1.5


def test_agent_state_primary_key_is_run_timestep_agent():
    session = _session()
    repo = AgentStateRepository(session)
    repo.record(
        AgentStateLogEntry(
            run_id="run-a",
            timestep=1,
            agent_id="buyer-1",
            risk_profile="low",
            cara_coefficient=0.0,
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
            cara_coefficient=0.0,
            real_purchasing_power=990.0,
            wallet_balances={"USDC": 990.0},
            utility_score=0.99,
        )
    )
    session.commit()

    rows = session.query(AgentStateRecord).order_by(AgentStateRecord.timestep).all()
    assert len(rows) == 2
    assert [r.timestep for r in rows] == [1, 2]


def test_agent_memory_log_repository_persists_episodic_text():
    session = _session()
    repo = AgentMemoryLogRepository(session)
    entry = AgentMemoryLogEntry(
        run_id="run-master-seed-0",
        timestep=12,
        agent_id="buyer-1",
        memory_type="Depeg",
        memory_text="On day 12 I was mid-transaction in USDT when it depegged 8%.",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(AgentMemoryLogRecord).all()
    assert len(rows) == 1
    assert rows[0].memory_type == "Depeg"
    assert rows[0].memory_text == "On day 12 I was mid-transaction in USDT when it depegged 8%."


def test_agent_memory_log_repository_allows_multiple_entries_per_agent():
    session = _session()
    repo = AgentMemoryLogRepository(session)
    repo.record(
        AgentMemoryLogEntry(
            run_id="run-master-seed-0", timestep=5, agent_id="buyer-1", memory_type="Network",
            memory_text="USDC is currently accepted by 97% of local merchants.",
        )
    )
    repo.record(
        AgentMemoryLogEntry(
            run_id="run-master-seed-0", timestep=6, agent_id="buyer-1", memory_type="GasSpike",
            memory_text="Ethereum gas exploded to 180 Gwei in timestep 391.",
        )
    )
    session.commit()

    rows = session.query(AgentMemoryLogRecord).filter_by(agent_id="buyer-1").all()
    assert len(rows) == 2
