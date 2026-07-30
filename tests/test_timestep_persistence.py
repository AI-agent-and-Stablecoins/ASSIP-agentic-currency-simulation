from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, TimestepLogRecord
from database.repository import TimestepLogEntry, TimestepLogRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_timestep_log_repository_persists_daily_macro_state():
    session = _session()
    repo = TimestepLogRepository(session)
    entry = TimestepLogEntry(
        run_id="run-master-seed-0",
        timestep=5,
        inflation_rate=0.03,
        confidence_index=0.95,
        eth_gas_fee_gwei=25.0,
        solana_gas_fee_usd=0.0007,
        eur_usd_exchange_rate=1.08,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(TimestepLogRecord).all()
    assert len(rows) == 1
    assert rows[0].run_id == "run-master-seed-0"
    assert rows[0].timestep == 5
    assert rows[0].eur_usd_exchange_rate == 1.08


def test_timestep_log_primary_key_is_run_id_and_timestep():
    session = _session()
    repo = TimestepLogRepository(session)
    repo.record(
        TimestepLogEntry(
            run_id="run-a",
            timestep=1,
            inflation_rate=0.02,
            confidence_index=1.0,
            eth_gas_fee_gwei=20.0,
            solana_gas_fee_usd=0.0005,
            eur_usd_exchange_rate=1.08,
        )
    )
    repo.record(
        TimestepLogEntry(
            run_id="run-b",
            timestep=1,
            inflation_rate=0.05,
            confidence_index=0.8,
            eth_gas_fee_gwei=40.0,
            solana_gas_fee_usd=0.001,
            eur_usd_exchange_rate=1.07,
        )
    )
    session.commit()

    rows = session.query(TimestepLogRecord).order_by(TimestepLogRecord.run_id).all()
    assert len(rows) == 2
    assert [r.run_id for r in rows] == ["run-a", "run-b"]
