from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, TimestepLogRecord
from database.repository import TimestepLogEntry, TimestepLogRepository
from src.simulation.matrix_runner import run_matrix


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


def test_llm_path_produces_identical_transaction_count_before_and_after_refactor():
    """Regression baseline for Plan 6a Task 1: extracting the per-buyer LLM
    body into _process_buyer_llm_day must not change how many transactions
    a fixed-seed run produces."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    results, failures = run_matrix(
        model_candidates=["vendor/fake-model"],
        seeds=[0],
        num_days=3,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
        keep_daily_results=True,
    )
    assert failures == []
    master_result = next(r for r in results if r.cell_key == "master")
    # Fixed seed + fixed mock decision -> deterministic transaction count.
    # This is the count observed BEFORE Task 1's refactor (recorded via
    # Step 2); Step 4 re-runs this test after the refactor to confirm it is
    # unchanged.
    assert master_result.total_transactions == 420


def test_max_workers_greater_than_one_produces_same_transaction_count_as_sequential():
    """Concurrency must not change WHAT happens, only how fast -- same
    fixed seed/mock decision, sequential vs max_workers=4, must agree on
    total transaction count (order of dict/list entries may differ, but
    counts must not)."""
    def _run(max_workers):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = Session(engine)
        results, failures = run_matrix(
            model_candidates=["vendor/fake-model"],
            seeds=[0],
            num_days=3,
            dry_run=True,
            exercise_llm_path=True,
            session=session,
            keep_daily_results=True,
            llm_max_workers=max_workers,
        )
        assert failures == []
        return next(r for r in results if r.cell_key == "master").total_transactions

    assert _run(max_workers=1) == _run(max_workers=8)
