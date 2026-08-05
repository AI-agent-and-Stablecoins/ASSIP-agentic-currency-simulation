from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.simulation.distributed_matrix_runner import run_matrix_distributed


def test_run_matrix_distributed_runs_all_13_cells_across_processes(tmp_path):
    db_path = tmp_path / "distributed_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    results, failures = run_matrix_distributed(
        model_candidates=["vendor/fake-model"],
        seeds=[0],
        num_days=2,
        dry_run=True,
        num_processes=2,
        matrix_run_id="distributed-test",
        database_url=f"sqlite:///{db_path}",
    )

    assert failures == []
    assert len(results) == 13
    cell_keys_seen = {r.cell_key for r in results}
    assert len(cell_keys_seen) == 13
