"""Read-only live-progress queries against the project's own SQLite
database -- this is the dashboard's source of truth for "how far has each
cell/seed gotten", since it's shared between single-process (`run_matrix`)
and cross-process (`run_matrix_distributed`) runs alike, and needs zero
new callback plumbing in either. Every `run_id` this project ever writes
is `f"{matrix_run_id}-{cell_key}-seed{seed}"` (see
`src/simulation/matrix_runner.py`), which is parsed back apart here.
"""

from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import LLMDecisionRecord, TimestepLogRecord


class CellSeedProgress(BaseModel):
    cell_key: str
    seed: int
    run_id: str
    current_day: int
    total_llm_decisions: int


def _parse_run_id(run_id: str, matrix_run_id: str) -> tuple[str, int] | None:
    """Splits f"{matrix_run_id}-{cell_key}-seed{seed}" back into
    (cell_key, seed). Returns None if run_id doesn't start with this
    matrix_run_id's prefix (a different run_matrix invocation sharing the
    same database) or doesn't match the expected "-seed{N}" suffix shape."""
    prefix = f"{matrix_run_id}-"
    if not run_id.startswith(prefix):
        return None
    remainder = run_id[len(prefix) :]
    if "-seed" not in remainder:
        return None
    cell_key, _, seed_str = remainder.rpartition("-seed")
    if not cell_key or not seed_str.isdigit():
        return None
    return cell_key, int(seed_str)


def get_progress_for_run(session: Session, matrix_run_id: str) -> list[CellSeedProgress]:
    run_ids = [
        row[0]
        for row in session.query(TimestepLogRecord.run_id)
        .filter(TimestepLogRecord.run_id.like(f"{matrix_run_id}-%"))
        .distinct()
        .all()
    ]

    results = []
    for run_id in run_ids:
        parsed = _parse_run_id(run_id, matrix_run_id)
        if parsed is None:
            continue
        cell_key, seed = parsed

        max_day = session.query(func.max(TimestepLogRecord.timestep)).filter(
            TimestepLogRecord.run_id == run_id
        ).scalar()
        decision_count = session.query(func.count(LLMDecisionRecord.id)).filter(
            LLMDecisionRecord.simulation_id == run_id
        ).scalar()

        results.append(
            CellSeedProgress(
                cell_key=cell_key,
                seed=seed,
                run_id=run_id,
                current_day=max_day or 0,
                total_llm_decisions=decision_count or 0,
            )
        )

    return results
