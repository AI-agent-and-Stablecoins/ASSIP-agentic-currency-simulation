"""Fast, fully-mocked pipeline-demo run -- NOT real data.

Runs the full 13-cell matrix with dry_run=True, exercise_llm_path=True
(mocked OpenRouter/Polygon clients, no real API calls, no spend) for
250 simulated days -- long enough for master's H4 crisis/depeg pairs
(first pair at day 210) to actually fall inside the window. Writes to
a SEPARATE sqlite file so it never touches the real
phase3-real-run-2026-08-04 data in assip.db.

Purpose: prove the econometrics pipeline (H1-H5) produces valid
regression output end-to-end. This is a demo of the CODE, not a
research result -- every decision here is a canned mock response, not
a real LLM's reasoning.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.simulation.matrix_runner import run_matrix
from src.utils.constants import REPO_ROOT

MATRIX_RUN_ID = "phase3-demo-run-2026-08-04"
NUM_DAYS = 250
SEEDS = [0]
DEMO_DB_PATH = REPO_ROOT / "demo_pipeline_run.db"


def progress(cell_key: str, seed: int, day: int) -> None:
    if day % 25 == 0 or day == NUM_DAYS - 1:
        print(f"[progress] cell={cell_key} seed={seed} day={day}", flush=True)


def main() -> None:
    if DEMO_DB_PATH.exists():
        DEMO_DB_PATH.unlink()

    engine = create_engine(f"sqlite:///{DEMO_DB_PATH}")
    Base.metadata.create_all(engine)
    session = Session(engine)

    print(f"Launching demo run_matrix: matrix_run_id={MATRIX_RUN_ID} num_days={NUM_DAYS} db={DEMO_DB_PATH}", flush=True)

    results, failures = run_matrix(
        model_candidates=["demo/mock-model"],
        seeds=SEEDS,
        num_days=NUM_DAYS,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
        matrix_run_id=MATRIX_RUN_ID,
        progress_callback=progress,
        keep_daily_results=False,
    )

    print("=== DEMO RUN COMPLETE ===", flush=True)
    for r in results:
        print(
            f"{r.cell_key} seed={r.seed}: {r.num_days_completed} days, "
            f"{r.total_transactions} tx, {r.total_llm_decisions} decisions",
            flush=True,
        )
    if failures:
        print("=== FAILURES ===", flush=True)
        for cell_key, seed, exc in failures:
            print(f"{cell_key} seed={seed}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
