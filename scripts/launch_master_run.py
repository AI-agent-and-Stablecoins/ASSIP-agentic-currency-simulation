"""Launches the real (dry_run=False) Phase 3 matrix run.

Scope per the 2026-08-04 deadline decision: full ~90-model roster
(configs/llm/model_roster_full.yaml), 1 seed, 14 simulated days.
Checkpointing is enabled so a mid-run failure or interruption never loses
already-completed days -- re-running this script resumes automatically
(same matrix_run_id, same database, same checkpoint_dir).
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from database.session import create_all_tables, new_session
from src.llm.llm_router import build_openrouter_client
from src.llm.market_intelligence import build_polygon_client
from src.simulation.matrix_runner import run_matrix
from src.utils.constants import CONFIG_ROOT, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

MATRIX_RUN_ID = "phase3-real-run-2026-08-04"
NUM_DAYS = 14
SEEDS = [0]


def load_model_roster() -> list[str]:
    with open(CONFIG_ROOT / "llm" / "model_roster_full.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [m["id"] for m in data["models"]]


def progress(cell_key: str, seed: int, day: int) -> None:
    print(f"[progress] cell={cell_key} seed={seed} day={day}", flush=True)


def main() -> None:
    create_all_tables()
    session = new_session()

    model_candidates = load_model_roster()
    print(f"Loaded {len(model_candidates)} candidate models from model_roster_full.yaml", flush=True)

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    polygon_key = os.getenv("POLYGON_API_KEY") or os.getenv("Polygon_API_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    if not polygon_key:
        raise RuntimeError("POLYGON_API_KEY is not set in .env")

    openrouter_client = build_openrouter_client(openrouter_key)
    polygon_client = build_polygon_client(polygon_key)

    checkpoint_dir = REPO_ROOT / "checkpoints" / MATRIX_RUN_ID

    print(
        f"Launching run_matrix: matrix_run_id={MATRIX_RUN_ID} seeds={SEEDS} "
        f"num_days={NUM_DAYS} checkpoint_dir={checkpoint_dir}",
        flush=True,
    )

    results, failures = run_matrix(
        model_candidates=model_candidates,
        seeds=SEEDS,
        num_days=NUM_DAYS,
        dry_run=False,
        openrouter_client=openrouter_client,
        polygon_client=polygon_client,
        session=session,
        matrix_run_id=MATRIX_RUN_ID,
        checkpoint_dir=checkpoint_dir,
        progress_callback=progress,
        keep_daily_results=False,
    )

    print("=== RUN COMPLETE ===", flush=True)
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
