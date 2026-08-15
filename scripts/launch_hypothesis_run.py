"""Launches a real (real LLM calls, real spend) hypothesis-sandbox run via
run_hypothesis_matrix.

Defaults to a SMALL, cheap smoke-test scope (one hypothesis, one utility
function, one model, 10 days) so a first run is fast and inexpensive to
verify everything works end-to-end. To run the real study (all 11
hypotheses' 24 cells x 3 utility functions x 365 days), edit the constants
below -- see the comments next to each one.

Prerequisite: `DATABASE_URL` must be set correctly in the shell environment
before running this (the tracked .env's DATABASE_URL has a typo and must
not be relied on) -- e.g.:

    DATABASE_URL="sqlite:///./assip.db" .venv/bin/python scripts/launch_hypothesis_run.py
"""

import os

from dotenv import load_dotenv

from database.session import create_all_tables, new_session
from src.economy.hypothesis_scenarios import build_hypothesis_cell_specs
from src.llm.llm_router import build_openrouter_client
from src.llm.market_intelligence import build_polygon_client
from src.simulation.hypothesis_matrix_runner import run_hypothesis_matrix
from src.utils.constants import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

MATRIX_RUN_ID = "hyp-smoke-test-2026-08-14"  # change this for a new/real run -- reusing an id resumes it

# Smoke-test scope. For the real study: HYPOTHESES = None (all 11), NUM_DAYS
# = 365, UTILITY_TYPES = None (all 3: crra/cara/epstein_zin_proxy), and
# MODEL_CANDIDATES = the full roster (configs/llm/model_roster_full.yaml,
# same pattern as launch_master_run.py's load_model_roster()).
MODEL_CANDIDATES = ["anthropic/claude-sonnet-5"]
NUM_DAYS = 10
SEEDS = [0]
HYPOTHESES = ["H3"]  # None -> all 11 hypotheses' 24 cells
UTILITY_TYPES = ["crra"]  # None -> all 3: crra, cara, epstein_zin_proxy


def progress(cell_key: str, seed: int, utility_type: str, day: int) -> None:
    print(f"[progress] cell={cell_key} seed={seed} utility={utility_type} day={day}", flush=True)


def main() -> None:
    create_all_tables()
    session = new_session()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    openrouter_client = build_openrouter_client(openrouter_key)

    polygon_key = os.getenv("POLYGON_API_KEY") or os.getenv("Polygon_API_KEY")
    polygon_client = build_polygon_client(polygon_key) if polygon_key else None

    checkpoint_dir = REPO_ROOT / "checkpoints" / MATRIX_RUN_ID

    selected_specs = [
        spec for spec in build_hypothesis_cell_specs() if HYPOTHESES is None or spec.hypothesis in HYPOTHESES
    ]
    print(
        f"Launching run_hypothesis_matrix: matrix_run_id={MATRIX_RUN_ID} "
        f"hypotheses={HYPOTHESES or 'ALL'} ({len(selected_specs)} cells) "
        f"utility_types={UTILITY_TYPES or 'ALL'} seeds={SEEDS} num_days={NUM_DAYS} "
        f"checkpoint_dir={checkpoint_dir}",
        flush=True,
    )

    results, failures = run_hypothesis_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=SEEDS,
        num_days=NUM_DAYS,
        openrouter_client=openrouter_client,
        polygon_client=polygon_client,
        session=session,
        matrix_run_id=MATRIX_RUN_ID,
        utility_types=UTILITY_TYPES,
        hypotheses=HYPOTHESES,
        progress_callback=progress,
        checkpoint_dir=checkpoint_dir,
    )

    print("=== RUN COMPLETE ===", flush=True)
    for r in results:
        print(
            f"{r.cell_key} utility={r.utility_type} seed={r.seed}: {r.num_days_completed} days, "
            f"{r.total_transactions} tx, {r.total_llm_decisions} decisions",
            flush=True,
        )
    if failures:
        print("=== FAILURES ===", flush=True)
        for cell_key, seed, utility_type, exc in failures:
            print(f"{cell_key} utility={utility_type} seed={seed}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
