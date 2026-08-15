"""Launches the real (real LLM calls, real spend) full hypothesis-sandbox
study via run_hypothesis_matrix: all 11 hypotheses' 24 cells x all 3 utility
functions x the full ~99-model roster, 365 simulated days each.

This is a large, real-money, many-hour(+) run. Checkpointing is enabled so
a mid-run failure or interruption never loses already-completed days --
re-running this script resumes automatically (same matrix_run_id, same
database, same checkpoint_dir). To do a small, cheap sanity check BEFORE
committing to the full run instead, see the commented-out smoke-test
constants below.

Prerequisite: `DATABASE_URL` must be set correctly in the shell environment
before running this (the tracked .env's DATABASE_URL has a typo and must
not be relied on) -- e.g.:

    DATABASE_URL="sqlite:///./assip.db" .venv/bin/python scripts/launch_hypothesis_run.py
"""

import os

import yaml
from dotenv import load_dotenv

from database.session import create_all_tables, new_session
from src.economy.hypothesis_scenarios import baseline_cell_keys, build_hypothesis_cell_specs
from src.llm.llm_router import build_openrouter_client, get_cumulative_usage
from src.llm.market_intelligence import build_polygon_client
from src.simulation.hypothesis_matrix_runner import run_hypothesis_matrix
from src.utils.constants import CONFIG_ROOT, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

MATRIX_RUN_ID = "hyp-real-run-2026-08-14"  # change this for a new/different run -- reusing an id resumes it


def load_model_roster() -> list[str]:
    with open(CONFIG_ROOT / "llm" / "model_roster_full.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [m["id"] for m in data["models"]]


# Full real study: all 11 hypotheses' 24 cells x all 3 utility functions x
# the full model roster x 365 days. To do a small, cheap sanity check
# first instead, comment this block out and uncomment the smoke-test block
# below.
MODEL_CANDIDATES = load_model_roster()
NUM_DAYS = 365
SEEDS = [0]
HYPOTHESES = None  # None -> all 11 hypotheses' 24 cells
UTILITY_TYPES = None  # None -> all 3: crra, cara, epstein_zin_proxy
CELL_KEYS = None  # None -> every selected cell (baseline + cross-border + event)

# Baseline-only (New info.pdf's "Section 1: Baseline model" -- skips every
# cross-border and event-based variant): uncomment this one line.
# CELL_KEYS = baseline_cell_keys()

# Smoke-test scope (uncomment this block and comment out the real-study
# block above for a fast, cheap first check that everything works):
# MODEL_CANDIDATES = ["anthropic/claude-sonnet-5"]
# NUM_DAYS = 10
# SEEDS = [0]
# HYPOTHESES = ["H3"]
# UTILITY_TYPES = ["crra"]
# CELL_KEYS = None


def progress(cell_key: str, seed: int, utility_type: str, day: int) -> None:
    print(f"[progress] cell={cell_key} seed={seed} utility={utility_type} day={day}", flush=True)
    if day % 30 == 0:
        usage = get_cumulative_usage()
        print(f"    cumulative usage so far: {usage.total_tokens} tokens ({usage.prompt_tokens} prompt, "
              f"{usage.completion_tokens} completion)", flush=True)


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
        spec
        for spec in build_hypothesis_cell_specs()
        if (HYPOTHESES is None or spec.hypothesis in HYPOTHESES)
        and (CELL_KEYS is None or spec.key in CELL_KEYS)
    ]
    print(
        f"Launching run_hypothesis_matrix: matrix_run_id={MATRIX_RUN_ID} "
        f"hypotheses={HYPOTHESES or 'ALL'} cell_keys={CELL_KEYS or 'ALL'} ({len(selected_specs)} cells) "
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
        cell_keys=CELL_KEYS,
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
