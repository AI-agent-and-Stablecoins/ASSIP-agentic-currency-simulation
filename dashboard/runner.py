"""Subprocess entrypoint wrapping run_matrix/run_matrix_distributed.

Invoked as `python -m dashboard.runner <args>` by
`dashboard/process_control.py`. Writes live status updates to
`dashboard/status_store` as it runs: this is the ONLY place token-usage
and final-completion state get reported (live progress -- day counts,
transaction/decision counts -- is read directly from the database by
`dashboard/queries.py` instead, needing no cooperation from this script
at all).
"""

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

from dashboard import status_store
from src.utils.constants import CONFIG_ROOT, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")


def _load_model_candidates() -> list[str]:
    """Full real-model roster for a real launch; a single placeholder for
    a dry run (dry_run's rule-based path never calls the LLM router at
    all unless exercise_llm_path is also set, which this dashboard never
    does, so the actual model ID value is irrelevant there)."""
    with open(CONFIG_ROOT / "llm" / "model_roster_full.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [m["id"] for m in data["models"]]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--cell-keys", default=None, help="Comma-separated; omit for all 13 cells")
    parser.add_argument("--seeds", required=True, help="Comma-separated integers")
    parser.add_argument("--num-days", required=True, type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num-processes", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    matrix_run_id = args.matrix_run_id
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    cell_keys = [c.strip() for c in args.cell_keys.split(",")] if args.cell_keys else None
    dry_run = args.dry_run
    checkpoint_dir = REPO_ROOT / "checkpoints" / matrix_run_id

    status_store.write_status(
        matrix_run_id,
        pid=os.getpid(),
        state="running",
        dry_run=dry_run,
        checkpoint_dir=str(checkpoint_dir),
        failures=None,
        error=None,
    )
    status_store.set_active_run_id(matrix_run_id)

    from src.llm.llm_router import LLMUsage

    def _write_usage(usage) -> None:
        status_store.write_status(
            matrix_run_id,
            cumulative_usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )

    _distributed_usage_total = LLMUsage()

    def _distributed_usage_callback(group_usage) -> None:
        # run_matrix_distributed's usage_callback receives ONE WORKER
        # GROUP's own final total per call, not a running grand total
        # across all groups (each worker process has its own separate
        # token counter -- see that function's own docstring) -- accumulate
        # here so the status file always reflects everything completed so
        # far, not just whichever group finished most recently.
        nonlocal _distributed_usage_total
        _distributed_usage_total = _distributed_usage_total + group_usage
        _write_usage(_distributed_usage_total)

    def _single_process_usage_callback(cell_key, seed, day, usage) -> None:
        # `usage` here is already this process's own running cumulative
        # total (run_matrix's usage_callback contract passes
        # src.llm.llm_router.get_cumulative_usage()'s snapshot directly),
        # so it can be written as-is with no extra accumulation needed.
        _write_usage(usage)

    try:
        model_candidates = _load_model_candidates()

        if args.distributed:
            from database.session import get_engine

            from src.simulation.distributed_matrix_runner import run_matrix_distributed

            openrouter_client_factory = None
            polygon_client_factory = None
            if not dry_run:
                openrouter_client_factory, polygon_client_factory = _real_client_factories()

            results, failures = run_matrix_distributed(
                model_candidates=model_candidates,
                seeds=seeds,
                num_days=args.num_days,
                database_url=str(get_engine().url),
                dry_run=dry_run,
                openrouter_client_factory=openrouter_client_factory,
                polygon_client_factory=polygon_client_factory,
                matrix_run_id=matrix_run_id,
                num_processes=args.num_processes,
                checkpoint_dir=checkpoint_dir,
                usage_callback=_distributed_usage_callback,
            )
        else:
            from database.session import create_all_tables, new_session

            from src.simulation.matrix_runner import run_matrix

            create_all_tables()
            session = new_session()

            openrouter_client = None
            polygon_client = None
            if not dry_run:
                openrouter_client, polygon_client = _real_clients()

            results, failures = run_matrix(
                model_candidates=model_candidates,
                seeds=seeds,
                num_days=args.num_days,
                dry_run=dry_run,
                openrouter_client=openrouter_client,
                polygon_client=polygon_client,
                session=session,
                matrix_run_id=matrix_run_id,
                cell_keys=cell_keys,
                checkpoint_dir=checkpoint_dir,
                usage_callback=_single_process_usage_callback,
            )
    except Exception as exc:  # noqa: BLE001 -- must always record a final
        # status rather than leaving "running" stale if launch/setup itself
        # blows up before run_matrix's own per-cell handling ever engages.
        status_store.write_status(matrix_run_id, state="failed", error=f"{type(exc).__name__}: {exc}")
        raise

    status_store.write_status(
        matrix_run_id,
        state="completed",
        failures=[[cell_key, seed, str(exc)] for cell_key, seed, exc in failures],
    )


def _real_clients():
    from src.llm.llm_router import build_openrouter_client
    from src.llm.market_intelligence import build_polygon_client

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    polygon_key = os.getenv("POLYGON_API_KEY") or os.getenv("Polygon_API_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    if not polygon_key:
        raise RuntimeError("POLYGON_API_KEY is not set in .env")
    return build_openrouter_client(openrouter_key), build_polygon_client(polygon_key)


def _real_client_factories():
    def openrouter_client_factory():
        return _real_clients()[0]

    def polygon_client_factory():
        return _real_clients()[1]

    return openrouter_client_factory, polygon_client_factory


if __name__ == "__main__":
    main()
