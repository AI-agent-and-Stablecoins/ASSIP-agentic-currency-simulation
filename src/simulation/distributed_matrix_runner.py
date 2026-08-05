"""Cross-process orchestrator partitioning the 13-cell matrix across
worker processes (Plan 6a Sec 2.2). Each worker process runs its own
disjoint subset of cell keys via `run_matrix(cell_keys=...)`, opening its
OWN database session against the SAME SQLite file (WAL mode, enabled in
`database/session.py`, is what makes concurrent-process writes to that
one shared file safe). httpx.Client objects are not picklable, so a real
client cannot cross the process boundary directly -- callers needing
dry_run=False pass factory callables instead, and each worker calls the
factory itself after the process starts.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from src.simulation.matrix_runner import MatrixCellResult, _build_cell_specs, run_matrix


def _partition(items: list, num_groups: int) -> list[list]:
    """Splits `items` into `num_groups` roughly-equal contiguous chunks
    (never more groups than items -- a group that would be empty is
    dropped, since spawning a process with zero work is pure overhead)."""
    num_groups = min(num_groups, len(items)) or 1
    chunk_size = -(-len(items) // num_groups)  # ceiling division
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_cell_group(
    cell_keys: list[str],
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    dry_run: bool,
    database_url: str,
    matrix_run_id: str,
    llm_max_workers: int,
    checkpoint_dir: Path | None,
    openrouter_client_factory: Callable[[], "httpx.Client"] | None,
    polygon_client_factory: Callable[[], "httpx.Client"] | None,
) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]]]:
    """Runs in a separate process: builds its OWN engine/session (engines
    aren't picklable/shareable across processes either) and its own real
    clients from the factories, if given, then calls run_matrix restricted
    to this group's cell_keys."""
    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, connection_record) -> None:
        if not database_url.startswith("sqlite"):
            return
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    session = Session(engine)
    openrouter_client = openrouter_client_factory() if openrouter_client_factory is not None else None
    polygon_client = polygon_client_factory() if polygon_client_factory is not None else None

    return run_matrix(
        model_candidates=model_candidates,
        seeds=seeds,
        num_days=num_days,
        dry_run=dry_run,
        openrouter_client=openrouter_client,
        polygon_client=polygon_client,
        session=session,
        matrix_run_id=matrix_run_id,
        cell_keys=cell_keys,
        llm_max_workers=llm_max_workers,
        checkpoint_dir=checkpoint_dir,
    )


def run_matrix_distributed(
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    database_url: str,
    dry_run: bool = True,
    openrouter_client_factory: Callable[[], "httpx.Client"] | None = None,
    polygon_client_factory: Callable[[], "httpx.Client"] | None = None,
    matrix_run_id: str | None = None,
    num_processes: int = 4,
    llm_max_workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]]]:
    """Partitions the 13 matrix cells into `num_processes` groups and runs
    each group in its own OS process via `run_matrix(cell_keys=...)`,
    against the same `database_url` (must be a file-based SQLite URL, or
    another DB that supports concurrent-process writes). Each worker
    process opens its OWN SQLAlchemy engine/session against `database_url`
    -- engines, like httpx.Client, are not picklable across the process
    boundary. `database/session.py`'s WAL-mode connect-event listener is
    registered against that module's specific `_engine` instance, not
    `database_url` globally, so it has no effect on engines built inside
    worker processes; `_run_cell_group` above registers the same pragmas
    on its own locally-built engine instead.

    `matrix_run_id`, if `None`, is generated once here (not per-group) so
    every cell across every process shares one consistent prefix -- see
    `run_matrix`'s own `matrix_run_id` docstring for why a stable shared
    prefix matters for resumability.
    """
    from src.utils.helpers import generate_id

    if matrix_run_id is None:
        matrix_run_id = generate_id("matrix")

    all_cell_keys = [spec.key for spec in _build_cell_specs()]
    groups = _partition(all_cell_keys, num_processes)

    with ProcessPoolExecutor(max_workers=len(groups)) as executor:
        futures = [
            executor.submit(
                _run_cell_group,
                group,
                model_candidates,
                seeds,
                num_days,
                dry_run,
                database_url,
                matrix_run_id,
                llm_max_workers,
                checkpoint_dir,
                openrouter_client_factory,
                polygon_client_factory,
            )
            for group in groups
        ]
        all_results: list[MatrixCellResult] = []
        all_failures: list[tuple[str, int, Exception]] = []
        for future in futures:
            group_results, group_failures = future.result()
            all_results.extend(group_results)
            all_failures.extend(group_failures)

    return all_results, all_failures
