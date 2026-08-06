"""Cross-process orchestrator partitioning the 13-cell x N-seed matrix
across worker processes (Plan 6a Sec 2.2). Each worker process runs its
own disjoint subset of (cell_key, seed) pairs via `run_matrix(cell_keys=
[...], seeds=[...])` (one call per pair, since `run_matrix` itself always
runs the full cross product of its cell_keys x seeds arguments -- calling
it once per pair is what lets an arbitrary, evenly-sized subset of the
cross product go to each worker), opening its OWN database session
against the SAME SQLite file (WAL mode, enabled in `database/session.py`,
is what makes concurrent-process writes to that one shared file safe).
httpx.Client objects are not picklable, so a real client cannot cross the
process boundary directly -- callers needing dry_run=False pass factory
callables instead, and each worker calls the factory itself after the
process starts.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from src.llm.llm_router import LLMUsage
from src.simulation.matrix_runner import MatrixCellResult, _build_cell_specs, run_matrix


def _partition(items: list, num_groups: int) -> list[list]:
    """Splits `items` into `num_groups` roughly-equal contiguous chunks
    (never more groups than items -- a group that would be empty is
    dropped, since spawning a process with zero work is pure overhead).
    A no-op (returns `[]`) on an empty `items` list, rather than raising on
    a zero chunk_size."""
    if not items:
        return []
    num_groups = min(num_groups, len(items)) or 1
    chunk_size = -(-len(items) // num_groups)  # ceiling division
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _sanitize_failures(
    failures: list[tuple[str, int, Exception]],
) -> list[tuple[str, int, Exception]]:
    """Replaces each failure's original exception with a plain `RuntimeError`
    built from a string (`f"{type(exc).__name__}: {exc}\\n{traceback}"`), so
    `(cell_key, seed, exception)` is guaranteed picklable no matter what
    `run_matrix` put there, while still carrying a full traceback for
    diagnosis.

    Why this is needed: `run_matrix`'s per-cell/seed `try/except Exception`
    (see `src/simulation/matrix_runner.py`) catches ANY exception a real
    day-loop iteration can raise -- and in a real (non-dry-run) production
    run, that can be an `httpx` exception from a failed OpenRouter/Polygon
    call, carrying `Request`/`Response` objects with open sockets/streams in
    their internal state. `_run_cell_group` runs in a `ProcessPoolExecutor`
    worker and returns `(results, failures)` as ONE atomic tuple that must
    be pickled to cross back to the parent process; if even one exception
    inside `failures` fails to pickle, the WHOLE tuple fails to pickle --
    silently losing every successfully-computed cell in that worker's group
    (not just the one failure) behind a confusing `PicklingError`/
    `BrokenProcessPool`, instead of surfacing the real underlying cause.

    A plain built-in exception constructed from a string is always
    picklable, so sanitizing here (after `run_matrix` returns, before
    `_run_cell_group`'s return value crosses the process boundary) makes
    that failure mode structurally impossible, regardless of what kind of
    exception object a future change to `run_matrix` or its callees might
    someday put into `failures`. The full traceback text is captured here
    (not just the exception's own str()) since pickling drops
    `__traceback__` regardless of sanitization -- without this, a real
    day-200 failure would surface only as e.g. "TypeError: unsupported
    operand", with no indication of where.
    """
    import traceback as _traceback

    sanitized = []
    for cell_key, seed, exc in failures:
        formatted_tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        sanitized.append((cell_key, seed, RuntimeError(f"{type(exc).__name__}: {exc}\n{formatted_tb}")))
    return sanitized


def _run_cell_group(
    cell_seed_pairs: list[tuple[str, int]],
    model_candidates: list[str],
    num_days: int,
    dry_run: bool,
    database_url: str,
    matrix_run_id: str,
    llm_max_workers: int,
    checkpoint_dir: Path | None,
    openrouter_client_factory: Callable[[], "httpx.Client"] | None,
    polygon_client_factory: Callable[[], "httpx.Client"] | None,
) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]], LLMUsage]:
    """Runs in a separate process: builds its OWN engine/session (engines
    aren't picklable/shareable across processes either) and its own real
    clients from the factories, if given, then calls `run_matrix` once per
    `(cell_key, seed)` pair assigned to this group (not once for the whole
    group) -- `run_matrix(cell_keys=[...], seeds=[...])` always runs the
    full cross product of its two list arguments, so calling it once per
    pair is what lets `run_matrix_distributed` hand this worker an
    arbitrary, evenly-sized slice of the full 13-cell x N-seed cross
    product rather than only whole cells (which, for a fixed seed list
    shared by every cell, cannot split more finely than `num_processes <=
    len(cell_keys)` and wastes wall-clock time whenever cells don't divide
    evenly across workers).

    Returns this worker's own `LLMUsage` total (from
    `src.llm.llm_router.get_cumulative_usage()`, read once after all this
    group's pairs finish) alongside `(results, failures)`, since
    `_cumulative_usage` is a per-process module global -- the parent
    process never sees a worker's accumulated usage unless the worker
    hands it back explicitly.
    """
    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, connection_record) -> None:
        # busy_timeout FIRST, then journal_mode -- the WAL conversion itself
        # briefly needs exclusive file access, so as the first statement on a
        # fresh connection it could raise an uncaught "database is locked"
        # while a sibling worker process was mid-connect. busy_timeout alone
        # doesn't cover that specific race (it's a different SQLite error
        # class than ordinary lock waits) -- see
        # `database.session._set_journal_mode_wal_with_retry`'s docstring for
        # the confirmed failure mode and why the retry there is reused here
        # rather than duplicated.
        if not database_url.startswith("sqlite"):
            return
        from database.session import _set_journal_mode_wal_with_retry

        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout=30000")
        _set_journal_mode_wal_with_retry(cursor)
        cursor.close()

    # Checked here, not just inside `run_matrix`, so a stale database aborts
    # this worker BEFORE the client factories below open real HTTP clients
    # (and before any of this group's cells register themselves in
    # `simulation_runs`). `_run_cell_group` is an independent entry point into
    # the shared database -- it builds its own engine and never calls
    # `create_all_tables()` -- so it needs its own guard. See
    # `database/session.py`'s `assert_schema_current` for the failure mode.
    from database.session import assert_schema_current

    assert_schema_current(engine)

    session = Session(engine)
    openrouter_client = openrouter_client_factory() if openrouter_client_factory is not None else None
    polygon_client = polygon_client_factory() if polygon_client_factory is not None else None

    all_results: list[MatrixCellResult] = []
    all_failures: list[tuple[str, int, Exception]] = []
    for cell_key, seed in cell_seed_pairs:
        # Each pair's run_matrix call is its own try/except: an exception
        # that escapes run_matrix's OWN per-cell/seed handling (e.g. its
        # cell_keys validation, which deliberately runs before that
        # try/except) must not abandon every OTHER pair still queued in
        # THIS worker's group -- only the one pair that actually failed.
        try:
            results, failures = run_matrix(
                model_candidates=model_candidates,
                seeds=[seed],
                num_days=num_days,
                dry_run=dry_run,
                openrouter_client=openrouter_client,
                polygon_client=polygon_client,
                session=session,
                matrix_run_id=matrix_run_id,
                cell_keys=[cell_key],
                llm_max_workers=llm_max_workers,
                checkpoint_dir=checkpoint_dir,
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, matches
            # run_matrix's own per-cell/seed except: this one pair failed
            # outside run_matrix's internal handling, but the rest of this
            # worker's assigned pairs must still be attempted.
            all_failures.append((cell_key, seed, exc))
            continue
        all_results.extend(results)
        all_failures.extend(failures)

    from src.llm.llm_router import get_cumulative_usage

    return all_results, _sanitize_failures(all_failures), get_cumulative_usage()


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
    usage_callback: Callable[[LLMUsage], None] | None = None,
) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]]]:
    """Partitions the full 13-cell x `seeds` cross product into
    `num_processes` groups of `(cell_key, seed)` pairs and runs each group
    in its own OS process via one `run_matrix(cell_keys=[...], seeds=[...])`
    call per pair, against the same `database_url` (must be a file-based
    SQLite URL, or another DB that supports concurrent-process writes).
    Each worker process opens its OWN SQLAlchemy engine/session against
    `database_url` -- engines, like httpx.Client, are not picklable across
    the process boundary. `database/session.py`'s WAL-mode connect-event
    listener is registered against that module's specific `_engine`
    instance, not `database_url` globally, so it has no effect on engines
    built inside worker processes; `_run_cell_group` above registers the
    same pragmas on its own locally-built engine instead.

    Partitioning the full cross product (not just cell keys, with the same
    `seeds` list applied to every cell) means `num_processes` can usefully
    exceed the cell count (up to `len(cell_keys) * len(seeds)`), and that
    work divides evenly across workers even when neither `len(cell_keys)`
    nor `len(seeds)` alone divides evenly by `num_processes`.

    `matrix_run_id`, if `None`, is generated once here (not per-group) so
    every cell across every process shares one consistent prefix -- see
    `run_matrix`'s own `matrix_run_id` docstring for why a stable shared
    prefix matters for resumability.

    If any single worker raises an exception `ProcessPoolExecutor` cannot
    recover from (e.g. the worker process crashed/was OOM-killed --
    distinct from an ordinary per-cell/seed failure, which `run_matrix`
    already catches into `failures` and this function returns normally),
    that is logged and skipped rather than raised: every completed group's
    `(results, failures)` are still returned to the caller. A `run_matrix`
    invocation this expensive should never lose already-computed results
    from OTHER groups just because one worker died -- the caller can
    inspect which cell/seed pairs are simply absent from `results` (by
    comparing against the original `cell_keys x seeds` cross product) and
    retry those specifically, using the same `matrix_run_id`/
    `checkpoint_dir` to resume rather than re-paying for already-persisted
    days.

    `usage_callback`, if given, is called once per completed worker group
    (as each one finishes, not batched at the end) with that group's own
    `LLMUsage` total -- letting a caller sum/log running spend across the
    whole distributed run without polling anything. `None` (the default)
    is a no-op. Unlike `run_matrix`'s own `usage_callback` (called once per
    simulated day, with a running total), this one is called once per
    worker-group and reports only that group's total, since
    `_cumulative_usage` is a separate per-process counter in each worker
    and there is no cheaper way to see a worker's progress before it
    finishes its whole group.
    """
    from src.utils.helpers import generate_id

    if matrix_run_id is None:
        matrix_run_id = generate_id("matrix")

    all_cell_keys = [spec.key for spec in _build_cell_specs()]
    all_pairs = [(cell_key, seed) for cell_key in all_cell_keys for seed in seeds]
    groups = _partition(all_pairs, num_processes)

    all_results: list[MatrixCellResult] = []
    all_failures: list[tuple[str, int, Exception]] = []

    with ProcessPoolExecutor(max_workers=len(groups)) as executor:
        futures = [
            executor.submit(
                _run_cell_group,
                group,
                model_candidates,
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
        for future in futures:
            try:
                group_results, group_failures, group_usage = future.result()
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: one worker process
                # dying (crash, OOM-kill, an exception ProcessPoolExecutor itself
                # couldn't recover from) must never discard every OTHER group's
                # already-completed results -- see this function's docstring.
                all_failures.append(("<worker process failure>", -1, exc))
                continue
            all_results.extend(group_results)
            all_failures.extend(group_failures)
            if usage_callback is not None:
                usage_callback(group_usage)

    return all_results, all_failures
