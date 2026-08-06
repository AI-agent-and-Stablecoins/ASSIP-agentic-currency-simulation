import pickle

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.simulation.distributed_matrix_runner import _run_cell_group, run_matrix_distributed

MODEL_CANDIDATES = ["vendor/fake-model"]


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


def test_run_cell_group_sanitizes_an_unpicklable_failure_without_losing_group_mates(tmp_path, monkeypatch):
    """Regression test for a task-review finding: `run_matrix`'s per-cell/
    seed `try/except Exception` (see `src/simulation/matrix_runner.py`) can
    catch ANY exception a real day-loop iteration raises -- in a real
    (non-dry-run) production run that can be an `httpx` exception carrying
    open sockets/streams that `pickle` cannot serialize. `_run_cell_group`
    returns `(results, failures)` as ONE atomic tuple that `ProcessPoolExecutor`
    must pickle to cross back to the parent process; before the fix, one
    unpicklable exception inside `failures` would fail to pickle the WHOLE
    tuple, silently losing every successfully-computed cell in that worker's
    group behind a confusing `PicklingError`/`BrokenProcessPool`.

    This test forces a REAL failure out of the REAL `run_matrix` code (via
    the same `monkeypatch`-on-`run_timestep` call-count trick
    `tests/test_matrix_runner.py::test_run_matrix_records_a_failed_cell_seed_
    and_still_completes_the_rest` uses), deliberately raising an exception
    type defined INSIDE this test function -- a locally-scoped class is a
    genuine, easy-to-construct example of an object `pickle` refuses to
    serialize ("Can't pickle <locals>.X: it's not the same object as ..."),
    standing in for the unpicklable-httpx-exception scenario described
    above without needing a real failed network call.

    `_run_cell_group` is called DIRECTLY here (a plain in-process function
    call), not through a real `ProcessPoolExecutor` subprocess: on Windows,
    `ProcessPoolExecutor` always uses the `spawn` start method, which starts
    a fresh interpreter that re-imports modules from disk rather than
    inheriting the parent process's already-monkeypatched module objects --
    so a monkeypatch applied in this test process has no effect inside a
    real spawned worker. Since the sanitization fix under test lives
    entirely inside `_run_cell_group` (it runs identically whether that
    function is invoked directly or via the pool), calling it directly here
    still exercises exactly the code path that matters, while still routing
    through the REAL `run_matrix` (not a hand-mocked return value) so the
    injected failure is genuine, not fabricated.
    """
    import src.simulation.matrix_runner as matrix_runner_module

    class _LocallyScopedError(Exception):
        """Defined inside the test function on purpose -- pickle cannot
        resolve a function-local class by qualified name, so an instance of
        this is a genuinely unpicklable exception, unlike a plain
        RuntimeError."""

    # Sanity-check the test's own premise: an instance of this class really
    # is unpicklable, so sanitizing it is not a no-op.
    try:
        pickle.dumps(_LocallyScopedError("boom"))
        raise AssertionError("expected _LocallyScopedError to be unpicklable -- test premise is wrong")
    except (pickle.PicklingError, AttributeError, TypeError):
        pass

    original_run_timestep = matrix_runner_module.run_timestep
    call_count = {"n": 0}

    def flaky_run_timestep(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _LocallyScopedError("injected unpicklable failure for testing")
        return original_run_timestep(*args, **kwargs)

    monkeypatch.setattr(matrix_runner_module, "run_timestep", flaky_run_timestep)

    db_path = tmp_path / "sanitize_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    all_cell_keys = [spec.key for spec in matrix_runner_module._build_cell_specs()]
    cell_seed_pairs = [(cell_key, 0) for cell_key in all_cell_keys]

    results, failures, usage = _run_cell_group(
        cell_seed_pairs=cell_seed_pairs,
        model_candidates=MODEL_CANDIDATES,
        num_days=1,
        dry_run=True,
        database_url=f"sqlite:///{db_path}",
        matrix_run_id="sanitize-test",
        llm_max_workers=1,
        checkpoint_dir=None,
        openrouter_client_factory=None,
        polygon_client_factory=None,
    )

    # Exactly one failure -- the first cell/seed processed (`_build_cell_
    # specs()` yields "master" first, matching test_matrix_runner.py's
    # analogous test).
    assert len(failures) == 1
    failed_cell_key, failed_seed, sanitized_exc = failures[0]
    assert failed_cell_key == "master"
    assert failed_seed == 0

    # The live exception object was replaced with a plain, always-picklable
    # RuntimeError...
    assert type(sanitized_exc) is RuntimeError
    assert not isinstance(sanitized_exc, _LocallyScopedError)
    # ...but the ORIGINAL exception's type name and message are still
    # readable in the new one, so the failure remains diagnosable.
    assert "_LocallyScopedError" in str(sanitized_exc)
    assert "injected unpicklable failure for testing" in str(sanitized_exc)

    # The other 12 cells in this SAME group must not be lost.
    assert len(results) == 12
    assert "master" not in {r.cell_key for r in results}

    # The direct proof: the exact tuple `_run_cell_group` returns (what
    # ProcessPoolExecutor must pickle to cross the process boundary) now
    # round-trips cleanly, which would NOT have been true with the original
    # unpicklable exception in `failures`.
    pickled = pickle.dumps((results, failures))
    roundtripped_results, roundtripped_failures = pickle.loads(pickled)
    assert len(roundtripped_results) == 12
    assert roundtripped_failures[0][0] == "master"
    assert isinstance(roundtripped_failures[0][2], RuntimeError)


def test_run_matrix_distributed_partitions_the_full_cell_seed_cross_product(tmp_path):
    """Regression test for a whole-branch review finding: partitioning must
    split the full 13-cell x N-seed cross product, not just cell keys with
    the same seed list applied to every one -- the latter caps useful
    `num_processes` at 13 and wastes wall-clock time whenever the cell
    count doesn't divide evenly. With 2 seeds x 13 cells = 26 pairs across
    4 processes, requesting more workers than cells (num_processes=4 <
    26 pairs, but > 13 cells alone would still have been meaningful) proves
    the split is over pairs: total result count must be 26, not 13."""
    db_path = tmp_path / "cross_product_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    results, failures = run_matrix_distributed(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0, 1],
        num_days=1,
        dry_run=True,
        num_processes=4,
        matrix_run_id="cross-product-test",
        database_url=f"sqlite:///{db_path}",
    )

    assert failures == []
    assert len(results) == 26  # 13 cells x 2 seeds
    seen_pairs = {(r.cell_key, r.seed) for r in results}
    assert len(seen_pairs) == 26


def test_run_cell_group_survives_one_bad_pair_without_losing_its_group_mates(tmp_path):
    """Regression test for a whole-branch review finding: before this fix,
    a single bad `(cell_key, seed)` pair raising an exception that escapes
    `run_matrix`'s OWN per-cell/seed handling (e.g. its `cell_keys`
    validation, which deliberately runs before that try/except -- a
    genuine escape, not a contrived one) would abort `_run_cell_group`'s
    entire loop, losing every OTHER pair already assigned to that SAME
    worker, not just the one bad pair.

    Forces this via a real, uncaught `ValueError`: `run_matrix`'s own
    `cell_keys` validation rejects any key not in its real
    `_build_cell_specs()` output, so simply including a fabricated,
    genuinely-unknown key in this worker's `cell_seed_pairs` triggers it
    naturally -- no monkeypatching needed.
    """
    import src.simulation.matrix_runner as matrix_runner_module

    real_specs = matrix_runner_module._build_cell_specs()

    db_path = tmp_path / "bad_pair_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    cell_seed_pairs = [("not-a-real-cell-key", 0)] + [(spec.key, 0) for spec in real_specs]

    results, failures, _usage = _run_cell_group(
        cell_seed_pairs=cell_seed_pairs,
        model_candidates=MODEL_CANDIDATES,
        num_days=1,
        dry_run=True,
        database_url=f"sqlite:///{db_path}",
        matrix_run_id="bad-pair-test",
        llm_max_workers=1,
        checkpoint_dir=None,
        openrouter_client_factory=None,
        polygon_client_factory=None,
    )

    # The bad pair's failure is recorded, not raised -- and it did not
    # abort the rest of this worker's assigned pairs.
    assert any(cell_key == "not-a-real-cell-key" for cell_key, _, _ in failures)
    assert len(results) == 13
    assert {r.cell_key for r in results} == {spec.key for spec in real_specs}


def test_run_matrix_distributed_calls_usage_callback_once_per_completed_worker_group(tmp_path):
    """Regression test for a whole-branch review finding (C1): the
    distributed runner had no way to see LLM spend at all -- `_cumulative_
    usage` is a per-process module global, so a worker's accumulated usage
    never reached the parent process unless the worker handed it back
    explicitly. `usage_callback` is called once per completed worker group
    (not per simulated day, unlike `run_matrix`'s own `usage_callback`),
    since a worker's usage isn't visible to the parent until that worker's
    whole group finishes."""
    db_path = tmp_path / "usage_callback_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    calls = []

    run_matrix_distributed(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        num_processes=2,
        matrix_run_id="usage-callback-test",
        database_url=f"sqlite:///{db_path}",
        usage_callback=lambda usage: calls.append(usage),
    )

    # One call per completed group (2 processes requested for 13 cells ->
    # 2 groups, per _partition's ceiling-division chunking).
    assert len(calls) == 2
    for usage in calls:
        assert hasattr(usage, "total_tokens")
