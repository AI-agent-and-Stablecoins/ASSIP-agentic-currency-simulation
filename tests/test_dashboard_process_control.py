import time

import pytest

from dashboard import status_store
from dashboard.process_control import RunConfig, is_alive, resume, start, stop


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    yield


def test_is_alive_is_true_for_the_current_process_and_false_for_a_bogus_pid():
    import os

    assert is_alive(os.getpid()) is True
    assert is_alive(999_999_999) is False


def test_start_launches_a_real_process_and_records_its_pid():
    """`start()`'s own status_store.write_status call (Task 4's `start`
    function) happens synchronously in THIS test process, using the
    `_isolated_state_dir` fixture's patched location -- so the PID is
    already on record the moment `start()` returns, no polling needed.
    The spawned subprocess itself is real (matching Task 3's own runner
    test) and writes its OWN separate status update to the REAL
    dashboard/state/ directory once it starts, since a fresh subprocess
    can't inherit this test's monkeypatch -- that real file is cleaned up
    in the `finally` block below so this test leaves no residue.
    """
    matrix_run_id = "process-control-start-test"
    config = RunConfig(
        matrix_run_id=matrix_run_id,
        cell_keys=["master"],
        seeds=[0],
        num_days=1,
        dry_run=True,
    )

    try:
        start(config)

        status = status_store.read_status(matrix_run_id)
        assert status is not None
        assert is_alive(status["pid"])

        stop(matrix_run_id)
        time.sleep(1.0)
        final_status = status_store.read_status(matrix_run_id)
        assert final_status["state"] in ("stopped", "completed")
    finally:
        from src.utils.constants import REPO_ROOT

        (REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.json").unlink(missing_ok=True)


def test_stop_is_a_noop_when_no_process_is_recorded():
    # Must not raise even though no run has ever been started under this id.
    stop("never-started")


def test_resume_requires_an_existing_status_record():
    with pytest.raises(ValueError, match="no prior status recorded"):
        resume("never-started")
