import subprocess
import sys
import time

import pytest

from dashboard import status_store
from dashboard.process_control import RunConfig, _build_popen_kwargs, is_alive, resume, start, stop


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


def test_build_popen_kwargs_uses_windows_detachment_flags_on_win32(monkeypatch):
    """`start_new_session=True` is a documented no-op on Windows (CPython's
    Windows `_execute_child` names the parameter `unused_start_new_session`
    and never acts on it), so on win32 we must use the real Windows
    detachment flags instead, and NOT start_new_session.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    kwargs = _build_popen_kwargs()
    assert kwargs == {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    assert "start_new_session" not in kwargs


def test_build_popen_kwargs_uses_start_new_session_on_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    kwargs = _build_popen_kwargs()
    assert kwargs == {"start_new_session": True}
    assert "creationflags" not in kwargs


def test_resume_raises_when_the_prior_process_is_still_running():
    """If the previously-recorded process is still alive, resume() must
    refuse rather than spawning a second `runner.py` against the same
    matrix_run_id (which would race on the same checkpoint dir/status
    file).
    """
    matrix_run_id = "process-control-resume-while-running-test"
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
        assert status["state"] == "running"
        assert is_alive(status["pid"])

        with pytest.raises(ValueError, match="already running"):
            resume(matrix_run_id)
    finally:
        stop(matrix_run_id)
        from src.utils.constants import REPO_ROOT

        (REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.json").unlink(missing_ok=True)


def test_start_marks_failed_when_the_child_exits_immediately():
    """`runner.py` rejects the --cell-keys + --distributed combination at
    argparse/validation time and exits almost instantly. `start()` must
    catch that within its grace-period poll and record state="failed"
    with a live-looking PID never written, instead of leaving a stale
    "running" status pointing at an already-dead process.
    """
    # Note: runner.py rejects this combination before it ever writes its own
    # status (that check runs before any status_store.write_status call in
    # main()), so unlike the other real-process tests here, no status file
    # ever lands in the REAL dashboard/state/ dir -- only start()'s own
    # write_status call fires, into this test's isolated tmp_path.
    matrix_run_id = "process-control-fast-failure-test"
    config = RunConfig(
        matrix_run_id=matrix_run_id,
        cell_keys=["master"],
        seeds=[0],
        num_days=1,
        dry_run=True,
        distributed=True,
    )

    start(config)

    status = status_store.read_status(matrix_run_id)
    assert status is not None
    assert status["state"] == "failed"
    assert "exited immediately" in status["error"]
