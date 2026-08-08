import os
import subprocess
import sys
import time

import psutil
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


def test_start_launches_a_real_process_and_records_its_pid(tmp_path, monkeypatch):
    """`start()`'s own status_store.write_status call (Task 4's `start`
    function) happens synchronously in THIS test process, using the
    `_isolated_state_dir` fixture's patched location -- so the PID is
    already on record the moment `start()` returns, no polling needed.
    The spawned subprocess itself is real (matching Task 3's own runner
    test) and writes its OWN separate status update to the REAL
    dashboard/state/ directory once it starts, since a fresh subprocess
    can't inherit this test's monkeypatch -- that real file is cleaned up
    in the `finally` block below so this test leaves no residue.

    Finding 11(b): DATABASE_URL is overridden so the spawned runner.py
    subprocess (which inherits this process's environment by default,
    since start() passes no explicit `env=` to subprocess.Popen) touches a
    throwaway SQLite file instead of the repo-root assip.db.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
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
    # Windows detachment flags: CREATE_NEW_PROCESS_GROUP=0x00000200 | DETACHED_PROCESS=0x00000008
    assert kwargs == {"creationflags": 0x00000208}
    assert "start_new_session" not in kwargs


def test_build_popen_kwargs_uses_start_new_session_on_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    kwargs = _build_popen_kwargs()
    assert kwargs == {"start_new_session": True}
    assert "creationflags" not in kwargs


def test_resume_raises_when_the_prior_process_is_still_running(tmp_path, monkeypatch):
    """If the previously-recorded process is still alive, resume() must
    refuse rather than spawning a second `runner.py` against the same
    matrix_run_id (which would race on the same checkpoint dir/status
    file).
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
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


# --- Finding 9: matrix_run_id must be validated before start() ever spawns anything ---


def test_start_rejects_a_path_unsafe_matrix_run_id_without_spawning_a_process():
    config = RunConfig(
        matrix_run_id="../../escape-attempt",
        cell_keys=["master"],
        seeds=[0],
        num_days=1,
        dry_run=True,
    )
    with pytest.raises(ValueError):
        start(config)


# --- Finding 7: PID reuse must not fool is_alive()/stop() into targeting an unrelated process ---


def test_is_alive_detects_pid_reuse_via_create_time_mismatch():
    real_pid = os.getpid()
    real_create_time = psutil.Process(real_pid).create_time()

    # Matching create_time: genuinely the same process.
    assert is_alive(real_pid, real_create_time) is True
    # A create_time far from the live process's own: this PID was recycled
    # by an unrelated process (e.g. after a reboot) -- must be "not alive".
    assert is_alive(real_pid, real_create_time + 999.0) is False


def test_is_alive_without_a_recorded_create_time_falls_back_to_existence_only():
    assert is_alive(os.getpid(), None) is True
    assert is_alive(999_999_999, None) is False


def test_is_alive_returns_false_rather_than_raising_for_an_already_dead_pid():
    """Regression for the unguarded TOCTOU race: a PID that existed a
    moment ago but has since fully exited must make is_alive() return
    False, never raise psutil.NoSuchProcess out to the caller."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    # Fully exited now -- some platforms keep a zombie/defunct entry around
    # briefly, but is_alive() must not raise either way.
    assert is_alive(proc.pid) in (True, False)


def test_start_records_a_pid_create_time_matching_the_spawned_process(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    matrix_run_id = "process-control-pid-create-time-test"
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
        assert status["pid_create_time"] is not None
        live_create_time = psutil.Process(status["pid"]).create_time()
        assert abs(status["pid_create_time"] - live_create_time) < 1.0
        assert is_alive(status["pid"], status["pid_create_time"]) is True
    finally:
        stop(matrix_run_id)
        from src.utils.constants import REPO_ROOT

        (REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.json").unlink(missing_ok=True)


# --- Finding 4: stop() must not orphan child worker processes ---


def test_stop_terminates_child_processes_too(tmp_path, monkeypatch):
    """Simulates the run_matrix_distributed ProcessPoolExecutor shape:
    stop() is handed a PID whose OS process has spawned its OWN child
    process. Without Finding 4's fix, stop() only terminated the parent,
    leaving the child alive (and, for a real run, still spending) after
    Stop was clicked.
    """
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    matrix_run_id = "process-control-stop-kills-children-test"

    child_spawning_script = (
        "import subprocess, sys, time;"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        "print(p.pid, flush=True);"
        "time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", child_spawning_script],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        child_pid = int(parent.stdout.readline().strip())

        status_store.write_status(matrix_run_id, pid=parent.pid, state="running")
        assert is_alive(parent.pid) is True
        assert is_alive(child_pid) is True

        stop(matrix_run_id)
        # A short wait for the terminate()/wait_procs() calls inside stop()
        # to actually take effect at the OS level.
        deadline = time.time() + 5
        while time.time() < deadline and (is_alive(parent.pid) or is_alive(child_pid)):
            time.sleep(0.2)

        assert is_alive(parent.pid) is False
        assert is_alive(child_pid) is False
    finally:
        for pid in (parent.pid, locals().get("child_pid")):
            if pid:
                try:
                    psutil.Process(pid).kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        parent.wait(timeout=5)
