"""Start/stop/resume control for the dashboard's subprocess runner
(`dashboard/runner.py`). This module owns the ONLY place a simulation
process is spawned or killed from the dashboard -- Streamlit's own
rerun-on-every-interaction model never touches the subprocess directly.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from pydantic import BaseModel

from dashboard import status_store
from src.utils.constants import REPO_ROOT

# Bound on how many trailing bytes of a log file read_log_tail() will ever
# read from disk. Rendered every 5s by app.py's auto-refresh fragment for as
# long as a failed run's status is displayed, so reading the WHOLE file on
# every tick (the pre-fix behavior) is wasteful/slow for a large log. This is
# an approximation -- if the last `num_lines` happen to span more than this
# many bytes, fewer complete lines than requested are returned -- an
# accepted tradeoff for a diagnostic log tail.
_LOG_TAIL_SEEK_BACK_BYTES = 64 * 1024

# Tolerance (seconds) for matching a live process's create_time() against the
# one recorded in the status file at launch time -- guards against PID reuse
# (e.g. after a reboot) making stop()/is_alive() target an unrelated process.
_CREATE_TIME_TOLERANCE_SECONDS = 1.0


class RunConfig(BaseModel):
    matrix_run_id: str
    cell_keys: list[str] | None = None
    seeds: list[int]
    num_days: int
    dry_run: bool
    distributed: bool = False
    num_processes: int = 4


def is_alive(pid: int, pid_create_time: float | None = None) -> bool:
    """True if `pid` is a live, non-zombie process. When `pid_create_time`
    is given (the value recorded in the status file at launch time), also
    verifies the live process's own create_time() matches within a small
    tolerance -- a mismatch means the PID was recycled by an unrelated
    process (e.g. after a reboot), which must be treated as "not alive".
    All psutil exceptions (the process can vanish between the existence
    check and the status/create_time reads -- a TOCTOU race) are treated
    as "not alive" rather than allowed to raise out to callers/the UI.
    """
    try:
        if not psutil.pid_exists(pid):
            return False
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if pid_create_time is not None:
            if abs(proc.create_time() - pid_create_time) >= _CREATE_TIME_TOLERANCE_SECONDS:
                return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _build_command(config: RunConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dashboard.runner",
        "--matrix-run-id",
        config.matrix_run_id,
        "--seeds",
        ",".join(str(s) for s in config.seeds),
        "--num-days",
        str(config.num_days),
    ]
    if config.cell_keys:
        command += ["--cell-keys", ",".join(config.cell_keys)]
    command.append("--dry-run" if config.dry_run else "--real")
    if config.distributed:
        command += ["--distributed", "--num-processes", str(config.num_processes)]
    return command


def _build_popen_kwargs() -> dict:
    """Platform-specific kwargs to detach the child process so it outlives
    this Streamlit process. `start_new_session=True` is a documented no-op
    on Windows (CPython's Windows `_execute_child` names the parameter
    `unused_start_new_session` and never acts on it), so Windows needs its
    own real detachment flags instead.
    """
    if sys.platform == "win32":
        return {
            "creationflags": (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            )
        }
    return {"start_new_session": True}


def log_path(matrix_run_id: str) -> Path:
    """Per-run log file the child process's stdout/stderr are redirected
    to (start() below) instead of subprocess.DEVNULL, so the app can show
    the tail of a failed run's diagnostics instead of just an unexplained
    "failed" status."""
    return status_store.state_dir() / f"{matrix_run_id}.log"


def read_log_tail(matrix_run_id: str, num_lines: int = 20) -> str | None:
    """Last `num_lines` lines of the child process's redirected
    stdout/stderr, or None if no log file exists yet for this run.

    Reads at most `_LOG_TAIL_SEEK_BACK_BYTES` trailing bytes of the file
    (via seek from the end) rather than the whole file, since this is
    called every 5s by app.py's auto-refresh fragment for as long as a
    failed run's status is displayed."""
    path = log_path(matrix_run_id)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _LOG_TAIL_SEEK_BACK_BYTES))
            chunk = f.read()
        lines = chunk.decode("utf-8", errors="replace").splitlines()
        return "\n".join(lines[-num_lines:])
    except OSError:
        return None


def start(config: RunConfig) -> None:
    status_store.validate_matrix_run_id(config.matrix_run_id)
    command = _build_command(config)
    log_file_path = log_path(config.matrix_run_id)
    log_file = open(log_file_path, "wb")
    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=log_file,
                stderr=log_file,
                **_build_popen_kwargs(),  # detach: must outlive this Streamlit rerun
            )
        except OSError as exc:
            status_store.write_status(config.matrix_run_id, state="failed", error=f"launch failed: {exc}")
            raise
    finally:
        # The child process inherits its own handle to the file (or, on the
        # exception path above, nothing was ever spawned); this parent-side
        # handle isn't needed past Popen() and must not be leaked.
        log_file.close()

    # Brief grace period to catch an immediate child failure (e.g. runner.py's
    # own CLI validation rejecting an invalid flag combination) before we
    # claim a live PID. Long enough to catch a fast argparse-level exit,
    # short enough not to meaningfully block the caller on a normal launch.
    time.sleep(0.3)
    exit_code = process.poll()
    if exit_code is not None:
        status_store.write_status(
            config.matrix_run_id,
            state="failed",
            error=f"process exited immediately with code {exit_code}",
        )
        return

    try:
        pid_create_time = psutil.Process(process.pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pid_create_time = None

    # seeds/num_days/cell_keys/distributed/num_processes are written here
    # (not by runner.py itself) specifically so `resume()` can reconstruct
    # a full RunConfig later purely from the status file, without needing
    # runner.py to duplicate this bookkeeping.
    status_store.write_status(
        config.matrix_run_id,
        pid=process.pid,
        pid_create_time=pid_create_time,
        state="running",
        dry_run=config.dry_run,
        seeds=config.seeds,
        num_days=config.num_days,
        cell_keys=config.cell_keys,
        distributed=config.distributed,
        num_processes=config.num_processes,
    )


def stop(matrix_run_id: str) -> None:
    status = status_store.read_status(matrix_run_id)
    if status is None or status.get("pid") is None:
        return
    pid = status["pid"]
    pid_create_time = status.get("pid_create_time")
    if is_alive(pid, pid_create_time):
        try:
            parent = psutil.Process(pid)
            # A distributed run (run_matrix_distributed) spawns a
            # ProcessPoolExecutor of worker processes under this same
            # parent PID -- terminating only the parent would orphan those
            # workers, leaving them alive (and, for a real/non-dry-run,
            # still spending) after Stop is clicked.
            children = parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            parent = None
            children = []

        if parent is not None:
            try:
                parent.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if children:
            gone, alive = psutil.wait_procs(children, timeout=2)
            for child in alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    status_store.write_status(matrix_run_id, state="stopped")


def resume(matrix_run_id: str) -> None:
    status = status_store.read_status(matrix_run_id)
    if status is None:
        raise ValueError(f"Cannot resume {matrix_run_id!r}: no prior status recorded for it")
    pid = status.get("pid")
    if pid is not None and is_alive(pid, status.get("pid_create_time")):
        raise ValueError(f"Cannot resume {matrix_run_id!r}: a process (pid {pid}) is already running for it")
    config = RunConfig(
        matrix_run_id=matrix_run_id,
        dry_run=status.get("dry_run", True),
        seeds=status.get("seeds", [0]),
        num_days=status.get("num_days", 1),
        cell_keys=status.get("cell_keys"),
        distributed=status.get("distributed", False),
        num_processes=status.get("num_processes", 4),
    )
    start(config)
