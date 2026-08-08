"""Start/stop/resume control for the dashboard's subprocess runner
(`dashboard/runner.py`). This module owns the ONLY place a simulation
process is spawned or killed from the dashboard -- Streamlit's own
rerun-on-every-interaction model never touches the subprocess directly.
"""

import subprocess
import sys
import time

import psutil
from pydantic import BaseModel

from dashboard import status_store
from src.utils.constants import REPO_ROOT


class RunConfig(BaseModel):
    matrix_run_id: str
    cell_keys: list[str] | None = None
    seeds: list[int]
    num_days: int
    dry_run: bool
    distributed: bool = False
    num_processes: int = 4


def is_alive(pid: int) -> bool:
    return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE


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
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    return {"start_new_session": True}


def start(config: RunConfig) -> None:
    command = _build_command(config)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_build_popen_kwargs(),  # detach: must outlive this Streamlit rerun
        )
    except OSError as exc:
        status_store.write_status(config.matrix_run_id, state="failed", error=f"launch failed: {exc}")
        raise

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

    # seeds/num_days/cell_keys/distributed/num_processes are written here
    # (not by runner.py itself) specifically so `resume()` can reconstruct
    # a full RunConfig later purely from the status file, without needing
    # runner.py to duplicate this bookkeeping.
    status_store.write_status(
        config.matrix_run_id,
        pid=process.pid,
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
    if is_alive(pid):
        psutil.Process(pid).terminate()
    status_store.write_status(matrix_run_id, state="stopped")


def resume(matrix_run_id: str) -> None:
    status = status_store.read_status(matrix_run_id)
    if status is None:
        raise ValueError(f"Cannot resume {matrix_run_id!r}: no prior status recorded for it")
    pid = status.get("pid")
    if pid is not None and is_alive(pid):
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
