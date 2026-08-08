"""Start/stop/resume control for the dashboard's subprocess runner
(`dashboard/runner.py`). This module owns the ONLY place a simulation
process is spawned or killed from the dashboard -- Streamlit's own
rerun-on-every-interaction model never touches the subprocess directly.
"""

import subprocess
import sys

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


def start(config: RunConfig) -> None:
    command = _build_command(config)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach: must outlive this Streamlit rerun
        )
    except OSError as exc:
        status_store.write_status(config.matrix_run_id, state="failed", error=f"launch failed: {exc}")
        raise
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
