"""JSON-backed status file for one dashboard-tracked simulation run.

One file per `matrix_run_id`, at `dashboard/state/<matrix_run_id>.json`.
This file is the control-plane state the dashboard's Streamlit process
and the subprocess runner (`dashboard/runner.py`) share -- NOT live
simulation progress, which is read directly from the project's own
SQLite database instead (see `dashboard/queries.py`). This file covers
only what the database doesn't hold: which PID owns the currently-tracked
run, the dry-run flag, best-effort token usage, and the final
results/failures once the run completes.

Schema (all fields optional except matrix_run_id, since write_status
merges incrementally):
{
    "matrix_run_id": str,
    "pid": int,
    "pid_create_time": float,  # psutil.Process(pid).create_time(), to detect PID reuse
    "state": "running" | "stopped" | "completed" | "failed" | "crashed",
    "dry_run": bool,
    "checkpoint_dir": str | None,
    "started_at": str (UTC ISO-8601),
    "last_updated": str (UTC ISO-8601),
    "cumulative_usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
    "failures": list[[cell_key, seed, message]] | None,
    "error": str | None
}
"""

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.utils.constants import REPO_ROOT

_STATE_DIR = REPO_ROOT / "dashboard" / "state"
_ACTIVE_RUN_POINTER = "_active_run_id.json"

_MATRIX_RUN_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")


class StatusFileCorruptedError(Exception):
    def __init__(self, matrix_run_id: str, path: Path):
        self.matrix_run_id = matrix_run_id
        self.path = path
        super().__init__(f"Status file for {matrix_run_id!r} at {path} is corrupted/unreadable")


def validate_matrix_run_id(matrix_run_id: str) -> None:
    """Guards against path traversal / filesystem-invalid characters, since
    matrix_run_id flows unsanitized into filesystem paths in this module
    (the status file) and in dashboard/runner.py (the checkpoint dir). A
    value like "../../foo" could otherwise write outside the intended
    directory."""
    if (
        not isinstance(matrix_run_id, str)
        or matrix_run_id in (".", "..")
        or not _MATRIX_RUN_ID_RE.fullmatch(matrix_run_id)
    ):
        raise ValueError(
            f"Invalid matrix_run_id {matrix_run_id!r}: must match "
            r"[A-Za-z0-9._-]{1,128}"
            " and must not be exactly '.' or '..'"
        )


def state_dir() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR


def _status_path(matrix_run_id: str) -> Path:
    return state_dir() / f"{matrix_run_id}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    # Unique-per-writer temp file name: process_control.start() (parent) and
    # runner.py's main() (child) both write to the SAME status file
    # concurrently by design, so a deterministic temp name would let them
    # collide on the same temp file and produce a corrupted/truncated result.
    tmp_path = path.with_name(f"{path.stem}.json.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)  # atomic on both POSIX and Windows


def write_status(matrix_run_id: str, **fields) -> None:
    validate_matrix_run_id(matrix_run_id)
    path = _status_path(matrix_run_id)
    existing = read_status(matrix_run_id) or {"matrix_run_id": matrix_run_id}
    existing.update(fields)
    existing["last_updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(path, existing)


def read_status(matrix_run_id: str) -> dict | None:
    path = _status_path(matrix_run_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise StatusFileCorruptedError(matrix_run_id, path) from exc


def set_active_run_id(matrix_run_id: str) -> None:
    _atomic_write_json(state_dir() / _ACTIVE_RUN_POINTER, {"matrix_run_id": matrix_run_id})


def get_active_run_id() -> str | None:
    path = state_dir() / _ACTIVE_RUN_POINTER
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["matrix_run_id"]
    except (json.JSONDecodeError, OSError, KeyError):
        # A corrupted active-run pointer should just mean "no default to
        # show," not a crash -- unlike a corrupted per-run status file
        # (read_status), there's no specific matrix_run_id to blame here.
        return None
