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
from datetime import datetime, timezone
from pathlib import Path

from src.utils.constants import REPO_ROOT

_STATE_DIR = REPO_ROOT / "dashboard" / "state"
_ACTIVE_RUN_POINTER = "_active_run_id.json"


class StatusFileCorruptedError(Exception):
    def __init__(self, matrix_run_id: str, path: Path):
        self.matrix_run_id = matrix_run_id
        self.path = path
        super().__init__(f"Status file for {matrix_run_id!r} at {path} is corrupted/unreadable")


def state_dir() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR


def _status_path(matrix_run_id: str) -> Path:
    return state_dir() / f"{matrix_run_id}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)  # atomic on both POSIX and Windows


def write_status(matrix_run_id: str, **fields) -> None:
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)["matrix_run_id"]
