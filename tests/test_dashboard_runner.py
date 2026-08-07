import json
import subprocess
import sys
from pathlib import Path

from src.utils.constants import REPO_ROOT


def test_runner_dry_run_writes_a_completed_status(tmp_path, monkeypatch):
    """Runs dashboard/runner.py as a REAL subprocess (matching how
    process_control.py will actually invoke it in Task 4), pointed at a
    throwaway SQLite file and a throwaway status-file directory, and
    confirms it writes a final "completed" status with zero failures."""
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path / "state")
    db_path = tmp_path / "runner_test.db"
    matrix_run_id = "runner-dry-run-test"

    env = {**__import__("os").environ, "DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dashboard.runner",
            "--matrix-run-id",
            matrix_run_id,
            "--seeds",
            "0",
            "--num-days",
            "1",
            "--cell-keys",
            "master",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr

    from dashboard.status_store import read_status

    # This test's own monkeypatch of _STATE_DIR only affects THIS process,
    # not the subprocess -- so read the subprocess's real status file
    # location directly instead of relying on the patched path.
    real_status_path = REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.json"
    try:
        status = json.loads(real_status_path.read_text(encoding="utf-8"))
        assert status["state"] == "completed"
        assert status["failures"] == []
    finally:
        real_status_path.unlink(missing_ok=True)
