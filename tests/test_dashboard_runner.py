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


def test_runner_rejects_cell_keys_with_distributed():
    """Verifies that combining --cell-keys with --distributed is rejected
    with a non-zero exit code and no status file is written, preventing the
    safety gap where --cell-keys would be silently ignored under --distributed."""
    matrix_run_id = "test-cell-keys-distributed-rejection"
    status_file_path = REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.json"

    env = {**__import__("os").environ}
    try:
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
                "--distributed",
                "--dry-run",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Must exit with non-zero status
        assert result.returncode != 0, f"Expected non-zero exit but got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # Must not have written a status file (safety: no partial state left behind)
        assert not status_file_path.exists(), f"Status file should not exist but found at {status_file_path}"

        # Error message should mention the incompatibility
        error_output = result.stderr + result.stdout
        assert "cell-keys" in error_output.lower() or "--distributed" in error_output, \
            f"Error message should mention the incompatibility. Output:\n{error_output}"
    finally:
        status_file_path.unlink(missing_ok=True)


def test_runner_rejects_a_path_unsafe_matrix_run_id(tmp_path, monkeypatch):
    """Finding 9: matrix_run_id flows unsanitized into a filesystem path
    (checkpoint_dir = REPO_ROOT / "checkpoints" / matrix_run_id) -- a value
    like "../../escape" must be rejected up front with a clear error,
    before checkpoint_dir is ever handed to run_matrix to create/write into,
    rather than crashing deep inside the simulation with an uncaught OSError
    or silently escaping the checkpoints/ directory."""
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path / "state")
    matrix_run_id = "../../escape-attempt"

    env = {**__import__("os").environ, "DATABASE_URL": f"sqlite:///{tmp_path}/runner_test.db"}
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
        timeout=30,
    )

    assert result.returncode != 0
    # No checkpoint directory should have been created outside checkpoints/.
    assert not (REPO_ROOT / "checkpoints" / ".." / ".." / "escape-attempt").resolve().exists()
