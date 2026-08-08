import json
import os
import threading

import pytest

from dashboard.status_store import (
    StatusFileCorruptedError,
    get_active_run_id,
    read_status,
    set_active_run_id,
    state_dir,
    validate_matrix_run_id,
    write_status,
)


def test_read_status_returns_none_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    assert read_status("no-such-run") is None


def test_write_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    write_status("run-1", pid=1234, state="running", dry_run=True)

    status = read_status("run-1")
    assert status["matrix_run_id"] == "run-1"
    assert status["pid"] == 1234
    assert status["state"] == "running"
    assert status["dry_run"] is True
    assert "last_updated" in status


def test_write_status_merges_fields_rather_than_replacing(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    write_status("run-1", pid=1234, state="running")
    write_status("run-1", state="stopped")

    status = read_status("run-1")
    assert status["pid"] == 1234  # preserved from the first write
    assert status["state"] == "stopped"  # updated by the second write


def test_read_status_raises_a_clear_error_on_corrupted_file(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run-1.json").write_text("not valid json{{{", encoding="utf-8")

    with pytest.raises(StatusFileCorruptedError):
        read_status("run-1")


def test_active_run_id_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    assert get_active_run_id() is None

    set_active_run_id("run-1")
    assert get_active_run_id() == "run-1"

    set_active_run_id("run-2")
    assert get_active_run_id() == "run-2"


def test_state_dir_is_created_if_missing(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "state"
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", target)
    assert not target.exists()

    result = state_dir()

    assert result == target
    assert target.exists()


# --- Finding 6: get_active_run_id() must not crash on a corrupted pointer file ---


def test_get_active_run_id_returns_none_on_corrupted_pointer_file(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "_active_run_id.json").write_text("not valid json{{{", encoding="utf-8")

    # Must return None (treat corruption as "no default to show"), not raise.
    assert get_active_run_id() is None


def test_get_active_run_id_returns_none_when_pointer_file_missing_matrix_run_id_key(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "_active_run_id.json").write_text(json.dumps({"unrelated": "field"}), encoding="utf-8")

    assert get_active_run_id() is None


# --- Finding 9: matrix_run_id must be validated before it reaches a filesystem path ---


@pytest.mark.parametrize(
    "valid_id",
    ["run-1", "matrix_run.2026-08-08", "A", "a" * 128],
)
def test_validate_matrix_run_id_accepts_well_formed_ids(valid_id):
    validate_matrix_run_id(valid_id)  # must not raise


@pytest.mark.parametrize(
    "invalid_id",
    ["../../etc/passwd", "run/with/slash", "run with space", "", "a" * 129, "run\\with\\backslash"],
)
def test_validate_matrix_run_id_rejects_path_unsafe_or_oversized_ids(invalid_id):
    with pytest.raises(ValueError):
        validate_matrix_run_id(invalid_id)


@pytest.mark.parametrize("dot_id", [".", ".."])
def test_validate_matrix_run_id_rejects_bare_dot_and_dotdot(dot_id):
    """Re-review finding: _MATRIX_RUN_ID_RE = r"[A-Za-z0-9._-]{1,128}" matches
    "." and ".." in full (the character class includes '.'), even though
    these values later get used to build a path
    (REPO_ROOT / "checkpoints" / matrix_run_id in dashboard/runner.py) --
    ".." would resolve the checkpoint dir to REPO_ROOT itself, and "."
    would collapse all runs' checkpoints into the same directory. Both
    must be explicitly rejected regardless of the regex."""
    with pytest.raises(ValueError):
        validate_matrix_run_id(dot_id)


def test_write_status_rejects_a_path_traversal_matrix_run_id_before_touching_the_filesystem(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    with pytest.raises(ValueError):
        write_status("../../escape-attempt", pid=1234, state="running")

    # No file should have been written anywhere under (or outside) tmp_path.
    assert list(tmp_path.rglob("*")) == [] or all(not p.is_file() for p in tmp_path.rglob("*"))


# --- Finding 10: concurrent writers must not collide on the same temp file name ---


def test_atomic_write_uses_a_unique_temp_file_name_per_writer(tmp_path, monkeypatch):
    """Two "writers" (simulated here by directly calling the private
    _atomic_write_json helper from two threads with distinct thread ids)
    must not collide on the same deterministic temp file name -- that was
    the actual bug: process_control.start() (parent) and runner.py's main()
    (child) both write the SAME status file concurrently by design.
    """
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    from dashboard.status_store import _atomic_write_json

    path = tmp_path / "concurrent-run.json"
    tmp_path.mkdir(parents=True, exist_ok=True)

    errors = []
    barrier = threading.Barrier(2)

    def _writer(payload: dict) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(20):
                _atomic_write_json(path, payload)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=_writer, args=({"who": "writer-1", "pid": os.getpid()},))
    t2 = threading.Thread(target=_writer, args=({"who": "writer-2", "pid": os.getpid()},))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert errors == []
    # The file must always be valid, complete JSON afterward -- never
    # truncated/corrupted by a temp-file collision between the two threads.
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["who"] in ("writer-1", "writer-2")
    # No leftover .tmp files after both writers finish.
    leftover_tmp_files = list(tmp_path.glob("*.tmp"))
    assert leftover_tmp_files == []
