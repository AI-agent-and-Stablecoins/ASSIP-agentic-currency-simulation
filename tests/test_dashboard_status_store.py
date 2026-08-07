import pytest

from dashboard.status_store import (
    StatusFileCorruptedError,
    get_active_run_id,
    read_status,
    set_active_run_id,
    state_dir,
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
