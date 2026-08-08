"""One end-to-end test exercising all five dashboard modules together:
process_control.start() spawns dashboard/runner.py as a real subprocess,
which writes to status_store's status file and to an isolated SQLite
database; this test polls status_store.read_status() for completion and
then reads the same database back through queries.get_progress_for_run(),
matching what dashboard/app.py itself does for the Status and Live
progress panels.
"""

import time

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from dashboard import status_store
from dashboard.process_control import RunConfig, start, stop
from dashboard.queries import get_progress_for_run
from src.utils.constants import REPO_ROOT

_POLL_TIMEOUT_SECONDS = 60
_POLL_INTERVAL_SECONDS = 0.5


def test_end_to_end_dry_run_completes_and_is_visible_in_live_progress(tmp_path, monkeypatch):
    # NOTE: deliberately NOT monkeypatching status_store._STATE_DIR here.
    # start() spawns dashboard/runner.py as a real, separate OS process,
    # which re-imports status_store fresh and so can never see this test
    # process's own monkeypatch -- it always writes its "running" ->
    # "completed" progression to the REAL dashboard/state/ directory. If
    # this test's own poll loop read from a patched tmp_path instead, it
    # would only ever observe start()'s own initial write and never the
    # child's completion, hanging until the timeout. Matches the same
    # accepted pattern already used in tests/test_dashboard_process_control.py
    # and tests/test_dashboard_runner.py: use the real directory, then clean
    # up the one specific known file afterward.
    db_path = tmp_path / "e2e_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    matrix_run_id = "dashboard-end-to-end-test"
    config = RunConfig(
        matrix_run_id=matrix_run_id,
        cell_keys=["master"],
        seeds=[0],
        num_days=1,
        dry_run=True,
    )

    try:
        start(config)

        deadline = time.time() + _POLL_TIMEOUT_SECONDS
        status = None
        while time.time() < deadline:
            status = status_store.read_status(matrix_run_id)
            if status is not None and status.get("state") in ("completed", "failed", "crashed"):
                break
            time.sleep(_POLL_INTERVAL_SECONDS)

        assert status is not None, "no status was ever written for this matrix_run_id"
        assert status["state"] == "completed", (
            f"expected 'completed' within {_POLL_TIMEOUT_SECONDS}s but got "
            f"{status.get('state')!r} (error={status.get('error')!r})"
        )
        assert status["failures"] == []

        # Read the SAME database the subprocess wrote to, exactly as
        # dashboard/app.py's "Live progress" section does via
        # database.session.new_session() -- here constructed directly
        # against the isolated tmp_path db file instead of relying on the
        # module-level engine (which is fixed at first import time, before
        # this test's DATABASE_URL monkeypatch could take effect).
        engine = create_engine(f"sqlite:///{db_path}")
        session = Session(engine)
        try:
            progress = get_progress_for_run(session, matrix_run_id)
        finally:
            session.close()

        assert len(progress) == 1
        row = progress[0]
        assert row.cell_key == "master"
        assert row.seed == 0
        assert row.current_day == 0  # num_days=1 -> a single timestep, index 0
    finally:
        stop(matrix_run_id)
        (REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.json").unlink(missing_ok=True)
        (REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.log").unlink(missing_ok=True)
