from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import status_store

# NOTE: the installed streamlit version (1.61.1) resolves AppTest.from_file's
# relative paths against the directory of the *calling test file* (tests/),
# not the CWD -- so the plain "dashboard/app.py" literal from the task brief
# resolves to the nonexistent tests/dashboard/app.py. Using an absolute path
# sidesteps that resolution entirely while still exercising the real script.
APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    # Without this, every test here reads/writes the REAL dashboard/state/
    # directory (and, via status_store.get_active_run_id(), can pick up
    # whatever matrix_run_id a real prior run left behind) -- matching the
    # isolation pattern already used in tests/test_dashboard_process_control.py.
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    yield


def test_app_renders_the_start_configuration_form():
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    # The matrix_run_id text input and the dry-run toggle must both be present.
    assert any("matrix_run_id" in inp.label.lower() or "run id" in inp.label.lower() for inp in at.text_input)
    assert any("dry" in cb.label.lower() for cb in at.checkbox)


def test_clicking_start_in_dry_run_mode_calls_process_control_start_without_extra_confirmation():
    with patch("dashboard.process_control.start") as mock_start:
        at = AppTest.from_file(APP_PATH)
        at.run()

        # Dry run is the default state -- clicking Start should call
        # process_control.start directly, no confirmation step required.
        start_buttons = [b for b in at.button if b.label.lower() == "start"]
        assert len(start_buttons) == 1
        start_buttons[0].click().run()

        assert mock_start.called


def test_turning_dry_run_off_requires_typing_the_matrix_run_id_before_a_real_launch_is_possible():
    with patch("dashboard.process_control.start") as mock_start:
        at = AppTest.from_file(APP_PATH)
        at.run()

        dry_run_checkboxes = [cb for cb in at.checkbox if "dry" in cb.label.lower()]
        assert len(dry_run_checkboxes) == 1
        dry_run_checkboxes[0].uncheck().run()

        # With dry_run off, the plain "Start" button must be gone/disabled,
        # replaced by a confirmation flow -- clicking whatever primary
        # action button remains, WITHOUT having typed the confirmation
        # text, must not launch anything.
        confirm_buttons = [b for b in at.button if "confirm" in b.label.lower()]
        assert len(confirm_buttons) == 1
        confirm_buttons[0].click().run()

        assert not mock_start.called


def test_typing_the_matching_matrix_run_id_and_confirming_actually_launches_a_real_run():
    with patch("dashboard.process_control.start") as mock_start:
        at = AppTest.from_file(APP_PATH)
        at.run()

        # The run id field itself is labeled exactly "matrix_run_id" -- the
        # confirmation field's label also contains that substring ("Type the
        # matrix_run_id above to confirm..."), so match on the exact label to
        # avoid grabbing the wrong widget.
        run_id_inputs = [inp for inp in at.text_input if inp.label == "matrix_run_id"]
        assert len(run_id_inputs) == 1
        matrix_run_id = run_id_inputs[0].value

        dry_run_checkboxes = [cb for cb in at.checkbox if "dry" in cb.label.lower()]
        assert len(dry_run_checkboxes) == 1
        dry_run_checkboxes[0].uncheck().run()

        confirmation_inputs = [inp for inp in at.text_input if "confirm" in inp.label.lower()]
        assert len(confirmation_inputs) == 1
        confirmation_inputs[0].set_value(matrix_run_id).run()

        confirm_buttons = [b for b in at.button if "confirm" in b.label.lower()]
        assert len(confirm_buttons) == 1
        confirm_buttons[0].click().run()

        assert not at.exception
        assert mock_start.called


# --- Finding 1: Resume on a real (non-dry-run) tracked run must require confirmation too ---


def test_resume_on_a_previously_real_run_requires_confirmation_like_start():
    with patch("dashboard.process_control.resume") as mock_resume:
        at = AppTest.from_file(APP_PATH)
        at.run()

        run_id_inputs = [inp for inp in at.text_input if inp.label == "matrix_run_id"]
        matrix_run_id = run_id_inputs[0].value

        # Simulate a previously-tracked REAL (non-dry-run) run that was
        # stopped -- Resume must not be a single bare click for this.
        status_store.write_status(matrix_run_id, pid=999_999_999, state="stopped", dry_run=False)
        at.run()

        bare_resume_buttons = [b for b in at.button if b.label.lower() == "resume"]
        assert bare_resume_buttons == [], "a bare 'Resume' button must not be offered for a real tracked run"

        confirm_resume_buttons = [
            b for b in at.button if "confirm" in b.label.lower() and "resume" in b.label.lower()
        ]
        assert len(confirm_resume_buttons) == 1

        # Clicking confirm WITHOUT typing the matching id must not resume.
        confirm_resume_buttons[0].click().run()
        assert not mock_resume.called

        # Typing the matching matrix_run_id and confirming DOES resume.
        resume_confirmation_inputs = [
            inp for inp in at.text_input if "confirm" in inp.label.lower() and "resume" in inp.label.lower()
        ]
        assert len(resume_confirmation_inputs) == 1
        resume_confirmation_inputs[0].set_value(matrix_run_id).run()

        confirm_resume_buttons = [
            b for b in at.button if "confirm" in b.label.lower() and "resume" in b.label.lower()
        ]
        confirm_resume_buttons[0].click().run()

        assert not at.exception
        assert mock_resume.called


def test_resume_on_a_previously_dry_run_tracked_run_needs_no_extra_confirmation():
    """Contrast case for Finding 1: a tracked run that WAS a dry run may
    still be resumed with a single click -- the extra confirmation gate is
    specific to a real (paid) underlying run."""
    with patch("dashboard.process_control.resume") as mock_resume:
        at = AppTest.from_file(APP_PATH)
        at.run()

        run_id_inputs = [inp for inp in at.text_input if inp.label == "matrix_run_id"]
        matrix_run_id = run_id_inputs[0].value

        status_store.write_status(matrix_run_id, pid=999_999_999, state="stopped", dry_run=True)
        at.run()

        bare_resume_buttons = [b for b in at.button if b.label.lower() == "resume"]
        assert len(bare_resume_buttons) == 1
        bare_resume_buttons[0].click().run()

        assert not at.exception
        assert mock_resume.called


# --- Finding 3: an empty "Specific cells" selection must block launch, not run all 13 cells ---


def test_specific_cells_with_no_selection_blocks_launch_with_an_error():
    with patch("dashboard.process_control.start") as mock_start:
        at = AppTest.from_file(APP_PATH)
        at.run()

        cell_scope_radios = [r for r in at.radio if "cell scope" in r.label.lower()]
        assert len(cell_scope_radios) == 1
        cell_scope_radios[0].set_value("Specific cells").run()

        # Multiselect defaults to an empty selection -- Start must not even
        # be offered in this state, only an explanatory error.
        start_buttons = [b for b in at.button if b.label.lower() == "start"]
        assert start_buttons == []
        assert any("select at least one cell" in msg.value.lower() for msg in at.error)
        assert not mock_start.called


def test_specific_cells_with_a_selection_allows_launch():
    with patch("dashboard.process_control.start") as mock_start:
        at = AppTest.from_file(APP_PATH)
        at.run()

        cell_scope_radios = [r for r in at.radio if "cell scope" in r.label.lower()]
        cell_scope_radios[0].set_value("Specific cells").run()

        cell_multiselects = [m for m in at.multiselect if m.label.lower() == "cells"]
        assert len(cell_multiselects) == 1
        cell_multiselects[0].select("master").run()

        start_buttons = [b for b in at.button if b.label.lower() == "start"]
        assert len(start_buttons) == 1
        start_buttons[0].click().run()

        assert not at.exception
        assert mock_start.called


# --- Finding 9: an invalid matrix_run_id must block launch with a friendly error ---


def test_invalid_matrix_run_id_blocks_launch_with_an_error():
    with patch("dashboard.process_control.start") as mock_start:
        at = AppTest.from_file(APP_PATH)
        at.run()

        run_id_inputs = [inp for inp in at.text_input if inp.label == "matrix_run_id"]
        assert len(run_id_inputs) == 1
        run_id_inputs[0].set_value("../../escape-attempt").run()

        start_buttons = [b for b in at.button if b.label.lower() == "start"]
        assert start_buttons == []
        assert any("invalid" in msg.value.lower() for msg in at.error)
        assert not at.exception
        assert not mock_start.called


# --- Finding 6: a corrupted status file must not crash Stop/Pause/Resume handlers ---


def test_clicking_stop_with_a_corrupted_status_file_shows_an_error_instead_of_crashing():
    at = AppTest.from_file(APP_PATH)
    at.run()

    run_id_inputs = [inp for inp in at.text_input if inp.label == "matrix_run_id"]
    matrix_run_id = run_id_inputs[0].value

    status_path = status_store.state_dir() / f"{matrix_run_id}.json"
    status_path.write_text("not valid json{{{", encoding="utf-8")
    at.run()
    assert not at.exception

    stop_buttons = [b for b in at.button if b.label.lower() == "stop"]
    assert len(stop_buttons) == 1
    stop_buttons[0].click().run()

    assert not at.exception
    assert any("corrupted" in msg.value.lower() for msg in at.error)


def test_clicking_pause_with_a_corrupted_status_file_shows_an_error_instead_of_crashing():
    at = AppTest.from_file(APP_PATH)
    at.run()

    run_id_inputs = [inp for inp in at.text_input if inp.label == "matrix_run_id"]
    matrix_run_id = run_id_inputs[0].value

    status_path = status_store.state_dir() / f"{matrix_run_id}.json"
    status_path.write_text("not valid json{{{", encoding="utf-8")
    at.run()
    assert not at.exception

    pause_buttons = [b for b in at.button if b.label.lower() == "pause"]
    assert len(pause_buttons) == 1
    pause_buttons[0].click().run()

    assert not at.exception
    assert any("corrupted" in msg.value.lower() for msg in at.error)


# --- Finding 12: the Pause button must carry a tooltip explaining it behaves like Stop ---


def test_pause_button_has_a_tooltip_explaining_it_behaves_like_stop():
    at = AppTest.from_file(APP_PATH)
    at.run()

    pause_buttons = [b for b in at.button if b.label.lower() == "pause"]
    assert len(pause_buttons) == 1
    assert "stop" in pause_buttons[0].help.lower()
    assert "resume" in pause_buttons[0].help.lower()


# --- Re-review finding: the live-progress fragment must not leak a SQLAlchemy
# Session on every 5s auto-refresh tick ---


def test_render_live_progress_does_not_leak_sqlalchemy_sessions_across_repeated_reruns():
    """Regression for the re-review finding: `_render_live_progress`
    previously called `database.session.new_session()` and never closed it.
    Run every 5s via `@st.fragment(run_every="5s")`, this could exhaust the
    default SQLAlchemy QueuePool (pool_size=5 + max_overflow=10 = 15
    outstanding connections) after roughly 75s of the page being open
    (reproduced directly: `QueuePool limit ... reached` after ~15 unclosed
    sessions).

    `AppTest.run()` re-executes the whole script on every call -- including
    the fragment function's own direct call at the bottom of app.py --
    exactly like a real Streamlit rerun would, so calling it repeatedly here
    reproduces the same leak pattern `run_every="5s"` would produce over
    time. Asserting the pool's checked-out count returns to 0 after EVERY
    rerun (not just that 15 reruns don't blow the pool) catches the leak on
    the very first iteration for the pre-fix code, since it never closed a
    session at all.
    """
    from database.session import get_engine

    engine = get_engine()
    at = AppTest.from_file(APP_PATH)

    for _ in range(20):
        at.run()
        assert not at.exception
        assert engine.pool.checkedout() == 0, (
            "a SQLAlchemy Session was left checked out after the live-progress "
            "fragment ran -- it must close its session every rerun, not just "
            "on some of them"
        )
