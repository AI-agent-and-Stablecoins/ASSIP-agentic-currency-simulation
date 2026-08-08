from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

# NOTE: the installed streamlit version (1.61.1) resolves AppTest.from_file's
# relative paths against the directory of the *calling test file* (tests/),
# not the CWD -- so the plain "dashboard/app.py" literal from the task brief
# resolves to the nonexistent tests/dashboard/app.py. Using an absolute path
# sidesteps that resolution entirely while still exercising the real script.
APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


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
