from unittest.mock import patch

import pandas as pd

from src.econometrics.regression_engine import RegressionResult
from src.econometrics.report import results_to_dataframe, run_all_hypotheses, write_report_csv

# run_all_hypotheses is pure orchestration (call every regress_hN function,
# return their results -- 15 results total: H1-H5 pooled once each, H7-H11
# each called twice for domestic/cross_border) -- each regress_hN's OWN
# statistical correctness is already exhaustively tested elsewhere (real,
# non-degenerate regression fits against real persisted-row fixtures).
# Re-verifying every hypothesis together against one shared
# run_matrix(...)-driven database is not possible without conflict: H2/H3
# need forced mock_llm_decision overrides that differ per hypothesis, H4
# needs num_days > 120 (H1/H2/H3/H5's fixtures use far fewer days), and no
# single database could satisfy every hypothesis's own filtering/sample
# requirements at once without paying the same multi-hour cost documented
# in tests/test_hypothesis_h4.py's header. So this test verifies the
# ORCHESTRATION itself (does it call every hypothesis, with the right
# cell_variant fan-out, and assemble the results correctly) via mocks --
# fast, and it tests exactly what this function is responsible for, no more.


def _fake_result(hypothesis: str) -> RegressionResult:
    return RegressionResult(
        hypothesis=hypothesis,
        regressor="test_regressor",
        beta=0.5,
        se=0.1,
        ci_lower=0.3,
        ci_upper=0.7,
        p_value=0.01,
        pseudo_r2=0.2,
        adjusted_pseudo_r2=0.15,
        n_obs=100,
    )


_ALL_HYPOTHESIS_LABELS = (
    "H1", "H2", "H3", "H4", "H5",
    "H7_domestic", "H7_cross_border",
    "H8_domestic", "H8_cross_border",
    "H9_domestic", "H9_cross_border",
    "H10_domestic", "H10_cross_border",
    "H11_domestic", "H11_cross_border",
)


def test_run_all_hypotheses_returns_one_result_per_hypothesis():
    fake_results = {h: _fake_result(h) for h in _ALL_HYPOTHESIS_LABELS}
    with (
        patch("src.econometrics.report.regress_h1", return_value=fake_results["H1"]) as m1,
        patch("src.econometrics.report.regress_h2", return_value=fake_results["H2"]) as m2,
        patch("src.econometrics.report.regress_h3", return_value=fake_results["H3"]) as m3,
        patch("src.econometrics.report.regress_h4", return_value=fake_results["H4"]) as m4,
        patch("src.econometrics.report.regress_h5", return_value=fake_results["H5"]) as m5,
        patch("src.econometrics.report.regress_h7", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H7_{cell_variant}"]) as m7,
        patch("src.econometrics.report.regress_h8", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H8_{cell_variant}"]) as m8,
        patch("src.econometrics.report.regress_h9", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H9_{cell_variant}"]) as m9,
        patch("src.econometrics.report.regress_h10", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H10_{cell_variant}"]) as m10,
        patch("src.econometrics.report.regress_h11", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H11_{cell_variant}"]) as m11,
    ):
        session = object()  # opaque sentinel -- run_all_hypotheses must pass it through unchanged
        results = run_all_hypotheses(session)

        for mock in (m1, m2, m3, m4, m5):
            mock.assert_called_once_with(session, matrix_run_id=None)
        for mock in (m7, m8, m9, m10, m11):
            assert mock.call_count == 2  # domestic + cross_border

    assert len(results) == 15
    assert {r.hypothesis for r in results} == set(_ALL_HYPOTHESIS_LABELS)
    assert all(isinstance(r, RegressionResult) for r in results)


def test_run_all_hypotheses_threads_matrix_run_id_to_every_hypothesis():
    """Plan 5 whole-branch review Fix C3, closing the gap the first review
    pass missed: run_all_hypotheses is the one production entry point the
    real report is expected to call, so an explicit matrix_run_id must
    reach every regress_hN -- not just the individual build_hN_dataset/
    regress_hN functions in isolation."""
    fake_results = {h: _fake_result(h) for h in _ALL_HYPOTHESIS_LABELS}
    with (
        patch("src.econometrics.report.regress_h1", return_value=fake_results["H1"]) as m1,
        patch("src.econometrics.report.regress_h2", return_value=fake_results["H2"]) as m2,
        patch("src.econometrics.report.regress_h3", return_value=fake_results["H3"]) as m3,
        patch("src.econometrics.report.regress_h4", return_value=fake_results["H4"]) as m4,
        patch("src.econometrics.report.regress_h5", return_value=fake_results["H5"]) as m5,
        patch("src.econometrics.report.regress_h7", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H7_{cell_variant}"]) as m7,
        patch("src.econometrics.report.regress_h8", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H8_{cell_variant}"]) as m8,
        patch("src.econometrics.report.regress_h9", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H9_{cell_variant}"]) as m9,
        patch("src.econometrics.report.regress_h10", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H10_{cell_variant}"]) as m10,
        patch("src.econometrics.report.regress_h11", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H11_{cell_variant}"]) as m11,
    ):
        session = object()
        run_all_hypotheses(session, matrix_run_id="phase3-real-run-2026-08-04")

        for mock in (m1, m2, m3, m4, m5):
            mock.assert_called_once_with(session, matrix_run_id="phase3-real-run-2026-08-04")
        for mock in (m7, m8, m9, m10, m11):
            mock.assert_any_call(session, cell_variant="domestic", matrix_run_id="phase3-real-run-2026-08-04")
            mock.assert_any_call(session, cell_variant="cross_border", matrix_run_id="phase3-real-run-2026-08-04")
            assert mock.call_count == 2


def test_results_to_dataframe_has_the_required_publication_columns():
    results = [_fake_result(h) for h in _ALL_HYPOTHESIS_LABELS]
    df = results_to_dataframe(results)

    assert set(df.columns) >= {
        "hypothesis", "regressor", "beta", "se", "ci_lower", "ci_upper",
        "p_value", "pseudo_r2", "adjusted_pseudo_r2", "n_obs",
    }
    assert len(df) == 15


def test_write_report_csv_writes_a_readable_file(tmp_path):
    results = [_fake_result(h) for h in _ALL_HYPOTHESIS_LABELS]
    out_path = tmp_path / "hypothesis_report.csv"

    write_report_csv(results, out_path)

    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == 15
    assert set(reloaded["hypothesis"]) == set(_ALL_HYPOTHESIS_LABELS)
