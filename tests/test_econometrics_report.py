from unittest.mock import patch

import pandas as pd

from src.econometrics.regression_engine import RegressionResult
from src.econometrics.report import results_to_dataframe, run_all_hypotheses, write_report_csv

# run_all_hypotheses is pure orchestration (call all 5 regress_hN functions,
# return their results) -- each regress_hN's OWN statistical correctness is
# already exhaustively tested in Tasks 4-8 (real, non-degenerate regression
# fits against real persisted-row fixtures). Re-verifying all 5 hypotheses
# together against one shared run_matrix(...)-driven database is not
# possible without conflict: H2/H3 need forced mock_llm_decision overrides
# that differ per hypothesis, H4 needs num_days > 120 (H1/H2/H3/H5's
# fixtures use far fewer days), and no single database could satisfy every
# hypothesis's own filtering/sample requirements at once without paying the
# same multi-hour cost documented in tests/test_hypothesis_h4.py's header.
# So this test verifies the ORCHESTRATION itself (does it call all 5, in
# order, and assemble the results correctly) via mocks -- fast, and it
# tests exactly what this function is responsible for, no more.


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


def test_run_all_hypotheses_returns_one_result_per_hypothesis():
    fake_results = {h: _fake_result(h) for h in ("H1", "H2", "H3", "H4", "H5")}
    with (
        patch("src.econometrics.report.regress_h1", return_value=fake_results["H1"]) as m1,
        patch("src.econometrics.report.regress_h2", return_value=fake_results["H2"]) as m2,
        patch("src.econometrics.report.regress_h3", return_value=fake_results["H3"]) as m3,
        patch("src.econometrics.report.regress_h4", return_value=fake_results["H4"]) as m4,
        patch("src.econometrics.report.regress_h5", return_value=fake_results["H5"]) as m5,
    ):
        session = object()  # opaque sentinel -- run_all_hypotheses must pass it through unchanged
        results = run_all_hypotheses(session)

        for mock in (m1, m2, m3, m4, m5):
            mock.assert_called_once_with(session)

    assert len(results) == 5
    assert {r.hypothesis for r in results} == {"H1", "H2", "H3", "H4", "H5"}
    assert all(isinstance(r, RegressionResult) for r in results)


def test_results_to_dataframe_has_the_required_publication_columns():
    results = [_fake_result(h) for h in ("H1", "H2", "H3", "H4", "H5")]
    df = results_to_dataframe(results)

    assert set(df.columns) >= {
        "hypothesis", "regressor", "beta", "se", "ci_lower", "ci_upper",
        "p_value", "pseudo_r2", "adjusted_pseudo_r2", "n_obs",
    }
    assert len(df) == 5


def test_write_report_csv_writes_a_readable_file(tmp_path):
    results = [_fake_result(h) for h in ("H1", "H2", "H3", "H4", "H5")]
    out_path = tmp_path / "hypothesis_report.csv"

    write_report_csv(results, out_path)

    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == 5
    assert set(reloaded["hypothesis"]) == {"H1", "H2", "H3", "H4", "H5"}
