"""Assembles all 5 in-scope hypotheses' regression results (H6 is
deferred, per docs/superpowers/specs/2026-07-29-phase3-full-scale-
simulation-design.md Sec 7) into one output table.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.econometrics.hypothesis_regressions import regress_h1, regress_h2, regress_h3, regress_h4, regress_h5
from src.econometrics.regression_engine import RegressionResult


def run_all_hypotheses(session: Session, matrix_run_id: str | None = None) -> list[RegressionResult]:
    """Runs every in-scope hypothesis's regression against `session`'s
    already-persisted matrix-runner data. Each hypothesis's own dataset
    builder independently filters to its own relevant cell(s) -- a
    hypothesis whose sample turns out empty raises `ValueError` from
    `fit_clustered_logit` rather than silently omitting itself, so a
    misconfigured run surfaces loudly instead of shipping a report
    missing a hypothesis with no explanation.

    `matrix_run_id`, if given, scopes every hypothesis to one `run_matrix`
    invocation (Plan 5 whole-branch review Fix C3) -- without it, a
    database holding more than one `run_matrix` call (e.g. a dry-run smoke
    test followed by the real run) silently pools all of them together.
    This is the one production entry point the eventual real report is
    expected to call, so it must thread the same scoping every `build_
    hN_dataset`/`regress_hN` already supports, not just those functions
    in isolation.
    """
    return [
        regress_h1(session, matrix_run_id=matrix_run_id),
        regress_h2(session, matrix_run_id=matrix_run_id),
        regress_h3(session, matrix_run_id=matrix_run_id),
        regress_h4(session, matrix_run_id=matrix_run_id),
        regress_h5(session, matrix_run_id=matrix_run_id),
    ]


def results_to_dataframe(results: list[RegressionResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis": r.hypothesis,
                "regressor": r.regressor,
                "beta": r.beta,
                "se": r.se,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "p_value": r.p_value,
                "pseudo_r2": r.pseudo_r2,
                "adjusted_pseudo_r2": r.adjusted_pseudo_r2,
                "n_obs": r.n_obs,
            }
            for r in results
        ]
    )


def write_report_csv(results: list[RegressionResult], path: Path) -> None:
    results_to_dataframe(results).to_csv(path, index=False)
