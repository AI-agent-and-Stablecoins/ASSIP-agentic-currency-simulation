"""Assembles all in-scope hypotheses' regression results into one output
table: H1-H5 (per docs/superpowers/specs/2026-07-29-phase3-full-scale-
simulation-design.md Sec 7 -- that document's own H6 is a separate,
still-deferred privacy hypothesis, NOT related to H7-H11 below) plus
H7-H11 (the 5 sandbox-preference hypotheses added by Plan 6, per
docs/superpowers/specs/2026-08-04-phase3-plan6-concurrency-and-sandbox-
hypotheses-design.md Sec 1 -- numbered starting at H7, not H6, specifically
to avoid colliding with the master spec's existing deferred H6).
"""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.legacy.econometrics.hypothesis_regressions import (
    regress_h1,
    regress_h2,
    regress_h3,
    regress_h4,
    regress_h5,
    regress_h7,
    regress_h8,
    regress_h9,
    regress_h10,
    regress_h11,
)
from src.legacy.econometrics.regression_engine import RegressionResult


def run_all_hypotheses(session: Session, matrix_run_id: str | None = None) -> list[RegressionResult]:
    """Runs every in-scope hypothesis's regression against `session`'s
    already-persisted matrix-runner data. Each hypothesis's own dataset
    builder independently filters to its own relevant cell(s) -- a
    hypothesis whose sample turns out empty raises `ValueError` from
    `fit_clustered_logit` rather than silently omitting itself, so a
    misconfigured run surfaces loudly instead of shipping a report
    missing a hypothesis with no explanation.

    H1-H5 each return one pooled result; H7-H11 each return two
    (domestic, cross_border), reported separately per Plan 6 design spec
    Sec 1 -- 15 results total.

    `matrix_run_id`, if given, scopes every hypothesis to one `run_matrix`
    invocation (Plan 5 whole-branch review Fix C3) -- without it, a
    database holding more than one `run_matrix` call (e.g. a dry-run smoke
    test followed by the real run) silently pools all of them together.
    This is the one production entry point the eventual real report is
    expected to call, so it must thread the same scoping every `build_
    hN_dataset`/`regress_hN` already supports, not just those functions
    in isolation.
    """
    results = [
        regress_h1(session, matrix_run_id=matrix_run_id),
        regress_h2(session, matrix_run_id=matrix_run_id),
        regress_h3(session, matrix_run_id=matrix_run_id),
        regress_h4(session, matrix_run_id=matrix_run_id),
        regress_h5(session, matrix_run_id=matrix_run_id),
    ]
    for regress_fn in (regress_h7, regress_h8, regress_h9, regress_h10, regress_h11):
        results.append(regress_fn(session, cell_variant="domestic", matrix_run_id=matrix_run_id))
        results.append(regress_fn(session, cell_variant="cross_border", matrix_run_id=matrix_run_id))
    return results


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
