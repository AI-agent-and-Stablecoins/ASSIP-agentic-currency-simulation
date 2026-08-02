"""One function per hypothesis: builds that hypothesis's dataset (`src.
econometrics.hypothesis_datasets`) and fits it (`src.econometrics
.regression_engine.fit_clustered_logit`), per docs/superpowers/specs/
2026-08-02-phase3-plan5-econometrics-design.md.
"""

from sqlalchemy.orm import Session

from src.econometrics.hypothesis_datasets import build_h1_dataset, build_h2_dataset, build_h3_dataset
from src.econometrics.regression_engine import RegressionResult, fit_clustered_logit


def regress_h1(session: Session) -> RegressionResult:
    df = build_h1_dataset(session)
    return fit_clustered_logit(
        hypothesis="H1",
        df=df,
        dependent_col="chose_usd_zone",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h2(session: Session) -> RegressionResult:
    df = build_h2_dataset(session)
    return fit_clustered_logit(
        hypothesis="H2",
        df=df,
        dependent_col="chose_spread_optimal",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h3(session: Session) -> RegressionResult:
    df = build_h3_dataset(session)
    return fit_clustered_logit(
        hypothesis="H3",
        df=df,
        dependent_col="chose_higher_governance",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model", "cell_key"],
    )
