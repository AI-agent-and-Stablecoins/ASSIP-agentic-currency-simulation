"""One function per hypothesis: builds that hypothesis's dataset (`src.
econometrics.hypothesis_datasets`) and fits it (`src.econometrics
.regression_engine.fit_clustered_logit`), per docs/superpowers/specs/
2026-08-02-phase3-plan5-econometrics-design.md.
"""

from sqlalchemy.orm import Session

from src.econometrics.hypothesis_datasets import build_h1_dataset
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
