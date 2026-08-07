"""One function per hypothesis: builds that hypothesis's dataset (`src.
econometrics.hypothesis_datasets`) and fits it (`src.econometrics
.regression_engine.fit_clustered_logit`), per docs/superpowers/specs/
2026-08-02-phase3-plan5-econometrics-design.md.
"""

from sqlalchemy.orm import Session

from src.econometrics.hypothesis_datasets import (
    build_h1_dataset,
    build_h2_dataset,
    build_h3_dataset,
    build_h4_dataset,
    build_h5_dataset,
    build_h7_dataset,
    build_h8_dataset,
    build_h9_dataset,
    build_h10_dataset,
    build_h11_dataset,
)
from src.econometrics.regression_engine import RegressionResult, fit_clustered_logit


def regress_h1(session: Session, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h1_dataset(session, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis="H1",
        df=df,
        dependent_col="chose_usd_zone",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h2(session: Session, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h2_dataset(session, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis="H2",
        df=df,
        dependent_col="chose_spread_optimal",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h3(session: Session, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h3_dataset(session, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis="H3",
        df=df,
        dependent_col="chose_higher_governance",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model", "cell_key"],
    )


def regress_h4(session: Session, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h4_dataset(session, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis="H4",
        df=df,
        dependent_col="chose_gold",
        regressor_col="proximity_days",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model", "cell_key"],
    )


def regress_h5(session: Session, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h5_dataset(session, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis="H5",
        df=df,
        dependent_col="chose_usd_zone",
        regressor_col="eur_usd_volatility",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h7(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h7_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H7_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h8(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h8_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H8_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h9(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h9_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H9_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h10(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h10_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H10_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h11(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h11_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H11_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )
