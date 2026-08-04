"""Shared logistic-regression fitting for every H1-H5 hypothesis, per
docs/superpowers/specs/2026-08-02-phase3-plan5-econometrics-design.md Sec 3:
per-decision logit, agent-clustered standard errors, McFadden pseudo-R^2/
adjusted pseudo-R^2 in place of OLS's R^2/adjusted R^2 (undefined for a
binary outcome).
"""

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class RegressionResult:
    hypothesis: str
    regressor: str
    beta: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    pseudo_r2: float
    adjusted_pseudo_r2: float
    n_obs: int


def fit_clustered_logit(
    hypothesis: str,
    df: pd.DataFrame,
    dependent_col: str,
    regressor_col: str,
    cluster_col: str,
    fixed_effect_cols: list[str],
) -> RegressionResult:
    """Fits `dependent_col ~ regressor_col + <fixed_effect_cols dummies>`
    via logistic regression with standard errors clustered by
    `cluster_col` (agent-level, per the design spec's Sec 0 decision).
    Returns `regressor_col`'s own coefficient/SE/CI/p-value plus the whole
    model's McFadden pseudo-R^2/adjusted pseudo-R^2 and sample size.
    Raises `ValueError` if `df` is empty, if `dependent_col` has fewer than
    2 distinct values, or if `regressor_col` has fewer than 2 distinct
    values (Plan 5 whole-branch review Fix I4). A hypothesis's dataset
    builder finding no eligible decisions, or finding decisions that never
    vary on the outcome/regressor, is a real problem to surface loudly --
    left unguarded, a constant dependent variable drives `result.llnull`
    to 0 and `adjusted_pseudo_r2`/`pseudo_r2` silently become `inf`/`nan`
    (confirmed by direct testing) rather than raising, and a constant
    regressor/design-matrix column can raise an opaque `numpy.linalg
    .LinAlgError: Singular matrix` deep inside statsmodels instead of a
    clear, hypothesis-scoped message.
    """
    if df.empty:
        raise ValueError(
            f"fit_clustered_logit({hypothesis!r}): received an empty DataFrame -- "
            "the dataset builder found no eligible decisions for this hypothesis."
        )
    if df[dependent_col].nunique() < 2:
        raise ValueError(
            f"fit_clustered_logit({hypothesis!r}): {dependent_col!r} has no variation "
            f"(all {df[dependent_col].iloc[0]!r}) -- cannot fit a logistic regression on a constant outcome."
        )
    if df[regressor_col].nunique() < 2:
        raise ValueError(
            f"fit_clustered_logit({hypothesis!r}): {regressor_col!r} has no variation "
            f"(all {df[regressor_col].iloc[0]!r}) -- cannot estimate its coefficient against a constant regressor."
        )

    y = df[dependent_col].astype(float)
    x_numeric = df[[regressor_col]].astype(float)
    if fixed_effect_cols:
        x_dummies = pd.get_dummies(df[fixed_effect_cols], drop_first=True, dtype=float)
        x = pd.concat([x_numeric, x_dummies], axis=1)
    else:
        x = x_numeric
    x = sm.add_constant(x)

    model = sm.Logit(y, x)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]}, disp=0)

    ci = result.conf_int().loc[regressor_col]
    # McFadden's adjusted R^2 uses K = total estimated parameters INCLUDING
    # the intercept; statsmodels' result.df_model excludes it (it's the
    # regressor count only), so +1 corrects for the constant sm.add_constant
    # adds above.
    num_params_including_intercept = result.df_model + 1
    adjusted_pseudo_r2 = 1.0 - (result.llf - num_params_including_intercept) / result.llnull

    return RegressionResult(
        hypothesis=hypothesis,
        regressor=regressor_col,
        beta=float(result.params[regressor_col]),
        se=float(result.bse[regressor_col]),
        ci_lower=float(ci.iloc[0]),
        ci_upper=float(ci.iloc[1]),
        p_value=float(result.pvalues[regressor_col]),
        pseudo_r2=float(result.prsquared),
        adjusted_pseudo_r2=float(adjusted_pseudo_r2),
        n_obs=int(result.nobs),
    )
