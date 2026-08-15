import numpy as np
import pandas as pd
import pytest

from src.legacy.econometrics.regression_engine import RegressionResult, fit_clustered_logit


def _synthetic_dataset(n_agents: int = 50, decisions_per_agent: int = 20, seed: int = 0) -> pd.DataFrame:
    """Builds a dataset where higher `regressor` genuinely makes `chose_x`
    more likely (a real logistic relationship, not noise), with several
    repeated decisions per agent -- exercising both the regression fit
    itself and the agent-clustering machinery."""
    rng = np.random.default_rng(seed)
    rows = []
    for agent_idx in range(n_agents):
        agent_regressor = rng.uniform(-2.0, 2.0)
        agent_type = "consumer" if agent_idx % 2 == 0 else "bank"
        model = "vendor/model-a" if agent_idx % 3 == 0 else "vendor/model-b"
        for _ in range(decisions_per_agent):
            probability = 1.0 / (1.0 + np.exp(-(2.0 * agent_regressor)))
            chose_x = 1 if rng.uniform(0.0, 1.0) < probability else 0
            rows.append(
                {
                    "agent_id": f"agent-{agent_idx}",
                    "chose_x": chose_x,
                    "regressor": agent_regressor,
                    "agent_type": agent_type,
                    "actual_model": model,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_fit_clustered_logit_recovers_a_positive_relationship():
    df = _synthetic_dataset()
    result = fit_clustered_logit(
        hypothesis="H_TEST",
        df=df,
        dependent_col="chose_x",
        regressor_col="regressor",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )

    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H_TEST"
    assert result.regressor == "regressor"
    assert result.beta > 0  # the synthetic data has a genuine positive relationship
    assert result.p_value < 0.05  # should be clearly statistically significant given the sample size
    assert result.ci_lower < result.beta < result.ci_upper
    assert 0.0 <= result.pseudo_r2 <= 1.0
    assert result.n_obs == len(df)


def test_fit_clustered_logit_works_with_no_fixed_effects():
    df = _synthetic_dataset(n_agents=30, decisions_per_agent=10, seed=1)
    result = fit_clustered_logit(
        hypothesis="H_TEST2",
        df=df,
        dependent_col="chose_x",
        regressor_col="regressor",
        cluster_col="agent_id",
        fixed_effect_cols=[],
    )
    assert result.n_obs == len(df)
    assert result.beta > 0  # same genuine positive relationship as the main test
    assert result.p_value < 0.05


def test_fit_clustered_logit_raises_on_empty_dataframe():
    df = pd.DataFrame(columns=["agent_id", "chose_x", "regressor", "agent_type", "actual_model"])
    with pytest.raises(ValueError):
        fit_clustered_logit(
            hypothesis="H_EMPTY",
            df=df,
            dependent_col="chose_x",
            regressor_col="regressor",
            cluster_col="agent_id",
            fixed_effect_cols=["agent_type"],
        )


def test_fit_clustered_logit_raises_on_constant_dependent_variable():
    df = _synthetic_dataset(n_agents=10, decisions_per_agent=5, seed=2)
    df["chose_x"] = 1  # no variation at all
    with pytest.raises(ValueError, match="chose_x"):
        fit_clustered_logit(
            hypothesis="H_CONSTANT_Y",
            df=df,
            dependent_col="chose_x",
            regressor_col="regressor",
            cluster_col="agent_id",
            fixed_effect_cols=["agent_type"],
        )


def test_fit_clustered_logit_raises_on_constant_regressor():
    df = _synthetic_dataset(n_agents=10, decisions_per_agent=5, seed=3)
    df["regressor"] = 0.5  # no variation at all
    with pytest.raises(ValueError, match="regressor"):
        fit_clustered_logit(
            hypothesis="H_CONSTANT_X",
            df=df,
            dependent_col="chose_x",
            regressor_col="regressor",
            cluster_col="agent_id",
            fixed_effect_cols=["agent_type"],
        )


def test_fit_clustered_logit_clustered_se_differs_from_unclustered():
    """Confirms cov_type="cluster" is actually applied, not a no-op:
    within-agent-correlated outcomes (the same agent's repeated decisions
    share a single per-agent random shock) should generally produce wider
    (more conservative) clustered SEs than naive unclustered ones."""
    import numpy as np
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    rows = []
    for agent_idx in range(40):
        agent_shock = rng.normal(0, 2.0)  # a single shared shock per agent -- induces within-agent correlation
        regressor = rng.uniform(-1.0, 1.0)
        for _ in range(15):
            probability = 1.0 / (1.0 + np.exp(-(agent_shock + regressor)))
            chose_x = 1 if rng.uniform(0.0, 1.0) < probability else 0
            rows.append({"agent_id": f"agent-{agent_idx}", "chose_x": chose_x, "regressor": regressor})
    df = pd.DataFrame.from_records(rows)

    clustered = fit_clustered_logit(
        hypothesis="H_CLUSTER_CHECK",
        df=df,
        dependent_col="chose_x",
        regressor_col="regressor",
        cluster_col="agent_id",
        fixed_effect_cols=[],
    )

    # Independently fit the same model with plain (unclustered) SEs to compare.
    y = df["chose_x"].astype(float)
    x = sm.add_constant(df[["regressor"]].astype(float))
    unclustered_result = sm.Logit(y, x).fit(disp=0)
    unclustered_se = float(unclustered_result.bse["regressor"])

    assert clustered.se > unclustered_se
