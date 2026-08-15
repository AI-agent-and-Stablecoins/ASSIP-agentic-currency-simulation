from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.legacy.econometrics.hypothesis_datasets import build_h3_dataset
from src.legacy.econometrics.hypothesis_regressions import regress_h3
from src.legacy.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 10) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
    )
    return session


def _populated_session_with_genuine_variation(num_days: int = 8) -> Session:
    """Under the default `exercise_llm_path=True` mock, `next(iter(spec.
    currencies))` picks the SAME `liquidity_vs_governance` symbol for both
    the sandbox's domestic AND cross-border cells within one `run_matrix`
    call (both cells share the identical currency pair -- see `_build_cell_
    specs`), so `chose_higher_governance` comes out perfectly constant.
    Fed straight into `fit_clustered_logit`, that isn't merely the
    already-accepted "degenerate PerfectSeparationWarning" limitation
    (confirmed for H1): here it makes `cell_key`'s one-hot dummy exactly
    collinear with the completely-separated outcome, which drives
    statsmodels' Newton-Raphson Hessian to EXACT singularity --
    `numpy.linalg.LinAlgError: Singular matrix`, not just a warning
    (confirmed by direct probing: dropping `cell_key` from `fixed_effect_
    cols` alone, with everything else unchanged, avoids the crash and
    fits with only the expected `PerfectSeparationWarning`).
    We run `run_matrix` twice into the same session, forcing the mock (via
    the already-supported `mock_llm_decision` override, applied verbatim
    across all 13 cells each call, exactly as H2's test does) to the
    sandbox's low-governance symbol in the first call and its
    high-governance symbol in the second. Each call alone still yields a
    constant outcome, but pooled, `chose_higher_governance` genuinely
    varies both overall AND within each `cell_key` group, which resolves
    the exact singularity while remaining a "genuine choice was recorded"
    sample per this hypothesis's design (mismatched proposals in the other
    11 sandbox cells / the master cell just fall back to a harmless
    synthetic WALK_AWAY, per matrix_runner's per-cell mock-currency
    docstring, so they don't pollute this dataset).
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_governance"]
    for call_index, symbol in enumerate((option_a.symbol, option_b.symbol)):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=num_days,
            dry_run=True,
            exercise_llm_path=True,
            matrix_run_id=f"h3-variation-{call_index}",
            mock_llm_decision={
                "action": "ACCEPT",
                "proposed_currency": symbol,
                "proposed_chain": "ethereum",
                "amount": 1.0,
                "price": 1.0,
                "reasoning": (
                    "exercise_llm_path canned response, forced to alternate between "
                    "liquidity_vs_governance's two symbols across two run_matrix calls so "
                    "chose_higher_governance genuinely varies within each cell_key group "
                    "instead of being perfectly collinear with it"
                ),
            },
            session=session,
        )
    return session


def test_build_h3_dataset_only_includes_the_liquidity_vs_governance_sandbox():
    session = _populated_session()
    df = build_h3_dataset(session)

    assert not df.empty
    assert set(df.columns) >= {
        "agent_id", "chose_higher_governance", "cara_a", "agent_type", "actual_model", "cell_key",
    }
    assert set(df["cell_key"].unique()) <= {
        "liquidity_vs_governance_domestic", "liquidity_vs_governance_cross_border",
    }
    assert df["chose_higher_governance"].isin([0, 1]).all()


def test_regress_h3_returns_a_regression_result():
    session = _populated_session_with_genuine_variation()
    result = regress_h3(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H3"
    assert result.regressor == "cara_a"
