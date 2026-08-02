from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.econometrics.hypothesis_datasets import build_h2_dataset
from src.econometrics.hypothesis_regressions import regress_h2
from src.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 10, mock_llm_decision: dict | None = None) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        mock_llm_decision=mock_llm_decision,
        session=session,
    )
    return session


def test_build_h2_dataset_only_includes_genuine_tradeoff_decisions():
    session = _populated_session()
    df = build_h2_dataset(session)

    assert set(df.columns) >= {"agent_id", "chose_spread_optimal", "cara_a", "agent_type", "actual_model"}
    assert df["chose_spread_optimal"].isin([0, 1]).all()


def test_regress_h2_returns_a_regression_result():
    # Under exercise_llm_path=True the default canned mock always proposes
    # currency="USDC", chain="ethereum" for every decision, every day. The
    # master cell's per-round spread-optimal/gas-optimal candidates are
    # deterministically (USDC, arbitrum)/(USDC, solana) -- "ethereum" is
    # never one of them, so the agent's chosen (currency, chain) can never
    # match either optimal candidate, no matter how many days are run.
    # build_h2_dataset's genuine-tradeoff filter therefore discards every
    # decision, unconditionally (confirmed by direct probing across many
    # num_days values -- this is deterministic, not a low-sample-size
    # issue). We force the mock to match the spread-optimal candidate
    # exactly so a genuine (if degenerate/constant-outcome) tradeoff
    # sample exists -- the same category of dry-run-mock limitation
    # already accepted for H1's degenerate chose_usd_zone fixture.
    session = _populated_session(
        num_days=15,
        mock_llm_decision={
            "action": "ACCEPT",
            "proposed_currency": "USDC",
            "proposed_chain": "arbitrum",
            "amount": 1.0,
            "price": 1.0,
            "reasoning": "exercise_llm_path canned response, forced to match H2's spread-optimal candidate so a genuine (if degenerate) tradeoff sample exists in this fast offline test",
        },
    )
    result = regress_h2(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H2"
    assert result.regressor == "cara_a"
