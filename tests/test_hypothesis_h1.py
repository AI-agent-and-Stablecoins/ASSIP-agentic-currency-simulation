import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.econometrics.hypothesis_datasets import build_h1_dataset
from src.econometrics.hypothesis_regressions import regress_h1
from src.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 5, seeds: list[int] | None = None) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=seeds or [0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
        keep_daily_results=False,
    )
    return session


def test_build_h1_dataset_only_includes_master_cell_decisions():
    session = _populated_session()
    df = build_h1_dataset(session)

    assert not df.empty
    assert set(df.columns) >= {"agent_id", "chose_usd_zone", "cara_a", "agent_type", "actual_model"}
    assert df["chose_usd_zone"].isin([0, 1]).all()


def test_build_h1_dataset_excludes_gold_backed_decisions():
    session = _populated_session()
    df = build_h1_dataset(session)
    # PAXG/XAUT (gold-backed, peg=None per CurrencyConfig -- see
    # src.economy.fx_tax.currency_zone_of) must never appear as a
    # dependent-variable observation -- H1 is a USD-vs-EUR contrast only.
    assert df["chose_usd_zone"].isin([0, 1]).all()


def test_regress_h1_returns_a_regression_result():
    session = _populated_session(num_days=10)
    result = regress_h1(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H1"
    assert result.regressor == "cara_a"
    assert result.n_obs > 0
