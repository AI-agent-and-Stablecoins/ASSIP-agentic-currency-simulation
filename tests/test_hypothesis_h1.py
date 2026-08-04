import random
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import AgentStateRecord, Base, LLMDecisionRecord
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


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _agent_state(run_id: str, timestep: int, agent_id: str, cara_coefficient: float) -> AgentStateRecord:
    return AgentStateRecord(
        run_id=run_id,
        timestep=timestep,
        agent_id=agent_id,
        risk_profile="medium",
        cara_coefficient=cara_coefficient,
        real_purchasing_power=1000.0,
        wallet_balances={},
        utility_score=0.0,
    )


def test_regress_h1_returns_a_regression_result():
    """A genuine (noisy, not perfectly separated) planted relationship:
    higher CARA `a` -> more likely to choose a USD-zone stablecoin, with
    real per-agent variation. Direct construction, not a run_matrix-driven
    fixture (Plan 5 whole-branch review Fix I4/I7): the default exercise_
    llm_path mock always proposes the same currency for every decision,
    which can only ever produce a constant chose_usd_zone -- exactly what
    the regression engine's degenerate-dependent-variable guard now
    correctly rejects."""
    rng = random.Random(0)
    session = _session()

    master_run_id = "matrix1-master-seed0"
    rows = []
    agent_states = []
    for agent_idx in range(60):
        agent_id = f"agent-{agent_idx}"
        cara_a = rng.uniform(-2.0, 2.0)
        agent_type = "consumer" if agent_idx % 2 == 0 else "bank"
        model = "vendor/model-a" if agent_idx % 3 == 0 else "vendor/model-b"
        agent_states.append(_agent_state(master_run_id, 0, agent_id, cara_a))

        probability_usd = 1.0 / (1.0 + pow(2.71828, -cara_a))
        currency = "USDC" if rng.uniform(0.0, 1.0) < probability_usd else "EURC"
        rows.append(
            LLMDecisionRecord(
                decision_id=f"dec-{agent_idx}",
                simulation_id=master_run_id,
                timestep=1,
                agent_id=agent_id,
                agent_type=agent_type,
                requested_model=model,
                actual_model=model,
                fallback_used=False,
                fallback_reason=None,
                model_attempts=[model],
                prompt_version="v1",
                rendered_prompt_hash="hash",
                system_prompt="prompt",
                action="ACCEPT",
                currency=currency,
                chain="ethereum",
                amount=1.0,
                price=1.0,
                reported_reasoning="test",
                negotiation_id=None,
                round=1,
                risk_profile="medium",
                utility_type="cara",
                utility_parameters={},
                scenario="master_simulation",
                domestic_or_cross_border="unknown",
                governance_prompt_enabled=False,
                timestamp=datetime.now(timezone.utc),
            )
        )
    session.add_all(agent_states)
    session.add_all(rows)
    session.commit()

    result = regress_h1(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H1"
    assert result.regressor == "cara_a"
    assert result.n_obs == len(rows)
