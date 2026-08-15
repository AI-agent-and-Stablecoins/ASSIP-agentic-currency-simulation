from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import AgentStateRecord, Base, LLMDecisionRecord
from src.legacy.econometrics.hypothesis_datasets import build_h2_dataset
from src.legacy.econometrics.hypothesis_regressions import regress_h2
from src.legacy.econometrics.regression_engine import RegressionResult
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


# --- Direct-row-construction test covering all 4 tie-break-robust
# classification branches (Plan 5 whole-branch review Fix I3/I7) ---------
#
# The forced-mock test above only ever produces the "chose_spread_optimal
# == 1" branch (per its own docstring), leaving the other 3 branches --
# chose_spread_optimal == 0, no-genuine-tradeoff exclusion, and
# ambiguous-choice exclusion -- with zero row-level coverage. These
# construct LLMDecisionRecord/AgentStateRecord rows directly (seconds,
# not minutes), following the pattern established in
# tests/test_hypothesis_h4.py and tests/test_hypothesis_h5.py.

_MASTER_RUN_ID = "matrix1-master-seed0"


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


def _decision(
    timestep: int,
    agent_id: str,
    currency: str,
    chain: str,
    spread_optimal_currency: str,
    spread_optimal_chain: str,
    gas_optimal_currency: str,
    gas_optimal_chain: str,
) -> LLMDecisionRecord:
    return LLMDecisionRecord(
        decision_id=f"dec-{timestep}-{agent_id}",
        simulation_id=_MASTER_RUN_ID,
        timestep=timestep,
        agent_id=agent_id,
        agent_type="consumer",
        requested_model="vendor/model-a",
        actual_model="vendor/model-a",
        fallback_used=False,
        fallback_reason=None,
        model_attempts=["vendor/model-a"],
        prompt_version="v1",
        rendered_prompt_hash="hash",
        system_prompt="prompt",
        action="ACCEPT",
        currency=currency,
        chain=chain,
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
        spread_optimal_currency=spread_optimal_currency,
        spread_optimal_chain=spread_optimal_chain,
        gas_optimal_currency=gas_optimal_currency,
        gas_optimal_chain=gas_optimal_chain,
        timestamp=datetime.now(timezone.utc),
    )


def test_build_h2_dataset_classifies_spread_optimal_choice_correctly():
    session = _session()
    session.add_all(
        [
            # Branch A: chose the spread-optimal CURRENCY, on a DIFFERENT
            # chain than gas-optimal -- chose_spread_optimal=1.
            _agent_state(_MASTER_RUN_ID, 0, "agent-spread", 0.5),
            _decision(1, "agent-spread", "USDT", "ethereum", "USDT", "arbitrum", "USDC", "solana"),
        ]
    )
    session.commit()

    df = build_h2_dataset(session)
    row = df[df["agent_id"] == "agent-spread"].iloc[0]
    assert row["chose_spread_optimal"] == 1


def test_build_h2_dataset_classifies_gas_optimal_choice_correctly():
    session = _session()
    session.add_all(
        [
            # Branch B: chose the gas-optimal CHAIN, with a DIFFERENT
            # currency than spread-optimal -- chose_spread_optimal=0.
            _agent_state(_MASTER_RUN_ID, 0, "agent-gas", 0.5),
            _decision(1, "agent-gas", "USDC", "solana", "USDT", "arbitrum", "USDC", "solana"),
        ]
    )
    session.commit()

    df = build_h2_dataset(session)
    row = df[df["agent_id"] == "agent-gas"].iloc[0]
    assert row["chose_spread_optimal"] == 0


def test_build_h2_dataset_excludes_decisions_with_no_genuine_tradeoff():
    session = _session()
    session.add_all(
        [
            # spread-optimal and gas-optimal are the SAME candidate this
            # round -- no genuine tradeoff existed, must be excluded.
            _agent_state(_MASTER_RUN_ID, 0, "agent-no-tradeoff", 0.5),
            _decision(1, "agent-no-tradeoff", "USDC", "solana", "USDC", "solana", "USDC", "solana"),
        ]
    )
    session.commit()

    df = build_h2_dataset(session)
    assert "agent-no-tradeoff" not in set(df["agent_id"])


def test_build_h2_dataset_excludes_ambiguous_choices():
    session = _session()
    session.add_all(
        [
            # Chose BOTH the spread-optimal currency AND the gas-optimal
            # chain simultaneously (possible under a gas tie across
            # currencies) -- doesn't reveal a preference between the two,
            # must be excluded.
            _agent_state(_MASTER_RUN_ID, 0, "agent-both", 0.5),
            _decision(1, "agent-both", "USDT", "solana", "USDT", "arbitrum", "USDC", "solana"),
            # Chose NEITHER the spread-optimal currency nor the gas-optimal
            # chain -- also ambiguous, must be excluded.
            _agent_state(_MASTER_RUN_ID, 1, "agent-neither", 0.5),
            _decision(2, "agent-neither", "DAI", "base", "USDT", "arbitrum", "USDC", "solana"),
        ]
    )
    session.commit()

    df = build_h2_dataset(session)
    assert "agent-both" not in set(df["agent_id"])
    assert "agent-neither" not in set(df["agent_id"])


def test_regress_h2_returns_a_regression_result():
    """A genuine (noisy, not perfectly separated) planted relationship:
    higher CARA `a` -> more likely to choose the spread-optimal option
    over the gas-optimal one, with real per-agent variation -- avoiding
    the degenerate/constant-outcome sample a single fixed `mock_llm_
    decision` would produce (a `run_matrix`-driven fixture can only ever
    return ONE canned decision for the whole run, which can never create
    genuine chose_spread_optimal variation -- exactly what Fix I4's new
    degenerate-dependent-variable guard now correctly rejects)."""
    import random

    rng = random.Random(0)
    session = _session()

    rows = []
    agent_states = []
    for agent_idx in range(60):
        agent_id = f"agent-{agent_idx}"
        cara_a = rng.uniform(-2.0, 2.0)
        agent_type = "consumer" if agent_idx % 2 == 0 else "bank"
        model = "vendor/model-a" if agent_idx % 3 == 0 else "vendor/model-b"
        agent_states.append(_agent_state(_MASTER_RUN_ID, 0, agent_id, cara_a))

        probability_spread = 1.0 / (1.0 + pow(2.71828, -cara_a))
        if rng.uniform(0.0, 1.0) < probability_spread:
            currency, chain = "USDT", "arbitrum"  # matches spread_optimal -- chose_spread_optimal=1
        else:
            currency, chain = "USDC", "solana"  # matches gas_optimal -- chose_spread_optimal=0
        rows.append(
            LLMDecisionRecord(
                decision_id=f"dec-{agent_idx}",
                simulation_id=_MASTER_RUN_ID,
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
                chain=chain,
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
                spread_optimal_currency="USDT",
                spread_optimal_chain="arbitrum",
                gas_optimal_currency="USDC",
                gas_optimal_chain="solana",
                timestamp=datetime.now(timezone.utc),
            )
        )
    session.add_all(agent_states)
    session.add_all(rows)
    session.commit()

    result = regress_h2(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H2"
    assert result.regressor == "cara_a"
    assert result.n_obs == len(rows)
