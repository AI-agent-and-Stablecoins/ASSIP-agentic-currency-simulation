import random
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import AgentRecord, Base, LLMDecisionRecord, TimestepLogRecord
from src.econometrics.hypothesis_datasets import build_h5_dataset
from src.econometrics.hypothesis_regressions import regress_h5
from src.econometrics.regression_engine import RegressionResult

# Driving this hypothesis's test through a real run_matrix(...) call hits the
# same wall H1-H4 already hit (see docs/superpowers/plans/
# 2026-08-02-phase3-plan5-Task-8-brief.md and tests/test_hypothesis_h4.py's
# header): the canned mock decision is constant across days/cells, so a
# run_matrix-driven test would be degenerate at best and cost real wall-clock
# minutes at worst. build_h5_dataset/regress_h5 only read persisted
# LLMDecisionRecord/TimestepLogRecord/AgentRecord rows -- no dependency on
# live LLM/Environment state -- so this test constructs those rows directly,
# exercising the exact same filtering/rolling-volatility/regression logic at
# a fraction of the cost.

_MASTER_RUN_ID = "matrix1-master-seed0"
_NON_MASTER_RUN_ID = "matrix1-liquidity_vs_governance_domestic-seed0"

_USD_SYMBOLS = ("USDC", "USDT", "DAI")
_EUR_SYMBOLS = ("EURC", "EURT")


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _agent(agent_id: str, currency_zone: str | None) -> AgentRecord:
    return AgentRecord(
        id=agent_id,
        agent_class="consumer",
        profile_name="test",
        risk_profile="medium",
        currency_zone=currency_zone,
        created_at=datetime.now(timezone.utc),
    )


def _rate(run_id: str, timestep: int, rate: float) -> TimestepLogRecord:
    return TimestepLogRecord(
        run_id=run_id,
        timestep=timestep,
        inflation_rate=0.02,
        confidence_index=0.5,
        eth_gas_fee_gwei=10.0,
        solana_gas_fee_usd=0.01,
        eur_usd_exchange_rate=rate,
    )


def _decision(
    run_id: str,
    timestep: int,
    agent_id: str,
    currency: str,
    agent_type: str = "consumer",
    actual_model: str = "vendor/model-a",
) -> LLMDecisionRecord:
    return LLMDecisionRecord(
        decision_id=f"dec-{run_id}-{timestep}-{agent_id}",
        simulation_id=run_id,
        timestep=timestep,
        agent_id=agent_id,
        agent_type=agent_type,
        requested_model=actual_model,
        actual_model=actual_model,
        fallback_used=False,
        fallback_reason=None,
        model_attempts=[actual_model],
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
        scenario="master",
        domestic_or_cross_border="unknown",
        governance_prompt_enabled=False,
        timestamp=datetime.now(timezone.utc),
    )


def test_build_h5_dataset_excludes_legacy_agents_and_zone_neutral_currencies():
    session = _session()
    session.add_all(
        [
            _agent("agent-usd-1", "USD"),
            _agent("agent-usd-2", "USD"),
            _agent("agent-eur-1", "EUR"),
            _agent("agent-eur-2", "EUR"),
            _agent("agent-legacy", None),
        ]
    )
    rates = [1.05, 1.06, 1.04, 1.08, 1.03, 1.09, 1.02, 1.07, 1.05]
    session.add_all([_rate(_MASTER_RUN_ID, day, rate) for day, rate in enumerate(rates)])
    session.add_all(
        [
            _decision(_MASTER_RUN_ID, 1, "agent-usd-1", "USDC"),
            _decision(_MASTER_RUN_ID, 2, "agent-eur-1", "EURC"),
            _decision(_MASTER_RUN_ID, 3, "agent-usd-2", "USDT"),
            _decision(_MASTER_RUN_ID, 4, "agent-eur-2", "EURT"),
            # Legacy count-based agent (no currency_zone) -- must be excluded.
            _decision(_MASTER_RUN_ID, 5, "agent-legacy", "USDC"),
            # Gold-backed (zone-neutral) currency -- must be excluded.
            _decision(_MASTER_RUN_ID, 6, "agent-usd-1", "PAXG"),
            # Non-master cell -- must be excluded regardless of zone.
            _decision(_NON_MASTER_RUN_ID, 4, "agent-usd-2", "USDC"),
        ]
    )
    session.commit()

    df = build_h5_dataset(session)

    assert set(df.columns) >= {
        "agent_id", "chose_usd_zone", "eur_usd_volatility", "agent_type", "actual_model",
    }
    assert df["chose_usd_zone"].isin([0, 1]).all()
    assert "agent-legacy" not in set(df["agent_id"])
    assert len(df) == 4
    assert set(df["agent_id"]) == {"agent-usd-1", "agent-usd-2", "agent-eur-1", "agent-eur-2"}

    usd1_row = df[df["agent_id"] == "agent-usd-1"].iloc[0]
    assert usd1_row["chose_usd_zone"] == 1
    eur1_row = df[df["agent_id"] == "agent-eur-1"].iloc[0]
    assert eur1_row["chose_usd_zone"] == 0
    # eur_usd_volatility is a real (nonzero) rolling std, not a placeholder.
    assert (df["eur_usd_volatility"] > 0).all()


def test_regress_h5_returns_a_regression_result():
    """A genuine (noisy, not perfectly separated) planted relationship:
    days later in the run have a wider-swinging EUR/USD rate (higher
    realized volatility) AND a higher probability of a USD-zone choice --
    pooled across many agents/models/types with real variation, so
    `fit_clustered_logit` has real signal without hitting the exact-
    singularity failure mode documented for H3."""
    rng = random.Random(0)
    session = _session()

    num_days = 60
    rates = []
    rate = 1.05
    for day in range(num_days):
        # Amplitude of the day-to-day swing grows with day -> later days have
        # a genuinely higher trailing rolling std than earlier ones.
        amplitude = 0.001 + 0.004 * (day / num_days)
        rate += rng.uniform(-amplitude, amplitude)
        rates.append(rate)
    session.add_all([_rate(_MASTER_RUN_ID, day, r) for day, r in enumerate(rates)])

    agent_zones = []
    rows = []
    for agent_idx in range(40):
        zone = "USD" if agent_idx % 2 == 0 else "EUR"
        agent_id = f"agent-{agent_idx}"
        agent_zones.append(_agent(agent_id, zone))
        agent_type = "consumer" if agent_idx % 2 == 0 else "bank"
        model = "vendor/model-a" if agent_idx % 3 == 0 else "vendor/model-b"
        timestep = rng.randint(5, num_days - 1)
        probability_usd = timestep / num_days  # later day -> more likely USD-zone
        currency = (
            rng.choice(_USD_SYMBOLS) if rng.uniform(0.0, 1.0) < probability_usd else rng.choice(_EUR_SYMBOLS)
        )
        rows.append(_decision(_MASTER_RUN_ID, timestep, agent_id, currency, agent_type, model))
    session.add_all(agent_zones)
    session.add_all(rows)
    session.commit()

    result = regress_h5(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H5"
    assert result.regressor == "eur_usd_volatility"
    assert result.n_obs == len(rows)
