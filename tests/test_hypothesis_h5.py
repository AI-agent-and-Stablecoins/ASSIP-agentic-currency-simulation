import random
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import AgentRecord, Base, LLMDecisionRecord, TimestepLogRecord
from src.econometrics.hypothesis_datasets import build_h5_dataset
from src.econometrics.hypothesis_regressions import regress_h5
from src.econometrics.regression_engine import RegressionResult

# Driving this hypothesis's test through a real run_matrix(...) call hits the
# same wall H1-H4 already hit (see tests/test_hypothesis_h4.py's header):
# the canned mock decision is constant across days/cells, so a run_matrix-
# driven test would be degenerate at best and cost real wall-clock minutes
# at worst. build_h5_dataset/regress_h5 only read persisted LLMDecisionRecord
# /TimestepLogRecord/AgentRecord rows -- no dependency on live LLM/
# Environment state -- so this test constructs those rows directly,
# exercising the exact same filtering/rolling-volatility/regression logic
# at a fraction of the cost.
#
# Cross-zone filter (Plan 5 whole-branch review Fix I5): build_h5_dataset
# now determines cross-zone eligibility from a negotiation's two
# participants' currency_zones (grouped by LLMDecisionRecord.negotiation_id),
# not just the deciding agent's own zone -- so this fixture explicitly
# constructs buyer/seller PAIRS sharing one negotiation_id, with differing
# or matching zones, to exercise both the cross-zone-included and
# same-zone-excluded paths.

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
    negotiation_id: str | None,
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
        negotiation_id=negotiation_id,
        round=1,
        risk_profile="medium",
        utility_type="cara",
        utility_parameters={},
        scenario="master",
        domestic_or_cross_border="unknown",
        governance_prompt_enabled=False,
        timestamp=datetime.now(timezone.utc),
    )


def test_build_h5_dataset_only_includes_cross_zone_negotiations():
    session = _session()
    session.add_all(
        [
            _agent("usd-buyer-1", "USD"),
            _agent("eur-seller-1", "EUR"),
            _agent("usd-buyer-2", "USD"),
            _agent("usd-seller-2", "USD"),  # same-zone pair -- must be excluded
            _agent("agent-legacy", None),
        ]
    )
    rates = [1.05, 1.06, 1.04, 1.08, 1.03, 1.09, 1.02, 1.07, 1.05]
    session.add_all([_rate(_MASTER_RUN_ID, day, rate) for day, rate in enumerate(rates)])
    session.add_all(
        [
            # Cross-zone negotiation: USD-zone agent <-> EUR-zone agent,
            # sharing negotiation_id "neg-1" -- both decisions are eligible.
            _decision(_MASTER_RUN_ID, 5, "usd-buyer-1", "USDC", negotiation_id="neg-1"),
            _decision(_MASTER_RUN_ID, 5, "eur-seller-1", "USDC", negotiation_id="neg-1"),
            # Same-zone negotiation (both USD) -- must be excluded even though
            # both agents have a real zone and the currency is USD-zone.
            _decision(_MASTER_RUN_ID, 6, "usd-buyer-2", "USDC", negotiation_id="neg-2"),
            _decision(_MASTER_RUN_ID, 6, "usd-seller-2", "USDC", negotiation_id="neg-2"),
            # No negotiation_id at all -- excluded (can't determine cross-zone status).
            _decision(_MASTER_RUN_ID, 7, "agent-legacy", "USDC", negotiation_id=None),
            # Non-master cell -- must be excluded regardless of zone.
            _decision(_NON_MASTER_RUN_ID, 6, "usd-buyer-2", "USDC", negotiation_id="neg-3"),
        ]
    )
    session.commit()

    df = build_h5_dataset(session)

    assert set(df.columns) >= {
        "agent_id", "chose_usd_zone", "eur_usd_volatility", "agent_type", "actual_model",
    }
    # Only the cross-zone negotiation's 2 decisions survive.
    assert set(df["agent_id"]) == {"usd-buyer-1", "eur-seller-1"}
    assert df["chose_usd_zone"].isin([0, 1]).all()
    assert (df["chose_usd_zone"] == 1).all()  # both chose a USD-zone currency (USDC) here
    assert (df["eur_usd_volatility"] > 0).all()


def test_regress_h5_returns_a_regression_result():
    """A genuine (noisy, not perfectly separated) planted relationship:
    days later in the run have a wider-swinging EUR/USD rate (higher
    realized volatility) AND a higher probability of a USD-zone choice --
    pooled across many genuinely cross-zone negotiation pairs with real
    variation, so `fit_clustered_logit` has real signal without hitting
    the exact-singularity failure mode documented for H3."""
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

    agent_records = []
    rows = []
    for pair_idx in range(40):
        buyer_zone = "USD" if pair_idx % 2 == 0 else "EUR"
        seller_zone = "EUR" if buyer_zone == "USD" else "USD"  # always cross-zone
        buyer_id = f"buyer-{pair_idx}"
        seller_id = f"seller-{pair_idx}"
        agent_records.append(_agent(buyer_id, buyer_zone))
        agent_records.append(_agent(seller_id, seller_zone))
        agent_type = "consumer" if pair_idx % 2 == 0 else "bank"
        model = "vendor/model-a" if pair_idx % 3 == 0 else "vendor/model-b"
        timestep = rng.randint(5, num_days - 1)
        probability_usd = timestep / num_days  # later day -> more likely USD-zone
        currency = (
            rng.choice(_USD_SYMBOLS) if rng.uniform(0.0, 1.0) < probability_usd else rng.choice(_EUR_SYMBOLS)
        )
        neg_id = f"neg-{pair_idx}"
        rows.append(_decision(_MASTER_RUN_ID, timestep, buyer_id, currency, neg_id, agent_type, model))
        rows.append(_decision(_MASTER_RUN_ID, timestep, seller_id, currency, neg_id, agent_type, model))
    session.add_all(agent_records)
    session.add_all(rows)
    session.commit()

    result = regress_h5(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H5"
    assert result.regressor == "eur_usd_volatility"
    assert result.n_obs == len(rows)
