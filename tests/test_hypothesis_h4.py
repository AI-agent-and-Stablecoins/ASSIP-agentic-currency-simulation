from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, InterventionLogRecord, LLMDecisionRecord
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.econometrics.hypothesis_datasets import build_h4_dataset
from src.econometrics.hypothesis_regressions import regress_h4
from src.econometrics.regression_engine import RegressionResult

# Driving this hypothesis's test through a real run_matrix(...) call (as
# H1-H3's tests do) is prohibitively slow here: H4 needs num_days > 120 for
# the crisis_warning/depeg_event pair to fire at all, and pooling 4 cells
# with genuine chose_gold variation (per H3's precedent) needs several such
# calls -- each simulating all 13 matrix cells, not just H4's 4 -- measured
# at tens of minutes per call. build_h4_dataset/regress_h4 have no
# dependency on live LLM/Environment state (they only read persisted
# LLMDecisionRecord/InterventionLogRecord rows), so this test constructs
# those rows directly instead, exercising the exact same filtering/
# proximity/regression logic at a fraction of the cost.


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


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
        scenario="asset_backing_vs_liquidity_sandbox",
        domestic_or_cross_border="unknown",
        governance_prompt_enabled=False,
        timestamp=datetime.now(timezone.utc),
    )


def _shock(run_id: str, timestep: int, shock_type: str) -> InterventionLogRecord:
    return InterventionLogRecord(
        run_id=run_id,
        timestep=timestep,
        shock_type=shock_type,
        target_currency=None,
        target_issuer=None,
        magnitude=0.1,
    )


_LIQUIDITY_GOLD, _LIQUIDITY_STABLE = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_liquidity"]
_STABILITY_GOLD, _STABILITY_STABLE = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_stability"]


def test_build_h4_dataset_only_includes_gold_backed_sandboxes():
    session = _session()

    # Eligible: asset_backing_vs_liquidity_domestic, with a crisis/depeg
    # pair at day 110/120 -- one decision approaching (day 105, gold) and
    # one past (day 130, non-gold).
    run_id = "matrix1-asset_backing_vs_liquidity_domestic-seed0"
    session.add_all(
        [
            _shock(run_id, 110, "crisis_warning"),
            _shock(run_id, 120, "depeg_event"),
            _decision(run_id, 105, "agent-1", _LIQUIDITY_GOLD.symbol),
            _decision(run_id, 130, "agent-2", _LIQUIDITY_STABLE.symbol),
        ]
    )
    # Eligible: asset_backing_vs_stability_domestic, exercising the OTHER
    # sandbox pair's gold detection (both pairs share one gold_symbols
    # comprehension in build_h4_dataset -- this confirms it isn't
    # accidentally scoped to just the liquidity pair).
    stability_run_id = "matrix1-asset_backing_vs_stability_domestic-seed0"
    session.add_all(
        [
            _shock(stability_run_id, 110, "crisis_warning"),
            _shock(stability_run_id, 120, "depeg_event"),
            _decision(stability_run_id, 108, "agent-5", _STABILITY_GOLD.symbol),
            _decision(stability_run_id, 122, "agent-6", _STABILITY_STABLE.symbol),
        ]
    )
    # Ineligible: master cell (not one of H4's 4 sandboxes) -- must be excluded.
    master_run_id = "matrix1-master-seed0"
    session.add_all(
        [
            _shock(master_run_id, 110, "crisis_warning"),
            _decision(master_run_id, 105, "agent-3", "USDC"),
        ]
    )
    # Ineligible: an asset_backing_vs_stability_cross_border run with NO
    # crisis/depeg event at all -- must be excluded (no proximity to measure).
    no_event_run_id = "matrix1-asset_backing_vs_stability_cross_border-seed1"
    session.add(_decision(no_event_run_id, 50, "agent-4", _STABILITY_GOLD.symbol))
    session.commit()

    df = build_h4_dataset(session)

    assert set(df.columns) >= {
        "agent_id", "chose_gold", "proximity_days", "agent_type", "actual_model", "cell_key",
    }
    # Only the 4 eligible decisions from the two eligible runs survive.
    assert len(df) == 4
    assert set(df["cell_key"]) == {"asset_backing_vs_liquidity_domestic", "asset_backing_vs_stability_domestic"}
    assert set(df["agent_id"]) == {"agent-1", "agent-2", "agent-5", "agent-6"}

    approaching = df[df["agent_id"] == "agent-1"].iloc[0]
    assert approaching["chose_gold"] == 1
    assert approaching["proximity_days"] == 105 - 110  # signed, negative = approaching

    past = df[df["agent_id"] == "agent-2"].iloc[0]
    assert past["chose_gold"] == 0
    assert past["proximity_days"] == 130 - 120  # signed, positive = past (nearest event is the depeg at 120)

    stability_approaching = df[df["agent_id"] == "agent-5"].iloc[0]
    assert stability_approaching["chose_gold"] == 1  # stability pair's gold option correctly detected
    assert stability_approaching["proximity_days"] == 108 - 110

    stability_past = df[df["agent_id"] == "agent-6"].iloc[0]
    assert stability_past["chose_gold"] == 0  # stability pair's non-gold option correctly detected
    assert stability_past["proximity_days"] == 122 - 120


def test_regress_h4_returns_a_regression_result():
    """A genuine (noisy, not perfectly separated) planted relationship:
    closer proximity to the nearest event -> more likely gold, pooled
    across all 4 of H4's cells with real agent/model/cell_key variation,
    so `fit_clustered_logit` has real signal without hitting the
    exact-singularity failure mode documented for H3."""
    import random

    rng = random.Random(0)
    session = _session()

    cells = [
        ("asset_backing_vs_liquidity_domestic", _LIQUIDITY_GOLD.symbol, _LIQUIDITY_STABLE.symbol),
        ("asset_backing_vs_liquidity_cross_border", _LIQUIDITY_GOLD.symbol, _LIQUIDITY_STABLE.symbol),
        ("asset_backing_vs_stability_domestic", _STABILITY_GOLD.symbol, _STABILITY_STABLE.symbol),
        ("asset_backing_vs_stability_cross_border", _STABILITY_GOLD.symbol, _STABILITY_STABLE.symbol),
    ]

    rows = []
    for cell_key, gold_symbol, stable_symbol in cells:
        run_id = f"matrix1-{cell_key}-seed0"
        session.add(_shock(run_id, 110, "crisis_warning"))
        session.add(_shock(run_id, 120, "depeg_event"))
        for agent_idx in range(15):
            agent_id = f"{cell_key}-agent-{agent_idx}"
            agent_type = "consumer" if agent_idx % 2 == 0 else "bank"
            model = "vendor/model-a" if agent_idx % 3 == 0 else "vendor/model-b"
            timestep = rng.choice([90, 100, 108, 112, 118, 125, 140, 160])
            proximity = timestep - (110 if abs(timestep - 110) <= abs(timestep - 120) else 120)
            probability_gold = 1.0 / (1.0 + pow(2.71828, 0.15 * abs(proximity)))
            currency = gold_symbol if rng.uniform(0.0, 1.0) < probability_gold else stable_symbol
            rows.append(_decision(run_id, timestep, agent_id, currency, agent_type, model))

    session.add_all(rows)
    session.commit()

    result = regress_h4(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H4"
    assert result.regressor == "proximity_days"
    assert result.n_obs == len(rows)
