"""Per-hypothesis dataset builders: each function returns a `pandas.
DataFrame`, one row per eligible `LLMDecisionRecord`, ready for `src.
econometrics.regression_engine.fit_clustered_logit`. See
docs/superpowers/specs/2026-08-02-phase3-plan5-econometrics-design.md
Sec 1 for the exact per-hypothesis data-source/dependent-variable/
regressor design this implements.
"""

import pandas as pd
from sqlalchemy.orm import Session

from database.models import AgentStateRecord, InterventionLogRecord, LLMDecisionRecord
from src.currencies.currency import load_currency_universe
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.econometrics.cell_identity import cell_key_from_run_id
from src.economy.fx_tax import currency_zone_of

_DECIDED_ACTIONS = ("ACCEPT", "OFFER")


def _join_cara_a(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Joins each row's agent's CARA `a` AT THAT DECISION'S timestep from
    `AgentStateRecord` (matched on run_id/timestep/agent_id) -- the correct
    source per the design spec Sec 1 (NOT `LLMDecisionRecord.utility_
    parameters`, which omits risk-neutral agents' `a=0.0` entirely).
    `df` must already have `run_id`/`timestep`/`agent_id` columns. Rows
    with no matching `AgentStateRecord` (shouldn't happen in practice --
    every persisted day writes one per agent -- but defensively dropped
    rather than silently coerced) are excluded.
    """
    if df.empty:
        return df.assign(cara_a=pd.Series(dtype=float))

    run_ids = df["run_id"].unique().tolist()
    states = (
        session.query(
            AgentStateRecord.run_id,
            AgentStateRecord.timestep,
            AgentStateRecord.agent_id,
            AgentStateRecord.cara_coefficient,
        )
        .filter(AgentStateRecord.run_id.in_(run_ids))
        .all()
    )
    states_df = pd.DataFrame(states, columns=["run_id", "timestep", "agent_id", "cara_a"])
    merged = df.merge(states_df, on=["run_id", "timestep", "agent_id"], how="left")
    return merged.dropna(subset=["cara_a"]).reset_index(drop=True)


def build_h1_dataset(session: Session) -> pd.DataFrame:
    """H1: higher CARA `a` -> stronger preference for USD-zone stablecoins
    over EUR-zone. Master cell only (the only cell with real currency-zone
    variation). Gold-backed/zone-neutral decisions (currency_zone_of
    returns None) are excluded -- H1 is a USD-vs-EUR contrast only."""
    currencies = load_currency_universe()

    decisions = (
        session.query(LLMDecisionRecord)
        .filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
        .all()
    )

    records = []
    for decision in decisions:
        if cell_key_from_run_id(decision.simulation_id) != "master":
            continue
        currency = currencies.get(decision.currency)
        if currency is None:
            continue
        zone = currency_zone_of(currency)
        if zone is None:
            continue
        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_usd_zone": 1 if zone == "USD" else 0,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
            }
        )

    df = pd.DataFrame.from_records(
        records, columns=["run_id", "timestep", "agent_id", "chose_usd_zone", "agent_type", "actual_model"]
    )
    return _join_cara_a(session, df)


def build_h2_dataset(session: Session) -> pd.DataFrame:
    """H2: higher CARA `a` -> prioritizes low spread (liquidity_score, the
    codebase's spread proxy) over low gas fees. Master cell only. Keeps
    only decisions where the round's spread-optimal and gas-optimal
    candidates DIFFERED (a genuine tradeoff existed) AND the agent's
    actual choice matches one of those two candidates -- per the design
    spec Sec 2's resolved tradeoff-sample design.
    """
    decisions = (
        session.query(LLMDecisionRecord)
        .filter(
            LLMDecisionRecord.action.in_(_DECIDED_ACTIONS),
            LLMDecisionRecord.spread_optimal_currency.isnot(None),
            LLMDecisionRecord.spread_optimal_currency != "",
        )
        .all()
    )

    records = []
    for decision in decisions:
        if cell_key_from_run_id(decision.simulation_id) != "master":
            continue
        spread_optimal = (decision.spread_optimal_currency, decision.spread_optimal_chain)
        gas_optimal = (decision.gas_optimal_currency, decision.gas_optimal_chain)
        if spread_optimal == gas_optimal:
            continue  # no genuine tradeoff this round
        chosen = (decision.currency, decision.chain)
        if chosen not in (spread_optimal, gas_optimal):
            continue  # chose neither optimal option -- ambiguous, excluded
        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_spread_optimal": 1 if chosen == spread_optimal else 0,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
            }
        )

    df = pd.DataFrame.from_records(
        records, columns=["run_id", "timestep", "agent_id", "chose_spread_optimal", "agent_type", "actual_model"]
    )
    return _join_cara_a(session, df)


_H3_SANDBOX_KEY = "liquidity_vs_governance"
_H3_CELLS = {f"{_H3_SANDBOX_KEY}_domestic", f"{_H3_SANDBOX_KEY}_cross_border"}


def build_h3_dataset(session: Session) -> pd.DataFrame:
    """H3: higher CARA `a` -> prioritizes GENIUS Act compliance/governance
    over liquidity. The `liquidity_vs_governance` sandbox (domestic +
    cross-border pooled, with `cell_key` as a fixed effect distinguishing
    the two -- see design spec Sec 1)."""
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[_H3_SANDBOX_KEY]
    higher_governance_symbol = (
        option_a.symbol if option_a.governance_score >= option_b.governance_score else option_b.symbol
    )

    decisions = (
        session.query(LLMDecisionRecord)
        .filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
        .all()
    )

    records = []
    for decision in decisions:
        cell_key = cell_key_from_run_id(decision.simulation_id)
        if cell_key not in _H3_CELLS:
            continue
        if decision.currency not in (option_a.symbol, option_b.symbol):
            continue
        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_higher_governance": 1 if decision.currency == higher_governance_symbol else 0,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
                "cell_key": cell_key,
            }
        )

    df = pd.DataFrame.from_records(
        records,
        columns=["run_id", "timestep", "agent_id", "chose_higher_governance", "agent_type", "actual_model", "cell_key"],
    )
    return _join_cara_a(session, df)


_H4_SANDBOX_KEYS = ("asset_backing_vs_liquidity", "asset_backing_vs_stability")
_H4_CELLS = {f"{key}_{suffix}" for key in _H4_SANDBOX_KEYS for suffix in ("domestic", "cross_border")}
_H4_PROXIMITY_SHOCK_TYPES = ("crisis_warning", "depeg_event")


def _signed_proximity(timestep: int, event_days: list[int]) -> int:
    """Signed distance (in days) from `timestep` to the nearest crisis_
    warning/depeg_event day for this run: negative = approaching (before
    the event), positive = past (after it)."""
    nearest = min(event_days, key=lambda day: abs(day - timestep))
    return timestep - nearest


def build_h4_dataset(session: Session) -> pd.DataFrame:
    """H4: closer crisis/depeg proximity -> stronger shift to gold-backed
    tokens. The two sandboxes with a gold option (asset_backing_vs_
    liquidity, asset_backing_vs_stability), domestic + cross-border
    pooled with `cell_key` as a fixed effect. `proximity_days` is signed
    (negative = approaching, positive = past the nearest crisis_warning/
    depeg_event) -- see design spec Sec 0's continuous-proximity decision.
    """
    gold_symbols = {
        cfg.symbol
        for sandbox_key in _H4_SANDBOX_KEYS
        for cfg in SANDBOX_CURRENCY_PAIRS[sandbox_key]
        if cfg.peg == "XAU"
    }

    decisions = (
        session.query(LLMDecisionRecord)
        .filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
        .all()
    )

    relevant_run_ids = {
        decision.simulation_id
        for decision in decisions
        if cell_key_from_run_id(decision.simulation_id) in _H4_CELLS
    }
    if not relevant_run_ids:
        return pd.DataFrame(columns=["agent_id", "chose_gold", "proximity_days", "agent_type", "actual_model", "cell_key"])

    intervention_rows = (
        session.query(InterventionLogRecord.run_id, InterventionLogRecord.timestep)
        .filter(
            InterventionLogRecord.run_id.in_(relevant_run_ids),
            InterventionLogRecord.shock_type.in_(_H4_PROXIMITY_SHOCK_TYPES),
        )
        .all()
    )
    event_days_by_run: dict[str, list[int]] = {}
    for run_id, timestep in intervention_rows:
        event_days_by_run.setdefault(run_id, []).append(timestep)

    records = []
    for decision in decisions:
        cell_key = cell_key_from_run_id(decision.simulation_id)
        if cell_key not in _H4_CELLS:
            continue
        event_days = event_days_by_run.get(decision.simulation_id)
        if not event_days:
            continue  # this cell/seed's data has no crisis/depeg event at all -- no proximity to measure
        records.append(
            {
                "agent_id": decision.agent_id,
                "chose_gold": 1 if decision.currency in gold_symbols else 0,
                "proximity_days": _signed_proximity(decision.timestep, event_days),
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
                "cell_key": cell_key,
            }
        )

    return pd.DataFrame.from_records(
        records, columns=["agent_id", "chose_gold", "proximity_days", "agent_type", "actual_model", "cell_key"]
    )
