"""Per-hypothesis dataset builders: each function returns a `pandas.
DataFrame`, one row per eligible `LLMDecisionRecord`, ready for `src.
econometrics.regression_engine.fit_clustered_logit`. See
docs/superpowers/specs/2026-08-02-phase3-plan5-econometrics-design.md
Sec 1 for the exact per-hypothesis data-source/dependent-variable/
regressor design this implements.
"""

from typing import Callable

import pandas as pd
from sqlalchemy.orm import Session

from database.models import AgentRecord, AgentStateRecord, InterventionLogRecord, LLMDecisionRecord, TimestepLogRecord
from src.currencies.currency import CurrencyConfig, load_currency_universe
from src.currencies.gold_token import GoldBackedConfig
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.currencies.tokenized_deposit import TokenizedDepositConfig
from src.legacy.econometrics.cell_identity import cell_key_from_run_id
from src.economy.fx_tax import currency_zone_of

# Every LLM currency-choice decision is one observation (Plan 5 whole-branch
# review Fix I2): COUNTER_OFFER is included alongside ACCEPT/OFFER -- in a
# multi-round run_llm_negotiation, counter-offers are the majority of
# logged decisions after round 0 and are themselves genuine currency/chain
# choices; excluding them shrinks N and selects on negotiation outcome
# (only openers and terminal accepts would survive).
_DECIDED_ACTIONS = ("ACCEPT", "OFFER", "COUNTER_OFFER")


def _safe_cell_key(run_id: str) -> str | None:
    """`cell_key_from_run_id` raises `ValueError` for any `run_id` not
    produced by `run_matrix` (Plan 5 whole-branch review Fix C2) -- e.g.
    `experiments/experiment_007_governance_prompting.py` writes
    `LLMDecisionRecord` rows to the same default database with its own
    `simulation_id` scheme. A foreign row sharing the database is a
    co-tenant, not a misconfiguration, so this returns `None` instead of
    propagating the exception; every builder below treats `None` as
    "does not belong to any of my cells" and skips the row."""
    try:
        return cell_key_from_run_id(run_id)
    except ValueError:
        return None


def _matches_matrix_run_id(simulation_id: str, matrix_run_id: str | None) -> bool:
    """Scopes a dataset to one `run_matrix` invocation (Plan 5 whole-branch
    review Fix C3): without this, two separate `run_matrix` calls against
    the same database (e.g. a dry-run smoke test followed by the real run)
    would silently pool together, since every builder otherwise queries
    every `LLMDecisionRecord` row in the table regardless of which
    `matrix_run_id` produced it. `matrix_run_id=None` (the default)
    preserves the original pool-everything behavior for callers that
    intentionally want it (e.g. a single-call test fixture)."""
    return matrix_run_id is None or simulation_id.startswith(f"{matrix_run_id}-")


def _join_cara_a(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Joins each row's agent's CARA `a` AT THAT DECISION'S timestep from
    `AgentStateRecord` -- the correct source per the design spec Sec 1 (NOT
    `LLMDecisionRecord.utility_parameters`, which omits risk-neutral
    agents' `a=0.0` entirely). `df` must already have `run_id`/`timestep`/
    `agent_id` columns.

    Join key is `timestep - 1` (clamped to 0), not `timestep` (Plan 5
    whole-branch review Fix I1): `persist_full_timestep` calls `adapt_cara_
    coefficient` BEFORE writing that day's `AgentStateRecord`
    (`database/repository.py`), so the row at `timestep == d` holds the
    value AFTER day `d`'s own realized-loss adaptation -- i.e. the value
    that will be used for day `d+1`'s decisions, not the value day `d`'s
    OWN decisions were actually made with. Day 0 is the exception: its
    first-ever adaptation call is a no-op seed (no "before" to compare
    against), so `AgentStateRecord.timestep==0` already holds the
    un-adapted initial value -- exactly what day 0's decisions used --
    which is why `max(timestep - 1, 0)` (not `timestep - 1` unclamped)
    is correct for day 0 too.

    Rows with no matching `AgentStateRecord` (shouldn't happen in
    practice -- every persisted day writes one per agent -- but
    defensively dropped rather than silently coerced) are excluded.
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

    df = df.assign(_join_timestep=(df["timestep"] - 1).clip(lower=0))
    merged = df.merge(
        states_df,
        left_on=["run_id", "_join_timestep", "agent_id"],
        right_on=["run_id", "timestep", "agent_id"],
        how="left",
        suffixes=("", "_state"),
    )
    merged = merged.drop(columns=["_join_timestep", "timestep_state"])
    return merged.dropna(subset=["cara_a"]).reset_index(drop=True)


def build_h1_dataset(session: Session, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H1: higher CARA `a` -> stronger preference for USD-zone stablecoins
    over EUR-zone. Master cell only (the only cell with real currency-zone
    variation). Gold-backed/zone-neutral decisions (currency_zone_of
    returns None) are excluded -- H1 is a USD-vs-EUR contrast only."""
    currencies = load_currency_universe()

    query = session.query(LLMDecisionRecord).filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
    decisions = [d for d in query.all() if _matches_matrix_run_id(d.simulation_id, matrix_run_id)]

    records = []
    for decision in decisions:
        if _safe_cell_key(decision.simulation_id) != "master":
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


def build_h2_dataset(session: Session, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H2: higher CARA `a` -> prioritizes low spread (liquidity_score, the
    codebase's spread proxy) over low gas fees. Master cell only. Keeps
    only decisions where the round's spread-optimal and gas-optimal
    candidates DIFFERED (a genuine tradeoff existed) AND the agent's
    actual choice reveals a preference for one side or the other.

    Classification is by CURRENCY (for spread) and CHAIN (for gas), not
    exact-tuple equality (Plan 5 whole-branch review Fix I3):
    `generate_candidates` sets `gas_fee` from the chain alone (`src/
    blockchain/routing_engine.py`), so every currency on the cheapest
    chain ties on gas -- `gas_optimal_currency` is therefore an arbitrary
    tie-break, not a meaningful "gas-optimal currency." Requiring an exact
    `(currency, chain)` match against that arbitrary tuple silently
    misclassifies (or discards) any decision that legitimately chose the
    gas-optimal CHAIN with a different currency than the tie-break
    happened to pick. A decision is `chose_spread_optimal=1` only if it
    picked the spread-optimal currency AND NOT the gas-optimal chain;
    `=0` only if the reverse; a decision landing on both (possible under a
    gas tie) or neither is excluded as not revealing a preference between
    the two.
    """
    query = session.query(LLMDecisionRecord).filter(
        LLMDecisionRecord.action.in_(_DECIDED_ACTIONS),
        LLMDecisionRecord.spread_optimal_currency.isnot(None),
        LLMDecisionRecord.spread_optimal_currency != "",
    )
    decisions = [d for d in query.all() if _matches_matrix_run_id(d.simulation_id, matrix_run_id)]

    records = []
    for decision in decisions:
        if _safe_cell_key(decision.simulation_id) != "master":
            continue
        if (
            decision.spread_optimal_currency == decision.gas_optimal_currency
            and decision.spread_optimal_chain == decision.gas_optimal_chain
        ):
            continue  # no genuine tradeoff this round

        chose_spread_currency = decision.currency == decision.spread_optimal_currency
        chose_gas_chain = decision.chain == decision.gas_optimal_chain
        if chose_spread_currency and not chose_gas_chain:
            chose_spread_optimal = 1
        elif chose_gas_chain and not chose_spread_currency:
            chose_spread_optimal = 0
        else:
            continue  # picked both (possible under a gas tie) or neither -- ambiguous, excluded

        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_spread_optimal": chose_spread_optimal,
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


def build_h3_dataset(session: Session, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H3: higher CARA `a` -> prioritizes GENIUS Act compliance/governance
    over liquidity. The `liquidity_vs_governance` sandbox (domestic +
    cross-border pooled, with `cell_key` as a fixed effect distinguishing
    the two -- see design spec Sec 1)."""
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[_H3_SANDBOX_KEY]
    higher_governance_symbol = (
        option_a.symbol if option_a.governance_score >= option_b.governance_score else option_b.symbol
    )

    query = session.query(LLMDecisionRecord).filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
    decisions = [d for d in query.all() if _matches_matrix_run_id(d.simulation_id, matrix_run_id)]

    records = []
    for decision in decisions:
        cell_key = _safe_cell_key(decision.simulation_id)
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
_H4_CELLS = {f"{key}_{suffix}" for key in _H4_SANDBOX_KEYS for suffix in ("domestic", "cross_border")} | {"master"}
_H4_PROXIMITY_SHOCK_TYPES = ("crisis_warning", "depeg_event")


def _nearest_event_distance(timestep: int, event_days: list[int]) -> int:
    """Absolute distance (in days) from `timestep` to the nearest crisis_
    warning/depeg_event day for this run -- unsigned (Plan 5 whole-branch
    review Fix C4): H4 claims CLOSER proximity (smaller distance) predicts
    a STRONGER shift to gold, a claim about magnitude, not direction. A
    signed distance is not monotonic in "closeness" (a day equally before
    or after the event has the same closeness but opposite sign), so it
    cannot test this claim; the expected relationship under H4 is a
    NEGATIVE coefficient on this distance (farther away -> less likely
    gold)."""
    return min(abs(day - timestep) for day in event_days)


def build_h4_dataset(session: Session, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H4: closer crisis/depeg proximity -> stronger shift to gold-backed
    tokens. The two sandboxes with a gold option (asset_backing_vs_
    liquidity, asset_backing_vs_stability) PLUS the master cell's own H4
    proximity sweep (Plan 5 whole-branch review Fix C4 -- design spec
    Sec 1 requires master's sweep instances; omitting them also left the
    remaining 4 sandbox cells sharing one identical fixed crisis-proximity
    gap, making `proximity_days` collinear with the day index rather than
    a genuine proximity effect). `cell_key` is a fixed effect distinguishing
    all 5 cells. `proximity_days` is the unsigned distance to the nearest
    eligible crisis_warning/depeg_event day (see `_nearest_event_distance`).

    Crisis/depeg events that themselves TARGET a gold-backed symbol are
    excluded from the proximity signal (Fix C4): master's own sweep
    includes a pair targeting PAXG itself (day 300/320) -- a threat TO
    gold is not a "flee toward gold" trigger, and counting it would give
    exactly the wrong sign for a subset of the sample.
    """
    real_currencies = load_currency_universe()
    gold_symbols = {
        cfg.symbol
        for sandbox_key in _H4_SANDBOX_KEYS
        for cfg in SANDBOX_CURRENCY_PAIRS[sandbox_key]
        if cfg.peg == "XAU"
    } | {symbol for symbol, cfg in real_currencies.items() if cfg.peg == "XAU"}

    query = session.query(LLMDecisionRecord).filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
    decisions = [d for d in query.all() if _matches_matrix_run_id(d.simulation_id, matrix_run_id)]

    relevant_run_ids = {
        decision.simulation_id for decision in decisions if _safe_cell_key(decision.simulation_id) in _H4_CELLS
    }
    if not relevant_run_ids:
        return pd.DataFrame(columns=["agent_id", "chose_gold", "proximity_days", "agent_type", "actual_model", "cell_key"])

    intervention_rows = (
        session.query(
            InterventionLogRecord.run_id, InterventionLogRecord.timestep, InterventionLogRecord.target_currency
        )
        .filter(
            InterventionLogRecord.run_id.in_(relevant_run_ids),
            InterventionLogRecord.shock_type.in_(_H4_PROXIMITY_SHOCK_TYPES),
        )
        .all()
    )
    event_days_by_run: dict[str, list[int]] = {}
    for run_id, timestep, target_currency in intervention_rows:
        if target_currency in gold_symbols:
            continue  # a threat TO gold itself isn't a "flee toward gold" trigger
        event_days_by_run.setdefault(run_id, []).append(timestep)

    records = []
    for decision in decisions:
        cell_key = _safe_cell_key(decision.simulation_id)
        if cell_key not in _H4_CELLS:
            continue
        event_days = event_days_by_run.get(decision.simulation_id)
        if not event_days:
            continue  # this cell/seed's data has no eligible crisis/depeg event -- no proximity to measure
        records.append(
            {
                "agent_id": decision.agent_id,
                "chose_gold": 1 if decision.currency in gold_symbols else 0,
                "proximity_days": _nearest_event_distance(decision.timestep, event_days),
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
                "cell_key": cell_key,
            }
        )

    return pd.DataFrame.from_records(
        records, columns=["agent_id", "chose_gold", "proximity_days", "agent_type", "actual_model", "cell_key"]
    )


_H5_VOLATILITY_WINDOW_DAYS = 30  # trailing window for realized EUR/USD volatility -- see design spec Sec 1


def _rolling_volatility(rates_by_day: dict[int, float], day: int, window: int) -> float | None:
    """Sample standard deviation of `eur_usd_exchange_rate` over the
    `window` days up to and including `day`. Returns None if fewer than 2
    days of history exist yet (std of a single point is undefined)."""
    window_days = [d for d in rates_by_day if day - window < d <= day]
    if len(window_days) < 2:
        return None
    values = pd.Series([rates_by_day[d] for d in window_days])
    return float(values.std(ddof=1))


def build_h5_dataset(session: Session, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H5: higher EUR/USD volatility -> stronger preference for USD-zone
    stablecoins in cross-border settlement. Master cell only, filtered to
    decisions belonging to a negotiation whose participants' `currency_
    zone`s genuinely differ (Plan 5 whole-branch review Fix I5 -- the
    design spec's actual requirement, implemented via `LLMDecisionRecord
    .negotiation_id` grouping: every decision sharing a negotiation_id is
    one negotiation's buyer/seller pair, so the SET of zones across a
    negotiation's own decisions reveals whether it was cross-zone. This
    replaces an earlier approximation that only checked the deciding
    agent's own zone was set, which for a 100%-zoned Plan-3 population
    made H5's sample identical to H1's and never actually tested
    "cross-border" at all).
    """
    currencies = load_currency_universe()

    query = session.query(LLMDecisionRecord).filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
    decisions = [d for d in query.all() if _matches_matrix_run_id(d.simulation_id, matrix_run_id)]
    master_decisions = [d for d in decisions if _safe_cell_key(d.simulation_id) == "master"]
    if not master_decisions:
        return pd.DataFrame(columns=["agent_id", "chose_usd_zone", "eur_usd_volatility", "agent_type", "actual_model"])

    # Scoped by simulation_id (at most a handful of run_ids), NOT by
    # negotiation_id (Plan 5 whole-branch review, second pass): at real
    # scale a single master run can have tens of thousands of negotiations
    # -- an IN clause with one bound parameter per negotiation_id blows
    # past SQLite's SQLITE_MAX_VARIABLE_NUMBER (32766) and crashes with
    # "too many SQL variables" (reproduced directly). Querying by run_id
    # first and grouping by negotiation_id in Python afterward gets the
    # exact same result set without binding one parameter per negotiation.
    run_ids = {d.simulation_id for d in master_decisions}
    neg_participants = (
        session.query(LLMDecisionRecord.negotiation_id, LLMDecisionRecord.simulation_id, LLMDecisionRecord.agent_id)
        .filter(LLMDecisionRecord.simulation_id.in_(run_ids), LLMDecisionRecord.negotiation_id.isnot(None))
        .all()
    )
    # Each negotiation's participants are tracked as `(run_id, agent_id)`
    # pairs, not bare agent ids, because the zone lookup below is keyed the
    # same way -- see the comment on `agent_zones`.
    agents_by_negotiation: dict[str, set[tuple[str, str]]] = {}
    for neg_id, sim_id, agent_id in neg_participants:
        agents_by_negotiation.setdefault(neg_id, set()).add((sim_id, agent_id))

    # Zones are looked up per `(run_id, agent_id)`, and the filter is scoped
    # by run_id as well as agent id (round-2 review finding I1). `agents` is
    # keyed `(run_id, id)` (see database/models.py's AgentRecord docstring)
    # and agent ids are a pure function of `(profile_name, seed, slot_index)`,
    # so the same id exists once per cell/seed/matrix_run_id sharing this
    # database. Filtering on `AgentRecord.id` alone returned up to one row PER
    # run_id for each id and `dict(...)` silently kept whichever arrived last,
    # meaning a negotiation's cross-zone test could be decided by a DIFFERENT
    # run's agent rows. That happened to be harmless only while
    # `currency_zone` was a pure function of the agent id -- an invariant of
    # today's `generate_agent_population` that nothing asserts and no caller
    # is entitled to rely on.
    all_agent_ids = {agent_id for participants in agents_by_negotiation.values() for _, agent_id in participants}
    agent_zones = {
        (row_run_id, agent_id): zone
        for row_run_id, agent_id, zone in session.query(
            AgentRecord.run_id, AgentRecord.id, AgentRecord.currency_zone
        )
        .filter(AgentRecord.run_id.in_(run_ids), AgentRecord.id.in_(all_agent_ids))
        .all()
    }

    cross_zone_negotiations = set()
    for neg_id, participants in agents_by_negotiation.items():
        zones = {agent_zones.get(key) for key in participants} - {None}
        if len(zones) >= 2:
            cross_zone_negotiations.add(neg_id)

    timestep_rows = (
        session.query(TimestepLogRecord.run_id, TimestepLogRecord.timestep, TimestepLogRecord.eur_usd_exchange_rate)
        .filter(TimestepLogRecord.run_id.in_(run_ids))
        .all()
    )
    rates_by_run: dict[str, dict[int, float]] = {}
    for run_id, timestep, rate in timestep_rows:
        rates_by_run.setdefault(run_id, {})[timestep] = rate

    records = []
    for decision in master_decisions:
        if decision.negotiation_id not in cross_zone_negotiations:
            continue
        currency = currencies.get(decision.currency)
        if currency is None:
            continue
        zone = currency_zone_of(currency)
        if zone is None:
            continue
        volatility = _rolling_volatility(
            rates_by_run.get(decision.simulation_id, {}), decision.timestep, _H5_VOLATILITY_WINDOW_DAYS
        )
        if volatility is None:
            continue
        records.append(
            {
                "agent_id": decision.agent_id,
                "chose_usd_zone": 1 if zone == "USD" else 0,
                "eur_usd_volatility": volatility,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
            }
        )

    return pd.DataFrame.from_records(
        records, columns=["agent_id", "chose_usd_zone", "eur_usd_volatility", "agent_type", "actual_model"]
    )


def build_sandbox_preference_dataset(
    session: Session,
    sandbox_key: str,
    higher_option_selector: Callable[[CurrencyConfig, CurrencyConfig], str],
    cell_variant: str,
    matrix_run_id: str | None = None,
) -> pd.DataFrame:
    """Shared H7-H11 dataset builder (Plan 6b): per-decision logit sample
    for exactly ONE of a sandbox's two cells (domestic XOR cross_border --
    unlike H3, which pools both with a cell_key fixed effect; H7-H11 report
    each cell variant separately per the Plan 6 design spec Sec 1). One row
    per eligible LLMDecisionRecord: `chose_higher_option=1` if the agent's
    proposed currency is `higher_option_selector`'s pick, `0` if it's the
    sandbox's other option, excluded entirely if the decision's currency
    isn't one of this sandbox's two symbols at all.

    `cell_variant` must be `"domestic"` or `"cross_border"` -- any other
    value raises ValueError immediately (a typo here should never silently
    return an empty/wrong-cell dataset).
    """
    if cell_variant not in ("domestic", "cross_border"):
        raise ValueError(f"cell_variant must be 'domestic' or 'cross_border', got {cell_variant!r}")

    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    higher_option_symbol = higher_option_selector(option_a, option_b)
    target_cell_key = f"{sandbox_key}_{cell_variant}"

    query = session.query(LLMDecisionRecord).filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
    decisions = [d for d in query.all() if _matches_matrix_run_id(d.simulation_id, matrix_run_id)]

    records = []
    for decision in decisions:
        if _safe_cell_key(decision.simulation_id) != target_cell_key:
            continue
        if decision.currency not in (option_a.symbol, option_b.symbol):
            continue
        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_higher_option": 1 if decision.currency == higher_option_symbol else 0,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
            }
        )

    df = pd.DataFrame.from_records(
        records,
        columns=["run_id", "timestep", "agent_id", "chose_higher_option", "agent_type", "actual_model"],
    )
    return _join_cara_a(session, df)


def build_h7_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H7: higher CARA `a` -> prioritizes peg stability (lower peg_error)
    over governance/compliance. governance_vs_stability sandbox."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="governance_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )


def build_h8_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H8: higher CARA `a` -> prioritizes peg stability over liquidity.
    liquidity_vs_stability sandbox."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="liquidity_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )


def build_h9_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H9: higher CARA `a` -> prioritizes gold/hard-asset backing over
    liquidity. asset_backing_vs_liquidity sandbox (static baseline
    preference, not crisis-proximity-driven like H4)."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="asset_backing_vs_liquidity",
        higher_option_selector=lambda a, b: a.symbol if isinstance(a, GoldBackedConfig) else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )


def build_h10_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H10: higher CARA `a` -> prioritizes the FDIC-insured deposit option
    (better peg + insurance) over gold backing. asset_backing_vs_stability
    sandbox. Lower-confidence hypothesis (approved as-is, see design spec
    Sec 1): this sandbox bundles asset-class AND a large peg_error gap
    (0.015 vs 0.0001) in one swap."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="asset_backing_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if isinstance(a, TokenizedDepositConfig) else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )


def build_h11_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H11: higher CARA `a` -> prioritizes governance/compliance quality
    over asset-backing type. asset_backing_vs_governance sandbox.
    Lower-confidence hypothesis (approved as-is, see design spec Sec 1):
    the two options' governance_score (0.75 vs 0.70) and issuer_risk (0.25
    vs 0.20) are close, a subtler contrast than the other pairs."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="asset_backing_vs_governance",
        higher_option_selector=lambda a, b: a.symbol if a.governance_score >= b.governance_score else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )
