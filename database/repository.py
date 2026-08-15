"""Thin DAO layer so simulation code never imports SQLAlchemy directly.

src/simulation/simulation_runner.py takes an on_timestep callback rather
than constructing its own session -- persist_timestep below is the function
callers (e2b/sandbox_launcher.py, scripts, notebooks) wire into that hook.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session
from pydantic import BaseModel

from database.models import (
    AgentMemoryLogRecord,
    AgentRecord,
    AgentStateRecord,
    CohortHoldingsRecord,
    HallucinationRecord,
    IndifferencePointRecord,
    InterventionLogRecord,
    LLMDecisionRecord,
    MarketSnapshotRecord,
    MetricRecord,
    NegotiationRecord,
    SimulationRunRecord,
    TimestepLogRecord,
    TransactionRecord,
    WalletRecord,
)
from src.agents.base_agent import BaseAgent
from src.agents.wealth import real_purchasing_power
from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.risk_adaptation import adapt_cara_coefficient, load_risk_adaptation_params
from src.llm.hallucination_detector import HallucinationDirection
from src.negotiation.conversation_history import ConversationLog
from src.simulation.environment import Environment
# Aliased to avoid colliding with database.models.LLMDecisionRecord (the
# SQLAlchemy table) above -- this is Task 5's thin per-decision outcome
# shape from TimestepResult.llm_decisions, a different class with the same
# name. Safe to import at module level: timestep.py itself never imports
# httpx/src.llm.* at module level (see its own docstring and
# tests/test_simulation.py's AST check), so this doesn't reintroduce the
# sandbox-breaking hard dependency database/repository.py otherwise avoids.
from src.simulation.timestep import LLMDecisionRecord as TimestepLLMDecisionRecord
from src.simulation.timestep import TimestepResult
from src.transactions.transaction import Transaction
from src.utils.helpers import generate_id


class LLMDecisionLogEntry(BaseModel):
    decision_id: str
    simulation_id: str
    timestep: int
    agent_id: str
    agent_type: str
    requested_model: str
    actual_model: str
    fallback_used: bool
    fallback_reason: str | None
    model_attempts: list[str]
    prompt_version: str
    rendered_prompt_hash: str
    system_prompt: str
    action: str
    currency: str
    chain: str
    amount: float
    price: float
    reported_reasoning: str
    negotiation_id: str | None
    round: int
    risk_profile: str
    utility_type: str
    utility_parameters: dict
    scenario: str
    domestic_or_cross_border: str
    governance_prompt_enabled: bool
    spread_optimal_currency: str = ""
    spread_optimal_chain: str = ""
    gas_optimal_currency: str = ""
    gas_optimal_chain: str = ""


class HallucinationLogEntry(BaseModel):
    decision_id: str | None = None
    transaction_id: str | None = None
    expected_price: float
    paid_price: float
    overpayment_pct: float
    direction: str
    is_hallucination: bool
    currency_symbol: str
    model_name: str | None = None


class MarketSnapshotLogEntry(BaseModel):
    source: str
    ticker: str
    price: float | None
    data_window: str | None
    negotiation_id: str | None = None


class SimulationRunLogEntry(BaseModel):
    run_id: str
    scenario_name: str
    research_mode: str
    random_seed: int
    model_roster_summary: str
    prompt_version_hash: str
    git_commit_hash: str
    config_hash: str


class InterventionLogEntry(BaseModel):
    run_id: str
    timestep: int
    shock_type: str
    target_currency: str | None = None
    target_issuer: str | None = None
    magnitude: float


class TimestepLogEntry(BaseModel):
    run_id: str
    timestep: int
    inflation_rate: float
    confidence_index: float
    eth_gas_fee_gwei: float
    solana_gas_fee_usd: float
    eur_usd_exchange_rate: float


class AgentStateLogEntry(BaseModel):
    run_id: str
    timestep: int
    agent_id: str
    risk_profile: str
    cara_coefficient: float | None = None
    real_purchasing_power: float
    wallet_balances: dict[str, float]
    utility_score: float


class AgentMemoryLogEntry(BaseModel):
    run_id: str
    timestep: int
    agent_id: str
    memory_type: str
    memory_text: str


class CohortHoldingsLogEntry(BaseModel):
    run_id: str
    risk_aversion_cohort: float
    currency_symbol: str
    pct_of_wealth: float


class IndifferencePointLogEntry(BaseModel):
    run_id: str
    hypothesis: str
    fixed_currency: str
    varied_currency: str
    varied_field: str
    risk_aversion_cohort: float
    compensation: float
    censored_fraction: float = 0.0


class AgentMemoryLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: AgentMemoryLogEntry) -> None:
        self.session.add(AgentMemoryLogRecord(**entry.model_dump()))


class TimestepLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: TimestepLogEntry) -> None:
        self.session.add(TimestepLogRecord(**entry.model_dump()))


class AgentStateRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: AgentStateLogEntry) -> None:
        self.session.add(AgentStateRecord(**entry.model_dump()))


class AgentRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert_agent(self, agent: BaseAgent, run_id: str) -> None:
        """Upserts this agent's identity row for `run_id`.

        `run_id` is part of the key, not decoration: `agents` is keyed
        `(run_id, id)` (see `AgentRecord`'s docstring) precisely because the
        13-cell matrix runs every cell with the same seeds and therefore the
        same deterministic agent ids. Without run-scoping here, a later
        cell's upsert found the earlier cell's row and silently overwrote its
        wallet mirror, and two cells racing across processes collided on
        `agents.id`.
        """
        record = self.session.get(AgentRecord, {"run_id": run_id, "id": agent.agent_id})
        if record is None:
            record = AgentRecord(
                run_id=run_id,
                id=agent.agent_id,
                agent_class=agent.agent_class,
                profile_name=agent.profile_name,
                risk_profile=agent.risk_profile,
                currency_zone=agent.currency_zone,
                assigned_model=agent.assigned_model,
                cara_coefficient=agent.cara_coefficient,
                created_at=datetime.now(timezone.utc),
            )
            self.session.add(record)
        self._sync_wallet(agent, run_id)

    def _sync_wallet(self, agent: BaseAgent, run_id: str) -> None:
        """Merges this agent's current balances into its wallet mirror for
        `run_id` only: currencies it already had are UPDATEd in place, newly
        held ones INSERTed, and ones it no longer holds DELETEd.

        Everything here is scoped by `run_id` AND `agent_id`. Scoping by
        `agent_id` alone (the original behavior) meant cell 2's very first
        day wiped cell 1's committed wallet rows for the same shared agent
        id, straight through all 13 cells -- last-write-wins clobbering with
        no error to notice it by. The net effect is still "replace", not
        "append": per-day history lives in `agent_states`, and this table
        holds only the latest snapshot.

        A merge, rather than the blind delete-everything-then-reinsert this
        used to do, because `wallets` is now keyed by its natural key
        `(run_id, agent_id, currency_symbol)` (see `WalletRecord`) and
        `matrix_runner` runs an entire cell/seed on ONE long-lived session.
        A blind reinsert re-derives, every simulated day, identity keys the
        session has already seen for rows it deleted moments earlier -- the
        identity-map hazard that previously argued for keeping a surrogate
        primary key here. Merging never re-derives a live identity key, so
        the natural key is safe by construction rather than by depending on
        how the ORM happens to synchronize a bulk DELETE with the identity
        map. It is also strictly less write traffic: the common case (same
        currencies, changed amounts) now emits UPDATEs instead of deleting
        and reinserting every row of every agent every day.
        """
        existing = {
            row.currency_symbol: row
            for row in self.session.query(WalletRecord)
            .filter(WalletRecord.run_id == run_id, WalletRecord.agent_id == agent.agent_id)
            .all()
        }
        for symbol, balance in agent.wallet.balances.items():
            row = existing.pop(symbol, None)
            if row is None:
                self.session.add(
                    WalletRecord(run_id=run_id, agent_id=agent.agent_id, currency_symbol=symbol, balance=balance)
                )
            elif row.balance != balance:
                row.balance = balance
        # Whatever is left in `existing` is a currency the agent held at the
        # last sync and does not hold now.
        for departed_row in existing.values():
            self.session.delete(departed_row)


class TransactionRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, tx: Transaction) -> None:
        self.session.add(
            TransactionRecord(
                id=tx.transaction_id,
                buyer_id=tx.buyer_id,
                seller_id=tx.seller_id,
                good_name=tx.good_name,
                currency_symbol=tx.currency_symbol,
                chain_name=tx.chain_name,
                gas_fee=tx.gas_fee,
                expected_value=tx.expected_value,
                paid_value=tx.paid_value,
                timestep=tx.timestep,
                status=tx.status.value,
                fx_tax_paid=tx.fx_tax_paid,
                timestamp=datetime.now(timezone.utc),
            )
        )


class NegotiationRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, log: ConversationLog, transaction_id: str | None = None) -> None:
        self.session.add(
            NegotiationRecord(
                transaction_id=transaction_id,
                rounds=len(log.offers),
                outcome=log.outcome or "unknown",
                log=[offer.model_dump() for offer in log.offers],
            )
        )


class MetricsRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, scenario_run_id: int, metric_name: str, timestep: int, value: float) -> None:
        self.session.add(
            MetricRecord(scenario_run_id=scenario_run_id, metric_name=metric_name, timestep=timestep, value=value)
        )


class LLMDecisionRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: LLMDecisionLogEntry) -> None:
        self.session.add(
            LLMDecisionRecord(
                **entry.model_dump(),
                timestamp=datetime.now(timezone.utc),
            )
        )


class HallucinationRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: HallucinationLogEntry) -> None:
        self.session.add(HallucinationRecord(**entry.model_dump()))


class MarketSnapshotRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: MarketSnapshotLogEntry) -> None:
        self.session.add(
            MarketSnapshotRecord(
                **entry.model_dump(),
                retrieval_timestamp=datetime.now(timezone.utc),
            )
        )


class SimulationRunRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: SimulationRunLogEntry) -> None:
        self.session.add(SimulationRunRecord(**entry.model_dump(), created_at=datetime.now(timezone.utc)))


class InterventionLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: InterventionLogEntry) -> None:
        self.session.add(InterventionLogRecord(**entry.model_dump()))


class CohortHoldingsRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: CohortHoldingsLogEntry) -> None:
        self.session.add(CohortHoldingsRecord(**entry.model_dump()))


class IndifferencePointRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: IndifferencePointLogEntry) -> None:
        self.session.add(IndifferencePointRecord(**entry.model_dump()))


def persist_timestep(
    session: Session, env: Environment, result: TimestepResult, run_id: str, commit: bool = True
) -> None:
    """Persist one day's agent/transaction/negotiation rows.

    `run_id` scopes the `agents`/`wallets` rows this function writes (see
    `AgentRepository.upsert_agent`). It is required, not optional-with-a-
    default: a default would silently reintroduce the cross-cell clobbering
    the composite `(run_id, id)` key exists to prevent.

    `commit` defaults to True (this function's original, standalone
    behavior: one commit covering just these rows). `persist_full_timestep`
    passes `commit=False` and does its own single commit at the very end,
    covering this function's writes plus its own -- see that function's
    docstring for why (Task 11 review Fix 2: the day must land in exactly
    one transaction, not two).
    """
    agent_repo = AgentRepository(session)
    tx_repo = TransactionRepository(session)
    negotiation_repo = NegotiationRepository(session)

    for agent in env.agents.values():
        agent_repo.upsert_agent(agent, run_id)
    for tx in result.transactions:
        tx_repo.record(tx)
    for log in result.negotiations:
        negotiation_repo.record(log)
    if commit:
        session.commit()


# A "no-op" CurrencyChainOption -- safety_multiplier=1.0 (governance_score=1,
# liquidity_score=1, peg_error=0), gas_fee=0.0 -- used only to resolve
# agent.utility_fn.evaluate() against the agent's own realized wealth,
# per the design spec Sec 7's utility_score resolution ("not a stakes
# decision, flagged for review"). currency_symbol/chain_name are unused
# labels; genius_compliant=True is the neutral/no-penalty choice for
# MultiAttributeUtility's compliance term (the only utility type that reads
# it) -- every UtilityFunction.evaluate() implementation accepts a `wealth`
# kwarg (or ignores it, for MultiAttributeUtility), so one call here works
# polymorphically across cara/risk_neutral/crra/multi_attribute/
# epstein_zin_proxy without per-type branching.
_NEUTRAL_UTILITY_CANDIDATE = CurrencyChainOption(
    currency_symbol="NEUTRAL",
    chain_name="NEUTRAL",
    governance_score=1.0,
    liquidity_score=1.0,
    peg_error=0.0,
    gas_fee=0.0,
    finality_seconds=0.0,
    genius_compliant=True,
)


def _agent_utility_score(agent: BaseAgent, w_real_after: float) -> float:
    return agent.utility_fn.evaluate(_NEUTRAL_UTILITY_CANDIDATE, wealth=w_real_after)


def _llm_decision_utility_parameters(agent: BaseAgent | None) -> dict:
    if agent is None:
        return {}
    params: dict = {}
    if agent.risk_aversion is not None:
        params["risk_aversion"] = agent.risk_aversion
    if agent.eis is not None:
        params["eis"] = agent.eis
    if agent.multi_attribute_weights is not None:
        params.update(agent.multi_attribute_weights.model_dump())
    return params


def _llm_decision_log_entry(
    decision: TimestepLLMDecisionRecord,
    decision_id: str,
    run_id: str,
    timestep: int,
    agent: BaseAgent | None,
    scenario_name: str,
) -> LLMDecisionLogEntry:
    # Function-local: agent_reasoning.py imports httpx at module level, so
    # this must not become a module-level import of database/repository.py
    # (see this module's docstring on why persist_timestep/
    # persist_full_timestep must stay importable without httpx installed).
    from src.llm.agent_reasoning import PROMPT_VERSIONS, hash_rendered_prompt

    # Known limitation (documented in task-11-report.md): Task 5's
    # TimestepLLMDecisionRecord is deliberately a thin, additive shape (see
    # its own docstring in src/simulation/timestep.py) that does not retain
    # the per-decision domestic_or_cross_border/governance_prompt_enabled
    # context that only exists inside run_timestep's buyer/seller loop.
    # Reconstructing those faithfully after the fact from result
    # .llm_decisions alone is not possible without threading more state
    # through Task 5's record (out of this task's scope) -- so those two
    # fields use documented placeholders rather than fabricated-but-wrong
    # values. `rendered_prompt` (and therefore `rendered_prompt_hash`/
    # `system_prompt` below), however, IS carried through `decision
    # .rendered_prompt` (a Task 11 review fix: this previously hashed
    # `decision.reasoning` -- the model's own OUTPUT text -- which produced a
    # real-looking, per-row-unique hash that was simply wrong, since the
    # column's whole contract is "hash of what the model actually saw").
    prompt_version = PROMPT_VERSIONS.get(decision.agent_type, "unknown")
    # Phase 3 assigns exactly one fixed model per agent with no fallback
    # chain (design spec Sec 1.2) -- fallback_used is always False here;
    # failure_reason (a total-decision-failure diagnostic, not a
    # model-substitution reason) is carried in fallback_reason since no
    # dedicated column exists for it.
    model_attempts = [decision.actual_model] * (decision.correction_attempts + 1)

    return LLMDecisionLogEntry(
        decision_id=decision_id,
        simulation_id=run_id,
        timestep=timestep,
        agent_id=decision.agent_id,
        agent_type=decision.agent_type,
        requested_model=decision.requested_model,
        actual_model=decision.actual_model,
        fallback_used=False,
        fallback_reason=decision.failure_reason,
        model_attempts=model_attempts,
        prompt_version=prompt_version,
        rendered_prompt_hash=hash_rendered_prompt(decision.rendered_prompt or ""),
        system_prompt=decision.rendered_prompt or "",
        action=decision.action or "NONE",
        currency=decision.currency_symbol or "",
        chain=decision.chain_name or "",
        amount=decision.amount if decision.amount is not None else 0.0,
        price=decision.price if decision.price is not None else 0.0,
        reported_reasoning=decision.reasoning or (decision.failure_reason or ""),
        negotiation_id=decision.negotiation_id,
        round=decision.round if decision.round is not None else 0,
        risk_profile=decision.risk_profile,
        utility_type=decision.utility_type,
        utility_parameters=_llm_decision_utility_parameters(agent),
        scenario=scenario_name,
        domestic_or_cross_border="unknown",
        governance_prompt_enabled=False,
        spread_optimal_currency=decision.spread_optimal_currency or "",
        spread_optimal_chain=decision.spread_optimal_chain or "",
        gas_optimal_currency=decision.gas_optimal_currency or "",
        gas_optimal_chain=decision.gas_optimal_chain or "",
    )


def persist_full_timestep(session: Session, env: Environment, result: TimestepResult, run_id: str) -> None:
    """Ties together every per-day/per-agent/per-decision persistence table
    (Tasks 2, 3, 5, 7, 8) into one call per simulated day. Extends
    `persist_timestep` (calls it, with `commit=False`, for its existing
    agent/transaction/negotiation behavior) rather than duplicating it.

    Atomicity (Task 11 review Fix 2): a whole simulated day must land in
    exactly ONE transaction. `persist_timestep` used to end with its own
    `session.commit()`, and this function committed again at the end -- so
    agent/transaction/negotiation rows became durable before the rest of the
    day's rows (timestep/agent-state/intervention/memory/LLM-decision) were
    even flushed; a failure partway through this function's own work left a
    silently half-persisted day. Fixed by threading `commit=False` through to
    `persist_timestep` so it only adds rows to the session, and doing the one
    real `session.commit()` at the very end of this function, after every
    repository call below -- a raised exception anywhere before that point
    now leaves the whole day uncommitted (a caller-triggered rollback drops
    all of it, not just "everything after persist_timestep").

    CARA-adaptation wiring (the integration gap this task closes): Task 7's
    `adapt_cara_coefficient` was built as a standalone function, never
    called from `run_timestep` or any day-loop driver -- nothing in
    production ever adapted an agent's cara_coefficient before this
    function. This is the first place that has a natural reason to compare
    an agent's real purchasing power day-over-day (it already needs
    `real_purchasing_power` for `AgentStateLogEntry`), so it drives that
    comparison itself: `env.previous_real_purchasing_power` (keyed by
    agent_id) holds each agent's most recent real purchasing power. On an
    agent's first-ever call, there is no genuine "before" to compare
    against, so adaptation is skipped and the value is just seeded;
    `adapt_cara_coefficient` is a no-op for non-CARA-eligible
    (cara_coefficient is None) agents automatically.
    """
    persist_timestep(session, env, result, run_id=run_id, commit=False)

    risk_adaptation_params = load_risk_adaptation_params()

    timestep_repo = TimestepLogRepository(session)
    agent_state_repo = AgentStateRepository(session)
    agent_memory_repo = AgentMemoryLogRepository(session)
    intervention_repo = InterventionLogRepository(session)
    llm_decision_repo = LLMDecisionRepository(session)
    hallucination_repo = HallucinationRepository(session)

    # `env.chains` is always the full chain universe: Environment.build and
    # Environment.build_from_population both call load_chain_universe()
    # unconditionally (currency restriction, e.g. the 6 factor-isolation
    # sandboxes' SANDBOX_CURRENCY_PAIRS, only ever narrows env.currencies,
    # never env.chains) -- so "ethereum"/"solana" are always present here.
    # No `if ... in env.chains` guard is needed; one would protect against
    # nothing real.
    #
    # Task 11 review Fix 3: ChainConfig.gas_fee (src/blockchain/chain.py) is
    # USD-denominated everywhere else in this codebase (subtracted directly
    # from wealth in src/utility/cara.py, risk_neutral.py, epstein_zin.py,
    # and carried as Transaction.gas_fee) -- there is no gwei-conversion
    # mechanism anywhere in the codebase (confirmed: no "gwei" constant or
    # helper exists outside this table's own column name and its tests).
    # `eth_gas_fee_gwei` below therefore currently holds a raw USD value
    # despite its name implying gwei units -- a pre-existing schema/naming
    # mismatch from an earlier plan. Inventing a fake USD->gwei conversion
    # here would be exactly the kind of unrequested economic assumption this
    # project's conventions avoid, so this fix leaves the value as-is and
    # flags it here for a future plan to decide whether to rename the
    # column; it is NOT fixed by this pass.
    timestep_repo.record(
        TimestepLogEntry(
            run_id=run_id,
            timestep=result.day,
            inflation_rate=env.macro_state.inflation,
            confidence_index=env.macro_state.confidence_index,
            eth_gas_fee_gwei=env.chains["ethereum"].gas_fee,
            solana_gas_fee_usd=env.chains["solana"].gas_fee,
            eur_usd_exchange_rate=env.macro_state.peg_reference_rates.get("EUR", 1.0),
        )
    )

    for agent in env.agents.values():
        w_real_after = real_purchasing_power(agent.wallet, env.exchange_rates, env.price_index)

        w_real_before = env.previous_real_purchasing_power.get(agent.agent_id)
        if w_real_before is not None:
            adapt_cara_coefficient(
                agent, w_real_before=w_real_before, w_real_after=w_real_after, params=risk_adaptation_params
            )
        env.previous_real_purchasing_power[agent.agent_id] = w_real_after

        agent_state_repo.record(
            AgentStateLogEntry(
                run_id=run_id,
                timestep=result.day,
                agent_id=agent.agent_id,
                risk_profile=agent.risk_profile,
                cara_coefficient=agent.cara_coefficient,
                real_purchasing_power=w_real_after,
                wallet_balances=dict(agent.wallet.balances),
                utility_score=_agent_utility_score(agent, w_real_after),
            )
        )

    for shock in result.fired_shocks:
        intervention_repo.record(
            InterventionLogEntry(
                run_id=run_id,
                timestep=result.day,
                shock_type=shock.type.value,
                target_currency=shock.target_currency,
                target_issuer=shock.target_issuer,
                magnitude=shock.magnitude,
            )
        )

    for agent_id, memory_type, memory_text in result.memory_events:
        agent_memory_repo.record(
            AgentMemoryLogEntry(
                run_id=run_id,
                timestep=result.day,
                agent_id=agent_id,
                memory_type=memory_type,
                memory_text=memory_text,
            )
        )

    for decision in result.llm_decisions:
        decision_id = generate_id("dec")
        agent = env.agents.get(decision.agent_id)
        llm_decision_repo.record(
            _llm_decision_log_entry(decision, decision_id, run_id, result.day, agent, env.scenario.name)
        )

        if decision.hallucination is not None:
            hallucination_repo.record(
                HallucinationLogEntry(
                    decision_id=decision_id,
                    transaction_id=None,
                    expected_price=decision.hallucination.expected_value,
                    paid_price=decision.hallucination.paid_value,
                    overpayment_pct=decision.hallucination.percentage_error,
                    direction=decision.hallucination.direction.value,
                    is_hallucination=decision.hallucination.direction != HallucinationDirection.ACCURATE,
                    currency_symbol=decision.hallucination.currency_symbol or "",
                    model_name=decision.hallucination.actual_model,
                )
            )

    session.commit()
