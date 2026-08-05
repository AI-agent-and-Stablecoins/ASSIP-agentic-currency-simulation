"""Defines database tables: agents, wallets, transactions, negotiations,
hallucinations, scenarios, metrics.

Source of truth for the schema -- database/schema.sql was removed in favor
of Base.metadata.create_all(), so there's exactly one place the schema is
defined (Alembic-style migrations are more machinery than this research
tool needs).
"""

from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, ForeignKeyConstraint, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON, DateTime


class Base(DeclarativeBase):
    pass


class AgentRecord(Base):
    """Per-run agent identity, keyed by `(run_id, id)`.

    Run-scoping (not merely `id`) is load-bearing, not defensive: agent ids
    from `src/agents/population.py`'s `generate_agent_population` are a pure
    function of `(profile_name, seed, slot_index)`
    (`f"{profile_name}-seed{seed}-{slot_index:03d}"`), and the 13-cell
    experiment matrix (`src/simulation/matrix_runner.py`'s `run_matrix`)
    runs EVERY cell with the SAME seeds -- so all 13 cells generate the
    identical 100 agent ids. With the original bare `id` primary key that
    meant each later cell silently overwrote the previous cell's `agents`
    (and `wallets`) rows for those shared ids, even in a fully sequential
    single-process run; under `distributed_matrix_runner`'s cross-process
    runner two cells racing on the same insert also produced an intermittent
    `UNIQUE constraint failed: agents.id` IntegrityError. Composite
    `(run_id, id)` gives each cell/seed its own independent agent rows,
    matching the pattern `AgentStateRecord` (`(run_id, timestep, agent_id)`)
    already used correctly.
    """

    __tablename__ = "agents"

    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"), primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_class: Mapped[str] = mapped_column(String)
    profile_name: Mapped[str] = mapped_column(String)
    risk_profile: Mapped[str] = mapped_column(String)
    currency_zone: Mapped[str | None] = mapped_column(String, nullable=True)
    assigned_model: Mapped[str | None] = mapped_column(String, nullable=True)
    cara_coefficient: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class WalletRecord(Base):
    """Latest-known wallet mirror, keyed by its natural key
    `(run_id, agent_id, currency_symbol)`.

    A row here means "the latest known balance of ONE currency, for ONE
    agent, in ONE run", so that triple IS the row's identity -- the same
    pattern `AgentStateRecord` (`(run_id, timestep, agent_id)`) and
    `AgentRecord` (`(run_id, id)`) already use. `run_id`'s presence in the
    key is the load-bearing part (see `AgentRecord`'s docstring for why the
    13-cell matrix makes agent ids collide across cells); including
    `currency_symbol` makes a duplicated run/agent/currency triple
    impossible at the database level rather than merely unlikely because
    `AgentRepository._sync_wallet` happens to rewrite carefully.

    An earlier revision kept a surrogate autoincrement `id` here out of
    concern that `_sync_wallet`'s delete-then-reinsert, run every simulated
    day on `matrix_runner`'s single long-lived per-cell session, would make
    each day's re-insert collide with a just-deleted row's entry in
    SQLAlchemy's identity map. `_sync_wallet` no longer deletes and
    reinserts: it merges (UPDATE in place / INSERT new symbols / ORM-DELETE
    departed symbols), which never re-derives an identity key that is
    already live in the session. See that method's docstring.
    """

    __tablename__ = "wallets"
    __table_args__ = (ForeignKeyConstraint(["run_id", "agent_id"], ["agents.run_id", "agents.id"]),)

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    currency_symbol: Mapped[str] = mapped_column(String, primary_key=True)
    balance: Mapped[float] = mapped_column(Float)


class TransactionRecord(Base):
    """`buyer_id`/`seller_id` are plain (undeclared-FK) agent-id columns.

    They used to declare `ForeignKey("agents.id")`, which stopped being
    well-formed SQL once `agents` became `(run_id, id)`-keyed (`id` alone is
    no longer unique, and a single-column reference to a non-unique parent
    column is invalid -- SQLite tolerates the DDL but reports "foreign key
    mismatch" the moment `PRAGMA foreign_keys=ON`, and Postgres refuses to
    create the table at all). This table carries no run-scope column of its
    own, so the composite `(run_id, agent_id)` reference the other
    agent-referencing tables now use is not available here; declaring
    nothing is strictly better than declaring something invalid. Nothing is
    lost at runtime -- FK enforcement is off by default in SQLite and this
    codebase never enables it, and no `relationship()` anywhere relies on
    this metadata.
    """

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    buyer_id: Mapped[str] = mapped_column(String)
    seller_id: Mapped[str] = mapped_column(String)
    good_name: Mapped[str] = mapped_column(String)
    currency_symbol: Mapped[str] = mapped_column(String)
    chain_name: Mapped[str] = mapped_column(String)
    gas_fee: Mapped[float] = mapped_column(Float)
    expected_value: Mapped[float] = mapped_column(Float)
    paid_value: Mapped[float] = mapped_column(Float)
    timestep: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String)
    fx_tax_paid: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime)


class NegotiationRecord(Base):
    __tablename__ = "negotiations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str | None] = mapped_column(String, ForeignKey("transactions.id"), nullable=True)
    rounds: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str] = mapped_column(String)
    log: Mapped[list] = mapped_column(JSON)


class HallucinationRecord(Base):
    """Every hallucination check, whether or not it ties to a settled
    transaction -- detection happens on any LLM decision, and a decision can
    be rejected/countered long before (or without ever) settling."""

    __tablename__ = "hallucinations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str | None] = mapped_column(String, nullable=True)
    transaction_id: Mapped[str | None] = mapped_column(String, ForeignKey("transactions.id"), nullable=True)
    expected_price: Mapped[float] = mapped_column(Float)
    paid_price: Mapped[float] = mapped_column(Float)
    overpayment_pct: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String)
    is_hallucination: Mapped[bool] = mapped_column(Boolean)
    currency_symbol: Mapped[str] = mapped_column(String)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)


class ScenarioRecord(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    config_snapshot: Mapped[dict] = mapped_column(JSON)
    run_at: Mapped[datetime] = mapped_column(DateTime)


class MetricRecord(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("scenarios.id"))
    metric_name: Mapped[str] = mapped_column(String)
    timestep: Mapped[int] = mapped_column(Integer)
    value: Mapped[float] = mapped_column(Float)


class LLMDecisionRecord(Base):
    """Every LLM decision, whether it came from the primary model, a fallback,
    or a same-model economic-correction reprompt (see fallback_used /
    fallback_reason / model_attempts).

    `simulation_id` IS this row's run_id (`database/repository.py`'s
    `_llm_decision_log_entry` sets `simulation_id=run_id`), so it pairs with
    `agent_id` to form the composite reference into the now
    `(run_id, id)`-keyed `agents` table -- the old single-column
    `ForeignKey("agents.id")` on `agent_id` is no longer well-formed (see
    `TransactionRecord`'s docstring)."""

    __tablename__ = "llm_decisions"
    __table_args__ = (ForeignKeyConstraint(["simulation_id", "agent_id"], ["agents.run_id", "agents.id"]),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(String)
    simulation_id: Mapped[str] = mapped_column(String)
    timestep: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[str] = mapped_column(String)
    agent_type: Mapped[str] = mapped_column(String)
    requested_model: Mapped[str] = mapped_column(String)
    actual_model: Mapped[str] = mapped_column(String)
    fallback_used: Mapped[bool] = mapped_column(Boolean)
    fallback_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    model_attempts: Mapped[list] = mapped_column(JSON)
    prompt_version: Mapped[str] = mapped_column(String)
    rendered_prompt_hash: Mapped[str] = mapped_column(String)
    system_prompt: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    currency: Mapped[str] = mapped_column(String)
    chain: Mapped[str] = mapped_column(String)
    amount: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    reported_reasoning: Mapped[str] = mapped_column(String)
    negotiation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    round: Mapped[int] = mapped_column(Integer)
    risk_profile: Mapped[str] = mapped_column(String)
    utility_type: Mapped[str] = mapped_column(String)
    utility_parameters: Mapped[dict] = mapped_column(JSON)
    scenario: Mapped[str] = mapped_column(String)
    domestic_or_cross_border: Mapped[str] = mapped_column(String)
    governance_prompt_enabled: Mapped[bool] = mapped_column(Boolean)
    spread_optimal_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    spread_optimal_chain: Mapped[str | None] = mapped_column(String, nullable=True)
    gas_optimal_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    gas_optimal_chain: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)


class MarketSnapshotRecord(Base):
    """A timestamped external market-data fetch (Polygon live price, or the
    static profile corpus's report_date) shown to an LLM -- persisted so a
    later re-run can see exactly what data the model was shown."""

    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    retrieval_timestamp: Mapped[datetime] = mapped_column(DateTime)
    source: Mapped[str] = mapped_column(String)
    ticker: Mapped[str] = mapped_column(String)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_window: Mapped[str | None] = mapped_column(String, nullable=True)
    negotiation_id: Mapped[str | None] = mapped_column(String, nullable=True)


class SimulationRunRecord(Base):
    """Provenance metadata captured once per run, before the first timestep.

    model_roster_summary is a short descriptor of the run's agent-to-model
    assignment (Phase 3 assigns one model per agent, not one per run) rather
    than a single openrouter_model_id -- see docs/superpowers/plans/
    2026-07-29-phase3-01-foundation-persistence.md Task 5 for why this
    deviates from Experiment.md's singular field name.
    """

    __tablename__ = "simulation_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    scenario_name: Mapped[str] = mapped_column(String)
    research_mode: Mapped[str] = mapped_column(String)
    random_seed: Mapped[int] = mapped_column(Integer)
    model_roster_summary: Mapped[str] = mapped_column(String)
    prompt_version_hash: Mapped[str] = mapped_column(String)
    git_commit_hash: Mapped[str] = mapped_column(String)
    config_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class InterventionLogRecord(Base):
    __tablename__ = "intervention_logs"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"))
    timestep: Mapped[int] = mapped_column(Integer)
    shock_type: Mapped[str] = mapped_column(String)
    target_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    target_issuer: Mapped[str | None] = mapped_column(String, nullable=True)
    magnitude: Mapped[float] = mapped_column(Float)


class TimestepLogRecord(Base):
    __tablename__ = "timestep_logs"

    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"), primary_key=True)
    timestep: Mapped[int] = mapped_column(Integer, primary_key=True)
    inflation_rate: Mapped[float] = mapped_column(Float)
    confidence_index: Mapped[float] = mapped_column(Float)
    eth_gas_fee_gwei: Mapped[float] = mapped_column(Float)
    solana_gas_fee_usd: Mapped[float] = mapped_column(Float)
    eur_usd_exchange_rate: Mapped[float] = mapped_column(Float)


class AgentStateRecord(Base):
    """Per-agent-per-day snapshot. wallet_balances is a JSON dict keyed by
    currency symbol rather than Experiment.md's fixed usd_balance/
    eur_balance/gold_balance columns -- this codebase's currency universe
    has nine currencies, not three, and a fixed schema would silently drop
    six of them. See docs/superpowers/plans/
    2026-07-29-phase3-01-foundation-persistence.md Task 8."""

    __tablename__ = "agent_states"
    __table_args__ = (ForeignKeyConstraint(["run_id", "agent_id"], ["agents.run_id", "agents.id"]),)

    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"), primary_key=True)
    timestep: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    risk_profile: Mapped[str] = mapped_column(String)
    cara_coefficient: Mapped[float | None] = mapped_column(Float, nullable=True)
    real_purchasing_power: Mapped[float] = mapped_column(Float)
    wallet_balances: Mapped[dict] = mapped_column(JSON)
    utility_score: Mapped[float] = mapped_column(Float)


class AgentMemoryLogRecord(Base):
    __tablename__ = "agent_memory_logs"
    __table_args__ = (ForeignKeyConstraint(["run_id", "agent_id"], ["agents.run_id", "agents.id"]),)

    memory_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"))
    timestep: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[str] = mapped_column(String)
    memory_type: Mapped[str] = mapped_column(String)
    memory_text: Mapped[str] = mapped_column(String)
