"""Defines database tables: agents, wallets, transactions, negotiations,
hallucinations, scenarios, metrics.

Source of truth for the schema -- database/schema.sql was removed in favor
of Base.metadata.create_all(), so there's exactly one place the schema is
defined (Alembic-style migrations are more machinery than this research
tool needs).
"""

from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON, DateTime


class Base(DeclarativeBase):
    pass


class AgentRecord(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_class: Mapped[str] = mapped_column(String)
    profile_name: Mapped[str] = mapped_column(String)
    risk_profile: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class WalletRecord(Base):
    __tablename__ = "wallets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"))
    currency_symbol: Mapped[str] = mapped_column(String)
    balance: Mapped[float] = mapped_column(Float)


class TransactionRecord(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    buyer_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"))
    seller_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"))
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
    fallback_reason / model_attempts)."""

    __tablename__ = "llm_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(String)
    simulation_id: Mapped[str] = mapped_column(String)
    timestep: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"))
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
