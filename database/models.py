"""Defines database tables: agents, wallets, transactions, negotiations,
hallucinations, scenarios, metrics.

Source of truth for the schema -- database/schema.sql was removed in favor
of Base.metadata.create_all(), so there's exactly one place the schema is
defined (Alembic-style migrations are more machinery than this research
tool needs).
"""

from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, String
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
    timestamp: Mapped[datetime] = mapped_column(DateTime)


class NegotiationRecord(Base):
    __tablename__ = "negotiations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str | None] = mapped_column(String, ForeignKey("transactions.id"), nullable=True)
    rounds: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str] = mapped_column(String)
    log: Mapped[list] = mapped_column(JSON)


class HallucinationRecord(Base):
    """Inert in Phase 1 -- populated once Phase 2's LLM decisions produce
    expected-vs-paid comparisons. Defined now so no migration is needed later."""

    __tablename__ = "hallucinations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String, ForeignKey("transactions.id"))
    expected_price: Mapped[float] = mapped_column(Float)
    paid_price: Mapped[float] = mapped_column(Float)
    overpayment_pct: Mapped[float] = mapped_column(Float)
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
