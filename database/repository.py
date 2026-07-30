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
    HallucinationRecord,
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
from src.negotiation.conversation_history import ConversationLog
from src.simulation.environment import Environment
from src.simulation.timestep import TimestepResult
from src.transactions.transaction import Transaction


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
    crra_sigma: float
    real_purchasing_power: float
    wallet_balances: dict[str, float]
    utility_score: float


class AgentMemoryLogEntry(BaseModel):
    run_id: str
    timestep: int
    agent_id: str
    memory_type: str
    memory_text: str


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

    def upsert_agent(self, agent: BaseAgent) -> None:
        record = self.session.get(AgentRecord, agent.agent_id)
        if record is None:
            record = AgentRecord(
                id=agent.agent_id,
                agent_class=agent.agent_class,
                profile_name=agent.profile_name,
                risk_profile=agent.risk_profile,
                created_at=datetime.now(timezone.utc),
            )
            self.session.add(record)
        self._sync_wallet(agent)

    def _sync_wallet(self, agent: BaseAgent) -> None:
        self.session.query(WalletRecord).filter(WalletRecord.agent_id == agent.agent_id).delete()
        for symbol, balance in agent.wallet.balances.items():
            self.session.add(WalletRecord(agent_id=agent.agent_id, currency_symbol=symbol, balance=balance))


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


def persist_timestep(session: Session, env: Environment, result: TimestepResult) -> None:
    agent_repo = AgentRepository(session)
    tx_repo = TransactionRepository(session)
    negotiation_repo = NegotiationRepository(session)

    for agent in env.agents.values():
        agent_repo.upsert_agent(agent)
    for tx in result.transactions:
        tx_repo.record(tx)
    for log in result.negotiations:
        negotiation_repo.record(log)
    session.commit()
