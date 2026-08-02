from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, LLMDecisionRecord, MarketSnapshotRecord, TransactionRecord
from database.repository import (
    LLMDecisionLogEntry,
    LLMDecisionRepository,
    MarketSnapshotLogEntry,
    MarketSnapshotRepository,
    TransactionRepository,
)
from src.transactions.transaction import Transaction


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_llm_decision_repository_persists_full_record():
    session = _session()
    repo = LLMDecisionRepository(session)
    entry = LLMDecisionLogEntry(
        decision_id="dec-1",
        simulation_id="sim-1",
        timestep=3,
        agent_id="buyer-1",
        agent_type="buyer",
        requested_model="claude-sonnet-5",
        actual_model="claude-sonnet-5",
        fallback_used=False,
        fallback_reason=None,
        model_attempts=["claude-sonnet-5"],
        prompt_version="buyer_prompt@v1",
        rendered_prompt_hash="abc123",
        system_prompt="You are a buyer agent. Candidates: USDC on ethereum...",
        action="OFFER",
        currency="USDC",
        chain="ethereum",
        amount=100.0,
        price=99.5,
        reported_reasoning="USDC offers the best governance/liquidity trade-off.",
        negotiation_id="neg-1",
        round=1,
        risk_profile="low",
        utility_type="crra",
        utility_parameters={"risk_aversion": 3.0},
        scenario="baseline",
        domestic_or_cross_border="domestic",
        governance_prompt_enabled=True,
        spread_optimal_currency="USDT",
        spread_optimal_chain="ethereum",
        gas_optimal_currency="USDC",
        gas_optimal_chain="ethereum",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(LLMDecisionRecord).all()
    assert len(rows) == 1
    assert rows[0].decision_id == "dec-1"
    assert rows[0].actual_model == "claude-sonnet-5"
    assert rows[0].model_attempts == ["claude-sonnet-5"]
    assert rows[0].fallback_used is False
    assert rows[0].system_prompt == "You are a buyer agent. Candidates: USDC on ethereum..."


def test_market_snapshot_repository_persists_and_allows_missing_price():
    session = _session()
    repo = MarketSnapshotRepository(session)
    entry = MarketSnapshotLogEntry(source="polygon", ticker="X:USDCUSD", price=None, data_window="live", negotiation_id="neg-1")

    repo.record(entry)
    session.commit()

    rows = session.query(MarketSnapshotRecord).all()
    assert len(rows) == 1
    assert rows[0].price is None
    assert rows[0].ticker == "X:USDCUSD"


def test_transaction_repository_persists_fx_tax_paid():
    session = _session()
    repo = TransactionRepository(session)
    tx = Transaction(
        buyer_id="buyer-1",
        seller_id="seller-1",
        good_name="cloud_compute",
        currency_symbol="USDC",
        chain_name="ethereum",
        gas_fee=0.5,
        expected_value=100.0,
        paid_value=100.0,
        timestep=0,
        fx_tax_paid=1.75,
    )

    repo.record(tx)
    session.commit()

    rows = session.query(TransactionRecord).all()
    assert len(rows) == 1
    assert rows[0].fx_tax_paid == 1.75
