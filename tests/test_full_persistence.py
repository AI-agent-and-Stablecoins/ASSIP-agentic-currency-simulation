"""Task 11: persist_full_timestep wires every per-day/per-agent/per-decision
table together into one persistence call per simulated day.

Also exercises the CARA-adaptation wiring gap this task closes: Task 7's
`adapt_cara_coefficient` was built as a standalone function, never called
from `run_timestep` or any day-loop driver. `persist_full_timestep` is the
first place that has a natural reason to compare a CARA-eligible agent's
real purchasing power day-over-day, so it drives that comparison itself
(see `database/repository.py`'s `persist_full_timestep` for the mechanism:
`Environment.previous_real_purchasing_power`).
"""

import random

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import (
    AgentMemoryLogRecord,
    AgentRecord,
    AgentStateRecord,
    Base,
    HallucinationRecord,
    InterventionLogRecord,
    LLMDecisionRecord,
    TimestepLogRecord,
    TransactionRecord,
)
from database.repository import persist_full_timestep
from src.agents.agent_factory import build_agent, load_agent_profiles
from src.economy.shocks import ShockEvent, ShockType
from src.simulation.environment import Environment
from src.simulation.event_queue import EventQueue
from src.simulation.timestep import TimestepResult, run_timestep
from tests.llm_test_helpers import mock_openrouter_client


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _build_env_with_shocks(shocks: list[ShockEvent], agent_mix: dict[str, int]) -> Environment:
    env = Environment.build("baseline", agent_mix)
    env.event_queue = EventQueue(shocks)
    return env


def _decision_json(action: str = "OFFER", currency: str = "USDC", chain: str = "ethereum", price: float = 90.0) -> dict:
    return {
        "action": action,
        "proposed_currency": currency,
        "proposed_chain": chain,
        "amount": 1.0,
        "price": price,
        "reasoning": "test reasoning",
    }


def test_persist_full_timestep_writes_agent_and_transaction_and_timestep_rows_on_a_quiet_day():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)
    result = run_timestep(env, day=0, rng=rng)

    session = _session()
    persist_full_timestep(session, env, result, run_id="run-quiet")

    assert session.query(AgentRecord).count() == 4
    assert session.query(TransactionRecord).count() == len(result.transactions)

    timestep_rows = session.query(TimestepLogRecord).all()
    assert len(timestep_rows) == 1
    assert timestep_rows[0].run_id == "run-quiet"
    assert timestep_rows[0].timestep == 0

    agent_state_rows = session.query(AgentStateRecord).filter_by(run_id="run-quiet", timestep=0).all()
    assert len(agent_state_rows) == 4
    for row in agent_state_rows:
        assert row.wallet_balances
        assert isinstance(row.utility_score, float)

    # baseline.yaml has no shocks, so a quiet day produces neither
    # intervention nor memory rows.
    assert session.query(InterventionLogRecord).count() == 0
    assert session.query(AgentMemoryLogRecord).count() == 0


def test_persist_full_timestep_records_intervention_and_memory_rows_on_a_shock_day():
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")],
        {"consumer": 2, "merchant": 2},
    )
    consumer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    consumer.wallet.balances["USDT"] = 100.0
    rng = random.Random(0)
    result = run_timestep(env, day=0, rng=rng)
    assert result.fired_shocks
    assert result.memory_events

    session = _session()
    persist_full_timestep(session, env, result, run_id="run-shock")

    intervention_rows = session.query(InterventionLogRecord).all()
    assert len(intervention_rows) == len(result.fired_shocks)
    assert intervention_rows[0].shock_type == "depeg_event"
    assert intervention_rows[0].target_currency == "USDT"

    memory_rows = session.query(AgentMemoryLogRecord).all()
    assert len(memory_rows) == len(result.memory_events)
    assert memory_rows[0].memory_type == "Depeg"


def test_persist_full_timestep_adapts_cara_coefficient_after_a_realized_loss_across_two_days():
    """The core CARA-adaptation wiring test: nothing in production ever
    called Task 7's adapt_cara_coefficient before this task. Across two
    persist_full_timestep calls on the same Environment, with a realized
    real-purchasing-power loss in between, the persisted
    AgentStateRecord.cara_coefficient must actually differ (increase) day
    over day -- not just "the function ran without error".
    """
    profiles = load_agent_profiles()
    agent = build_agent(profiles["consumer"], cara_override=("cara", 2.0))
    agent.wallet.balances = {"USDC": 1000.0}
    env = Environment.build_from_population("baseline", [agent])

    session = _session()

    persist_full_timestep(session, env, TimestepResult(day=0), run_id="run-cara")
    day0_row = session.query(AgentStateRecord).filter_by(
        run_id="run-cara", timestep=0, agent_id=agent.agent_id
    ).one()
    # First-ever call: no prior day to compare against, so no adaptation.
    assert day0_row.cara_coefficient == pytest.approx(2.0)
    assert agent.cara_coefficient == pytest.approx(2.0)

    # Simulate a realized loss in real purchasing power between day 0 and day 1.
    agent.wallet.balances["USDC"] = 500.0

    persist_full_timestep(session, env, TimestepResult(day=1), run_id="run-cara")
    day1_row = session.query(AgentStateRecord).filter_by(
        run_id="run-cara", timestep=1, agent_id=agent.agent_id
    ).one()

    assert day1_row.cara_coefficient > day0_row.cara_coefficient
    assert agent.cara_coefficient == pytest.approx(day1_row.cara_coefficient)


def test_persist_full_timestep_does_not_adapt_cara_coefficient_for_multi_attribute_agents():
    profiles = load_agent_profiles()
    agent = build_agent(profiles["merchant"])  # multi_attribute, cara_coefficient is None
    env = Environment.build_from_population("baseline", [agent])

    session = _session()
    persist_full_timestep(session, env, TimestepResult(day=0), run_id="run-ma")
    agent.wallet.balances["USDC"] = agent.wallet.balances.get("USDC", 0.0) / 2
    persist_full_timestep(session, env, TimestepResult(day=1), run_id="run-ma")

    rows = session.query(AgentStateRecord).filter_by(run_id="run-ma", agent_id=agent.agent_id).all()
    assert len(rows) == 2
    assert all(row.cara_coefficient is None for row in rows)


def test_persist_full_timestep_records_llm_decisions_and_hallucinations():
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    for a in env.agents.values():
        a.assigned_model = "test-vendor/buyer-model" if a.agent_class == "buyer" else "test-vendor/seller-model"

    client = mock_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", price=90.0),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", price=90.0),
        }
    )
    rng = random.Random(0)
    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)
    assert result.llm_decisions

    session = _session()
    persist_full_timestep(session, env, result, run_id="run-llm")

    decision_rows = session.query(LLMDecisionRecord).all()
    assert len(decision_rows) == len(result.llm_decisions)
    assert {row.agent_id for row in decision_rows} <= set(env.agents.keys())
    assert all(row.simulation_id == "run-llm" for row in decision_rows)

    expected_hallucination_count = sum(1 for d in result.llm_decisions if d.hallucination is not None)
    hallucination_rows = session.query(HallucinationRecord).all()
    assert len(hallucination_rows) == expected_hallucination_count
    assert expected_hallucination_count > 0
