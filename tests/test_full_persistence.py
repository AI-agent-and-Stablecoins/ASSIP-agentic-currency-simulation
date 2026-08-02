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
from database.repository import _llm_decision_log_entry, persist_full_timestep
from src.agents.agent_factory import build_agent, load_agent_profiles
from src.economy.shocks import ShockEvent, ShockType
from src.llm.agent_reasoning import hash_rendered_prompt
from src.simulation.environment import Environment
from src.simulation.event_queue import EventQueue
from src.simulation.timestep import LLMDecisionRecord as TimestepLLMDecisionRecord
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


def test_llm_decision_log_entry_hashes_rendered_prompt_not_reasoning():
    """Fix 1 (Critical, Task 11 review), direct unit test of the buggy
    function itself: `_llm_decision_log_entry` used to compute
    `rendered_prompt_hash=hash_rendered_prompt(decision.reasoning or "")`
    -- hashing the model's OUTPUT, not the rendered PROMPT it was given.
    Constructs a decision whose `reasoning` and `rendered_prompt` are
    deliberately different strings and confirms the persisted hash tracks
    `rendered_prompt`, never `reasoning`.
    """
    decision = TimestepLLMDecisionRecord(
        agent_id="buyer-1",
        agent_type="buyer",
        risk_profile="low",
        utility_type="cara",
        requested_model="test-vendor/buyer-model",
        actual_model="test-vendor/buyer-model",
        success=True,
        currency_symbol="USDC",
        chain_name="ethereum",
        amount=1.0,
        price=90.0,
        reasoning="USDC offers the best governance/liquidity trade-off.",
        rendered_prompt="# System\nYou are a buyer agent...\nCandidates: USDC on ethereum...",
    )

    entry = _llm_decision_log_entry(decision, "dec-1", "run-1", 0, agent=None, scenario_name="baseline")

    assert entry.rendered_prompt_hash == hash_rendered_prompt(decision.rendered_prompt)
    assert entry.rendered_prompt_hash != hash_rendered_prompt(decision.reasoning)
    assert entry.system_prompt == decision.rendered_prompt


def test_llm_decision_log_entry_carries_spread_and_gas_optimal_fields():
    decision = TimestepLLMDecisionRecord(
        agent_id="a1",
        agent_type="consumer",
        risk_profile="medium",
        utility_type="cara",
        requested_model="vendor/model",
        actual_model="vendor/model",
        success=True,
        action="ACCEPT",
        currency_symbol="USDC",
        chain_name="solana",
        amount=1.0,
        price=100.0,
        spread_optimal_currency="USDT",
        spread_optimal_chain="ethereum",
        gas_optimal_currency="USDC",
        gas_optimal_chain="solana",
    )

    entry = _llm_decision_log_entry(decision, "dec-1", "run-1", 0, agent=None, scenario_name="master_simulation")

    assert entry.spread_optimal_currency == "USDT"
    assert entry.spread_optimal_chain == "ethereum"
    assert entry.gas_optimal_currency == "USDC"
    assert entry.gas_optimal_chain == "solana"


def test_persist_full_timestep_persists_rendered_prompt_hash_derived_from_the_prompt_not_reasoning():
    """Fix 1 (Critical, Task 11 review), end-to-end: runs a real use_llm=True
    day (fixed `reasoning="test reasoning"` for every decision, per
    `_decision_json`), persists it, and confirms every persisted
    `LLMDecisionRecord.rendered_prompt_hash` matches
    `hash_rendered_prompt` of that decision's actual `rendered_prompt` --
    and that none of them match the old, wrong `hash_rendered_prompt
    ("test reasoning")` value.
    """
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
    assert all(d.rendered_prompt for d in result.llm_decisions)

    session = _session()
    persist_full_timestep(session, env, result, run_id="run-hash")

    decision_rows = session.query(LLMDecisionRecord).all()
    expected_hashes = {hash_rendered_prompt(d.rendered_prompt) for d in result.llm_decisions}
    persisted_hashes = {row.rendered_prompt_hash for row in decision_rows}
    assert persisted_hashes == expected_hashes
    assert all(row.system_prompt for row in decision_rows)

    wrong_hash = hash_rendered_prompt("test reasoning")
    assert wrong_hash not in persisted_hashes


def test_persist_full_timestep_is_atomic_a_failure_partway_through_leaves_nothing_committed(monkeypatch):
    """Fix 2 (Important, Task 11 review): persist_full_timestep used to call
    persist_timestep (which committed on its own) and then commit again at
    the end -- two separate transactions for one simulated day. A failure
    partway through the second phase left agent/transaction/negotiation
    rows durable while the rest of the day's rows never made it in. Now
    persist_timestep is called with commit=False and there is exactly one
    session.commit() at the very end of persist_full_timestep, so an
    exception anywhere in between must roll back EVERYTHING -- including
    the agent/transaction rows persist_timestep's own logic adds, which
    used to be safe from exactly this kind of rollback.
    """
    from database.repository import AgentStateRepository

    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)
    result = run_timestep(env, day=0, rng=rng)
    assert env.agents  # sanity: AgentRepository has something to write

    session = _session()

    def _boom(self, entry):
        raise RuntimeError("simulated failure mid-persistence")

    monkeypatch.setattr(AgentStateRepository, "record", _boom)

    with pytest.raises(RuntimeError):
        persist_full_timestep(session, env, result, run_id="run-atomic")

    session.rollback()

    assert session.query(AgentRecord).count() == 0
    assert session.query(TransactionRecord).count() == 0
    assert session.query(TimestepLogRecord).count() == 0
    assert session.query(AgentStateRecord).count() == 0


def test_persist_full_timestep_gas_fee_columns_read_chain_config_gas_fee_directly():
    """Fix 3/4 (Important, Task 11 review): the dead `if "ethereum"/"solana"
    in env.chains` guards were removed (env.chains is always the full chain
    universe -- both Environment.build and Environment.build_from_population
    call load_chain_universe() unconditionally). This pins the current,
    intentional values: eth_gas_fee_gwei/solana_gas_fee_usd both read
    ChainConfig.gas_fee directly, which is USD-denominated everywhere else
    in the codebase (a pre-existing eth_gas_fee_gwei naming mismatch this
    fix pass documents, but does not rename).
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    rng = random.Random(0)
    result = run_timestep(env, day=0, rng=rng)

    session = _session()
    persist_full_timestep(session, env, result, run_id="run-gas")

    row = session.query(TimestepLogRecord).filter_by(run_id="run-gas", timestep=0).one()
    assert row.eth_gas_fee_gwei == env.chains["ethereum"].gas_fee
    assert row.solana_gas_fee_usd == env.chains["solana"].gas_fee
