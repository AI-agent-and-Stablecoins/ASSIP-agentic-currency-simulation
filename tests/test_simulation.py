import ast
import importlib
import random
import sys

import httpx
import pytest

from src.blockchain.routing_engine import CurrencyChainOption, generate_candidates
from src.economy.macro_state import MacroState
from src.economy.shocks import ShockEvent, ShockType
from src.economy.trust import TrustLedger
from src.llm.agent_reasoning import AgentDecisionContext, AgentUtilityContext, TransactionContext, build_decision_context
from src.llm.decision_adapter import NegotiationAction
from src.llm.decision_schema import DecisionAction
from src.llm.llm_router import OPENROUTER_BASE_URL
from src.market.pricing_engine import true_price
from src.negotiation.llm_negotiation_engine import NegotiationSession, NegotiationStatus, run_llm_negotiation
from src.simulation.environment import Environment
from src.simulation.event_queue import EventQueue
from src.simulation.simulation_runner import SimulationConfig, SimulationRunner
from src.simulation.timestep import LLMDecisionRecord, _make_llm_decide_closure, decide_single_model, run_timestep
from src.transactions.transaction import TransactionStatus
from src.utils.constants import REPO_ROOT
from tests.llm_test_helpers import mock_openrouter_client


def _build_env_with_shocks(shocks: list[ShockEvent], agent_mix: dict[str, int]) -> Environment:
    """Build an Environment with custom shocks by reusing Environment.build
    and reassigning its event_queue with the given shocks.
    """
    env = Environment.build("baseline", agent_mix)
    env.event_queue = EventQueue(shocks)
    return env


def test_simulation_runs_end_to_end_without_errors():
    config = SimulationConfig(
        agent_mix={"consumer": 3, "merchant": 2},
        num_days=5,
        scenario="baseline",
        random_seed=42,
    )

    result = SimulationRunner().run(config)

    assert len(result.timesteps) == 5
    all_transactions = [tx for step in result.timesteps for tx in step.transactions]
    assert len(all_transactions) > 0
    settled = [tx for tx in all_transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0


def test_simulation_conserves_total_currency_balances():
    env = Environment.build("baseline", {"consumer": 3, "merchant": 2})

    def total_balances() -> dict[str, float]:
        totals: dict[str, float] = {}
        for agent in env.agents.values():
            for symbol, amount in agent.wallet.balances.items():
                totals[symbol] = totals.get(symbol, 0.0) + amount
        return totals

    before = total_balances()

    rng = random.Random(42)
    for day in range(5):
        run_timestep(env, day, rng)

    after = total_balances()

    for symbol, amount in before.items():
        assert after.get(symbol, 0.0) == pytest.approx(amount, rel=1e-6)


def test_run_timestep_reports_fired_shocks_on_the_day_they_fire():
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.INFLATION, magnitude=0.02)],
        {"consumer": 2, "merchant": 2},
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    assert len(result.fired_shocks) == 1
    assert result.fired_shocks[0].type == ShockType.INFLATION


def test_run_timestep_reports_no_fired_shocks_on_a_quiet_day():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    assert result.fired_shocks == []


def test_run_timestep_records_narrative_memory_for_agents_holding_a_shocked_currency():
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")],
        {"consumer": 2, "merchant": 2},
    )
    consumer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    consumer.wallet.balances["USDT"] = 100.0
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    matching = [e for e in result.memory_events if e[0] == consumer.agent_id]
    assert len(matching) == 1
    assert matching[0][1] == "Depeg"
    assert "USDT" in matching[0][2]
    assert matching[0][2] in consumer.memory.narrative_events


def test_environment_starts_with_price_index_of_one():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    assert env.price_index == 1.0


def test_environment_build_constructs_a_trust_ledger():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})

    assert isinstance(env.trust_ledger, TrustLedger)
    # Sanity: it should be initialized from the same currencies the
    # environment holds, not some detached default instance.
    for symbol in env.currencies:
        assert env.trust_ledger.trust_score(symbol) == pytest.approx(env.currencies[symbol].governance_score)


def test_run_timestep_advances_trust_ledger_so_depeg_offset_decays_across_days():
    """The review's core finding: TrustLedger must actually be driven by the
    real simulation loop (run_timestep), not just be constructible/callable
    in isolation. A depeg_event on day 0 should spike USDT's peg_error
    offset via env.trust_ledger; subsequent quiet days should decay that
    offset -- and generate_candidates() called through the production path
    (i.e. via env.trust_ledger) must reflect a shrinking, non-static
    peg_error day over day.
    """
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")],
        {"consumer": 2, "merchant": 2},
    )
    rng = random.Random(0)

    run_timestep(env, day=0, rng=rng)
    day0_offset = env.trust_ledger.peg_error_offset("USDT")
    day0_effective = env.trust_ledger.effective_peg_error("USDT", env.currencies["USDT"].peg_error)
    assert day0_offset == pytest.approx(0.08)

    run_timestep(env, day=1, rng=rng)
    day1_offset = env.trust_ledger.peg_error_offset("USDT")
    day1_effective = env.trust_ledger.effective_peg_error("USDT", env.currencies["USDT"].peg_error)

    # The production call site of generate_candidates (inside run_timestep)
    # is wired to the same ledger: re-deriving candidates directly with
    # env.currencies/env.chains/env.trust_ledger, at this point in time,
    # reproduces the same shrinking peg_error the loop itself is using.
    candidates_day1 = generate_candidates(
        {"USDT": 1.0}, env.currencies, env.chains, env.liquidity_pools, trust_ledger=env.trust_ledger
    )
    usdt_candidate = next(c for c in candidates_day1 if c.currency_symbol == "USDT")
    assert usdt_candidate.peg_error == pytest.approx(day1_effective)

    run_timestep(env, day=2, rng=rng)
    day2_offset = env.trust_ledger.peg_error_offset("USDT")

    # Present but shrinking -- proves the ledger is being genuinely advanced
    # once per day by run_timestep, not left static/dead.
    assert 0.0 < day1_offset < day0_offset
    assert 0.0 < day2_offset < day1_offset
    assert day1_effective < day0_effective


# ---------------------------------------------------------------------------
# use_llm=True: LLM-driven decisions + full LLM-vs-LLM negotiation
# ---------------------------------------------------------------------------


def _decision_json(action: str = "OFFER", currency: str = "USDC", chain: str = "ethereum", price: float = 90.0) -> dict:
    return {
        "action": action,
        "proposed_currency": currency,
        "proposed_chain": chain,
        "amount": 1.0,
        "price": price,
        "reasoning": "test reasoning",
    }


def _base_decision_context(wallet_balances: dict[str, float] | None = None) -> AgentDecisionContext:
    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances=wallet_balances or {"USDC": 1000.0},
    )
    candidates = [
        CurrencyChainOption(
            currency_symbol="USDC",
            chain_name="ethereum",
            governance_score=0.95,
            liquidity_score=0.97,
            peg_error=0.0003,
            gas_fee=2.5,
            finality_seconds=12.0,
            genius_compliant=True,
        )
    ]
    macro = MacroState()
    return build_decision_context(agent_context, candidates, {}, macro, macro, TransactionContext(is_cross_border=False))


def test_decide_single_model_returns_action_on_first_try():
    client = mock_openrouter_client({"test-vendor/model": _decision_json()})

    action = decide_single_model(
        "buyer", _base_decision_context(), "test-vendor/model", client, {"USDC"}, {"ethereum"}
    )

    assert isinstance(action, NegotiationAction)
    assert action.currency_symbol == "USDC"
    assert action.action == DecisionAction.OFFER


def test_decide_single_model_corrects_an_economically_invalid_decision_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        calls["count"] += 1
        body = _json.loads(request.content)
        assert body["model"] == "test-vendor/model"
        if calls["count"] == 1:
            content = _json.dumps(_decision_json(currency="NOTACOIN"))
        else:
            content = _json.dumps(_decision_json(currency="USDC"))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    telemetry: dict = {}

    action = decide_single_model(
        "buyer",
        _base_decision_context(),
        "test-vendor/model",
        client,
        {"USDC"},
        {"ethereum"},
        telemetry=telemetry,
    )

    assert action is not None
    assert action.currency_symbol == "USDC"
    assert calls["count"] == 2
    assert telemetry["correction_attempts"] == 1


def test_decide_single_model_returns_none_when_still_invalid_after_one_correction():
    client = mock_openrouter_client({"test-vendor/model": _decision_json(currency="NOTACOIN")})
    telemetry: dict = {}

    action = decide_single_model(
        "buyer",
        _base_decision_context(),
        "test-vendor/model",
        client,
        {"USDC"},
        {"ethereum"},
        telemetry=telemetry,
    )

    assert action is None
    assert telemetry["correction_attempts"] == 1
    assert telemetry["failure_reason"] is not None


def test_decide_single_model_returns_none_when_the_model_call_itself_fails():
    client = mock_openrouter_client({})  # no mocked models -> every call 404s
    telemetry: dict = {}

    action = decide_single_model(
        "buyer",
        _base_decision_context(),
        "test-vendor/unmocked-model",
        client,
        {"USDC"},
        {"ethereum"},
        telemetry=telemetry,
    )

    assert action is None
    assert telemetry["failure_reason"] is not None


def test_run_timestep_with_use_llm_true_requires_a_client():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    with pytest.raises(ValueError):
        run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=None)


def _assign_models(env: Environment, buyer_model: str, seller_model: str) -> None:
    for agent in env.agents.values():
        agent.assigned_model = buyer_model if agent.agent_class == "buyer" else seller_model


def test_run_timestep_with_use_llm_true_produces_llm_driven_transactions_with_ground_truth_expected_value():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")
    # Note: env.marketplace's Listing.true_price is actually populated with the
    # seller's marked-up asking price (see run_timestep's seller-listing loop:
    # `asking = seller.asking_price(price); post_listing(..., asking)`), not the
    # raw true_price(good) -- the existing rule-based path relies on this same
    # quirk (`expected_value=listing.true_price`), so this test mirrors it here
    # rather than "fixing" a naming quirk outside this task's scope.
    sellers = [a for a in env.agents.values() if a.agent_class == "seller"]
    asking_prices_by_good = {good.name: sellers[0].asking_price(true_price(good)) for good in env.goods}

    client = mock_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", price=90.0),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", price=90.0),
        }
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    assert isinstance(result.transactions, list)
    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0
    for tx in settled:
        # The critical Task 4 carry-forward: expected_value must be the real
        # ground-truth price (listing.true_price), NOT the negotiation's own
        # opening offer (which build_transaction_from_negotiation stubs it
        # to before this overwrite).
        assert tx.expected_value == pytest.approx(asking_prices_by_good[tx.good_name])
        assert tx.paid_value == pytest.approx(90.0)
        assert tx.currency_symbol == "USDC"
        assert tx.chain_name == "ethereum"

    assert len(result.llm_negotiations) > 0
    assert all(isinstance(session, NegotiationSession) for session in result.llm_negotiations)
    accepted_sessions = [s for s in result.llm_negotiations if s.status == NegotiationStatus.ACCEPTED]
    assert len(accepted_sessions) > 0

    assert len(result.llm_decisions) > 0
    assert all(isinstance(record, LLMDecisionRecord) for record in result.llm_decisions)
    assert all(record.success for record in result.llm_decisions)
    assert any(record.hallucination is not None for record in result.llm_decisions)


def test_run_timestep_with_use_llm_true_and_total_model_failure_skips_transactions_gracefully():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")
    client = mock_openrouter_client({})  # neither model is mocked -> every call fails
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    assert result.transactions == []
    assert len(result.llm_negotiations) > 0
    assert all(session.status == NegotiationStatus.WALKED_AWAY for session in result.llm_negotiations)
    assert len(result.llm_decisions) > 0
    assert all(not record.success for record in result.llm_decisions)
    assert all(record.failure_reason is not None for record in result.llm_decisions)
    assert all(record.hallucination is None for record in result.llm_decisions)


def test_run_timestep_with_use_llm_false_is_unchanged_from_before():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)  # use_llm defaults to False

    assert isinstance(result.transactions, list)
    assert result.llm_decisions == []
    assert result.llm_negotiations == []


# ---------------------------------------------------------------------------
# Task 5 review fixes
# ---------------------------------------------------------------------------


def test_timestep_module_has_no_top_level_httpx_or_llm_imports():
    """Fix 1 (Critical), static check: sandbox/sandbox_launcher.py's E2B
    provisioning installs only `pydantic sqlalchemy pyyaml python-dotenv
    pandas` (no httpx) and then imports simulation_runner -> timestep for a
    purely deterministic (use_llm=False) run. Any module-level `import
    httpx` / `from src.llm.agent_reasoning import ...` / `from
    src.llm.llm_router import ...` / `from src.llm.market_intelligence
    import ...` line in timestep.py would break that path even though it
    never touches the LLM code. This inspects the file's own module-level
    AST (not the live, possibly-already-cached sys.modules) so it can't be
    fooled by other test files having already imported httpx first.
    """
    source = (REPO_ROOT / "src" / "simulation" / "timestep.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_modules = {
        "httpx",
        "src.llm.agent_reasoning",
        "src.llm.llm_router",
        "src.llm.market_intelligence",
    }

    for node in tree.body:  # module level only -- deliberately not ast.walk
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, (
                    f"module-level `import {alias.name}` reintroduces the sandbox-breaking hard import"
                )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module not in forbidden_modules, (
                f"module-level `from {node.module} import ...` reintroduces the sandbox-breaking hard import"
            )


def test_run_timestep_use_llm_false_works_even_if_httpx_cannot_be_imported(monkeypatch):
    """Fix 1 (Critical), behavioral check: reproduce the actual sandbox
    failure mode by making a fresh `import httpx` raise, then confirming a
    freshly (re-)imported timestep module still works end-to-end for a
    deterministic run. `sys.modules["httpx"] = None` is the standard trick
    for forcing ImportError on `import httpx` (see Python docs on
    sys.modules); monkeypatch restores the original entries automatically.
    """
    monkeypatch.setitem(sys.modules, "httpx", None)
    monkeypatch.delitem(sys.modules, "src.simulation.timestep", raising=False)

    module = importlib.import_module("src.simulation.timestep")

    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)
    result = module.run_timestep(env, day=0, rng=rng)  # use_llm=False (default)

    assert isinstance(result.transactions, list)
    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0


def test_decide_single_model_accept_checks_payer_wallet_not_context_agents_own():
    """Fix 2 (Important): the funds check on ACCEPT must use the buyer's
    (payer's) wallet balances, never the wallet of whichever side happens to
    be making the decision. Here `context` belongs to a seller-like agent
    whose own wallet_balances hold none of the settlement currency (which is
    irrelevant -- the seller never pays); `payer_wallet_balances` is the
    buyer's, which does. The ACCEPT must succeed.
    """
    context = _base_decision_context(wallet_balances={"USDC": 0.0})
    client = mock_openrouter_client({"test-vendor/model": _decision_json(action="ACCEPT", price=90.0)})

    action = decide_single_model(
        "seller",
        context,
        "test-vendor/model",
        client,
        {"USDC"},
        {"ethereum"},
        payer_wallet_balances={"USDC": 1000.0},
    )

    assert action is not None
    assert action.action == DecisionAction.ACCEPT


def test_decide_single_model_accept_without_payer_override_falls_back_to_context_agent_wallet():
    """Companion to the above: confirm the default (no payer_wallet_balances
    given) still checks the context agent's own wallet -- i.e. the buyer-side
    call sites, which never pass an override other than their own balances,
    keep working exactly as before.
    """
    context = _base_decision_context(wallet_balances={"USDC": 0.0})
    client = mock_openrouter_client({"test-vendor/model": _decision_json(action="ACCEPT", price=90.0)})

    action = decide_single_model(
        "buyer",
        context,
        "test-vendor/model",
        client,
        {"USDC"},
        {"ethereum"},
    )

    assert action is None  # insufficient funds against context.agent.wallet_balances, no override given


def test_run_timestep_sellers_accept_is_not_rejected_for_lacking_the_settlement_currency():
    """Fix 2 (Important), end-to-end: give every seller a zero balance of the
    settlement currency the negotiation actually lands on, while the buyer
    holds plenty. Before the fix, the seller-side closure passed the
    seller's own (empty) wallet into adapt_decision's funds check, so the
    seller's ACCEPT would be spuriously invalidated and the negotiation
    would end in a synthetic WALK_AWAY instead of a settled transaction.
    """
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")
    for agent in env.agents.values():
        if agent.agent_class == "seller":
            agent.wallet.balances["USDC"] = 0.0

    client = mock_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", price=90.0),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", price=90.0),
        }
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0
    assert all(record.success for record in result.llm_decisions)
    accepted_sessions = [s for s in result.llm_negotiations if s.status == NegotiationStatus.ACCEPTED]
    assert len(accepted_sessions) > 0


def test_run_timestep_use_llm_true_raises_for_agent_missing_assigned_model():
    """Fix 3 (Important): Environment.build's count-based path never sets
    assigned_model, so with use_llm=True this must fail loudly and name the
    affected agent(s), rather than silently calling decide_single_model with
    model_id=None and producing a quiet zero-transaction run.
    """
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    client = mock_openrouter_client({})
    rng = random.Random(0)

    with pytest.raises(ValueError) as exc_info:
        run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    message = str(exc_info.value)
    assert "assigned_model" in message
    assert any(agent_id in message for agent_id in env.agents)


def test_run_timestep_use_llm_true_succeeds_once_every_agent_has_an_assigned_model():
    """Sanity companion: the same environment, once every agent is given an
    assigned_model, must not raise the Fix 3 guard.
    """
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")
    client = mock_openrouter_client({})
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    assert isinstance(result.transactions, list)


def test_negotiation_conversation_history_includes_both_sides_offers_by_round_three():
    """Fix 4 (Important): each side's conversation_history must be rebuilt
    from the full, both-sides NegotiationSession.conversation_history, not
    incrementally appended from only the opponent's last offer. With
    COUNTER_OFFER on both sides (never terminating early), by the seller's
    second turn (round 3 -- buyer:0, seller:1, buyer:2, seller:3) its
    rebuilt conversation_history must mention BOTH agent_ids, not just the
    buyer's (the old bug: a side's history only ever held the opponent's
    offers, never its own).
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    buyer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    seller = next(a for a in env.agents.values() if a.agent_class == "seller")

    buyer_context = _base_decision_context(wallet_balances=dict(buyer.wallet.balances))
    seller_context = _base_decision_context(wallet_balances=dict(seller.wallet.balances))

    client = mock_openrouter_client(
        {
            "buyer-model": _decision_json(action="COUNTER_OFFER", price=95.0),
            "seller-model": _decision_json(action="COUNTER_OFFER", price=85.0),
        }
    )
    decision_log: list = []

    buyer_decide = _make_llm_decide_closure(
        buyer,
        "buyer",
        buyer_context,
        "buyer-model",
        client,
        {"USDC"},
        {"ethereum"},
        90.0,
        decision_log,
        buyer_wallet_balances=dict(buyer.wallet.balances),
    )
    seller_decide = _make_llm_decide_closure(
        seller,
        "seller",
        seller_context,
        "seller-model",
        client,
        {"USDC"},
        {"ethereum"},
        90.0,
        decision_log,
        buyer_wallet_balances=dict(buyer.wallet.balances),
    )

    session = run_llm_negotiation(buyer.agent_id, seller.agent_id, buyer_decide, seller_decide, max_rounds=4)

    assert session.status == NegotiationStatus.MAX_ROUNDS_REACHED
    assert any(buyer.agent_id in line for line in seller_context.conversation_history)
    assert any(seller.agent_id in line for line in seller_context.conversation_history)
