import ast
import importlib
import json
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
from src.llm.hallucination_detector import HallucinationDirection
from src.llm.llm_router import OPENROUTER_BASE_URL
from src.market.goods import Good
from src.market.pricing_engine import true_price
from src.negotiation.llm_negotiation_engine import NegotiationSession, NegotiationStatus, run_llm_negotiation
from src.simulation.environment import Environment
from src.simulation.event_queue import EventQueue
from src.simulation.simulation_runner import SimulationConfig, SimulationRunner
from src.simulation.timestep import LLMDecisionRecord, _make_llm_decide_closure, decide_single_model, run_timestep
from src.transactions.transaction import TransactionStatus
from src.utils.constants import REPO_ROOT
from tests.llm_test_helpers import mock_openrouter_client, mock_polygon_client


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


def test_run_timestep_records_fired_shocks_into_the_event_log():
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.INFLATION, magnitude=0.02)],
        {"consumer": 2, "merchant": 2},
    )
    rng = random.Random(0)

    run_timestep(env, day=0, rng=rng)

    assert len(env.event_log.all_events()) == 1
    assert env.event_log.all_events()[0].type == ShockType.INFLATION


def test_run_timestep_does_not_record_anything_into_the_event_log_on_a_quiet_day():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    run_timestep(env, day=0, rng=rng)

    assert env.event_log.all_events() == []


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


def test_run_timestep_advances_price_index_by_compounding_inflation():
    """Fix 5 (Task 11 review): `advance_price_index`
    (src/agents/wealth.py) had zero callers anywhere in the codebase, so
    env.price_index stayed at its constructed 1.0 forever and
    real_purchasing_power never actually reflected inflation -- only
    nominal wallet changes. This fires an INFLATION shock on day 0 (which
    permanently raises env.macro_state.inflation, per apply_shock) and runs
    two days to confirm run_timestep now advances env.price_index by
    compounding that inflation rate's daily-equivalent rate once per day,
    matching advance_price_index's own annual-to-daily conversion formula
    (already covered in isolation by tests/test_wealth.py).
    """
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.INFLATION, magnitude=0.03)],
        {"consumer": 2, "merchant": 2},
    )
    rng = random.Random(0)
    assert env.price_index == pytest.approx(1.0)

    run_timestep(env, day=0, rng=rng)
    # baseline.yaml's own initial_state.inflation is added to by the shock's
    # magnitude (apply_shock: `updated.inflation += shock.magnitude`), so
    # this doesn't assume a starting inflation of exactly 0.0 -- only that
    # whatever env.macro_state.inflation ends up as on day 0 is the ANNUAL
    # rate env.price_index gets compounded by (via its daily-equivalent),
    # day over day.
    inflation_rate = env.macro_state.inflation
    daily_rate = (1 + inflation_rate) ** (1 / 365) - 1
    price_index_day_1 = env.price_index
    assert price_index_day_1 == pytest.approx(1.0 * (1 + daily_rate))

    run_timestep(env, day=1, rng=rng)
    # No shock fires on day 1, so inflation is unchanged -- confirms the
    # per-day compounding formula Task 3 already implemented and tested in
    # isolation (tests/test_wealth.py's test_advance_price_index_converts_
    # annual_rate_to_daily_equivalent).
    assert env.macro_state.inflation == pytest.approx(inflation_rate)
    price_index_day_2 = env.price_index
    assert price_index_day_2 == pytest.approx(price_index_day_1 * (1 + daily_rate))


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


def test_run_timestep_llm_decision_record_carries_the_actual_rendered_prompt_not_the_reasoning():
    """Fix 1 (Critical, Task 11 review): `database/repository.py`'s
    `_llm_decision_log_entry` used to hash `decision.reasoning` (the
    model's own OUTPUT text) into `rendered_prompt_hash`, a column whose
    whole contract is "hash of what the model actually saw" -- silently
    wrong, since `LLMDecisionRecord` never carried the rendered prompt text
    at all. This confirms the fix at its source: `decide_single_model`
    (via `_make_llm_decide_closure`) now populates
    `LLMDecisionRecord.rendered_prompt` with the exact text
    `render_prompt` produced (captured via `_capturing_openrouter_client`,
    which echoes back the literal request body `call_model` sent) -- and
    that text is NOT the model's `reasoning` string.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")

    captured_prompts: list[str] = []
    client = _capturing_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", price=90.0),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", price=90.0),
        },
        captured_prompts,
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    assert result.llm_decisions
    for record in result.llm_decisions:
        assert record.success
        # _decision_json's fixed "reasoning" value (see this file's
        # _decision_json helper) -- the model's OUTPUT, never the prompt.
        assert record.reasoning == "test reasoning"
        assert record.rendered_prompt is not None
        assert record.rendered_prompt != record.reasoning
        # The captured request body IS the literal text render_prompt built.
        assert record.rendered_prompt in captured_prompts


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
# Task 6: cross-border FX conversion tax
# ---------------------------------------------------------------------------


def test_run_timestep_deterministic_path_applies_fx_tax_on_cross_zone_settlement():
    """Buyer holds only EURC (so choose_currency_and_chain has no other
    option) and is tagged currency_zone="USD" -- a cross-zone settlement --
    so every settled transaction must carry a nonzero fx_tax_paid equal to
    paid_value * fx_tax_rate (0.0002), and the buyer's wallet must be
    debited price + tax (verified via settle()'s own tests; here we check
    the Transaction fields run_timestep produces).
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    buyer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    buyer.wallet.balances = {"EURC": 100000.0}
    buyer.currency_zone = "USD"
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0
    for tx in settled:
        assert tx.currency_symbol == "EURC"
        assert tx.fx_tax_paid == pytest.approx(tx.paid_value * 0.0002)
        assert tx.fx_tax_paid > 0.0


def test_run_timestep_deterministic_path_has_zero_fx_tax_when_buyer_zone_is_none():
    """Environment.build's count-based path never sets currency_zone -- the
    default None must mean zero FX tax, not a crash.
    """
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0
    assert all(tx.fx_tax_paid == 0.0 for tx in settled)


def test_run_timestep_llm_path_applies_fx_tax_on_cross_zone_settlement():
    """Same cross-zone setup as the deterministic-path test above, but
    driving the LLM-vs-LLM negotiation path (use_llm=True) so both
    Transaction-construction call sites in run_timestep are covered.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    buyer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    buyer.wallet.balances = {"EURC": 100000.0}
    buyer.currency_zone = "USD"
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")

    client = mock_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", currency="EURC", price=90.0),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", currency="EURC", price=90.0),
        }
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0
    for tx in settled:
        assert tx.currency_symbol == "EURC"
        assert tx.fx_tax_paid == pytest.approx(tx.paid_value * 0.0002)
        assert tx.fx_tax_paid > 0.0


def test_run_timestep_llm_path_has_zero_fx_tax_when_buyer_zone_is_none():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")

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
    assert all(tx.fx_tax_paid == 0.0 for tx in settled)


# ---------------------------------------------------------------------------
# Task 13: settlement-currency conversion for non-USD-pegged currencies
#
# Prices flow through true_price() -> negotiate()/LLM negotiation as raw USD
# numbers, but validate_transaction/settle() previously treated
# Transaction.paid_value as a literal NATIVE-UNIT amount of
# tx.currency_symbol with zero currency conversion. For a gold-pegged
# currency like PAXG (~2400 USD/unit, baseline.yaml's peg_reference_rates
# XAU=2400.0), a ~20-200 USD-priced good would try to debit ~20-200 *units*
# of PAXG (~48,000-480,000 USD) instead of the correct ~0.008-0.09 units --
# always failing "insufficient funds" for any currency that isn't pegged
# 1:1 to USD, even with an entirely adequate real-world balance.
# ---------------------------------------------------------------------------


def test_run_timestep_deterministic_path_converts_paid_value_to_native_units_for_gold_pegged_currency():
    """Buyer holds ONLY a modest, realistic 1.0 PAXG balance (~2400 USD) --
    comfortably enough to cover the correct native-unit amount for any of
    the ~20-200 USD goods, but nowhere near enough to cover the old bug's
    inflated debit (which would treat the raw ~20-200 USD price as ~20-200
    *units* of PAXG). Holding only PAXG also forces
    generate_candidates/choose_currency_and_chain to have no other option.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    buyer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    buyer.wallet.balances = {"PAXG": 1.0}
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0, "expected at least one settled PAXG transaction with a realistic 1.0 PAXG balance"
    for tx in settled:
        assert tx.currency_symbol == "PAXG"
        # Native-unit converted amount (~0.008-0.1 PAXG for a $20-$200 good
        # at 2400 USD/PAXG) -- never the raw USD-scale negotiated price
        # (~20-200), which is what the bug used to store here.
        assert 0.0 < tx.paid_value < 1.0
    # The buyer's wallet should still hold the bulk of its original 1.0
    # PAXG -- proof the debits were small native-unit amounts, not ~20-200
    # units each (which would have emptied -- and then blocked -- the
    # wallet after the very first good).
    assert buyer.wallet.balances["PAXG"] > 0.5


def test_run_timestep_llm_path_converts_paid_value_to_native_units_for_gold_pegged_currency():
    """Same bug, LLM-vs-LLM negotiation path: build_transaction_from_negotiation
    stores the LLM's raw negotiated price number verbatim (USD-scale, by the
    same convention the deterministic path uses and that
    tx.expected_value's forced USD overwrite right after it relies on).
    Confirms the settled Transaction.paid_value ends up in PAXG native
    units, not the raw USD number the model actually said.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    env.goods = [Good(name="test_good", category="test", base_price_usd=100.0)]
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")
    buyer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    seller = next(a for a in env.agents.values() if a.agent_class == "seller")
    buyer.wallet.balances = {"PAXG": 1000.0}  # ample native-unit balance either way

    asking_price = seller.asking_price(true_price(env.goods[0]))  # 100.0 * 1.10 = 110.0

    client = mock_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", currency="PAXG", price=asking_price),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", currency="PAXG", price=asking_price),
        }
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0
    expected_native_paid_value = asking_price / 2400.0  # baseline.yaml's peg_reference_rates XAU=2400.0
    for tx in settled:
        assert tx.currency_symbol == "PAXG"
        assert tx.paid_value == pytest.approx(expected_native_paid_value, rel=1e-6)
        # Guards directly against the raw-USD-number-stored-as-native-units bug.
        assert tx.paid_value != pytest.approx(asking_price)


def test_run_timestep_deterministic_path_fx_tax_matches_native_unit_paid_value_scale():
    """Ordering fix: fx_tax_paid must be computed on the CONVERTED
    (native-unit) paid_value, not the raw pre-conversion USD number --
    settle()'s `tx.paid_value + tx.fx_tax_paid` debits both in
    tx.currency_symbol native units, so mixing scales there would be
    nonsensical. EUR's peg_reference_rate is bumped to 5.0 here (vs.
    baseline.yaml's 1.08) so a same-scale-vs-mixed-scale bug produces an
    unmistakable ~5x discrepancy rather than a marginal, easy-to-miss one.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    env.macro_state.peg_reference_rates["EUR"] = 5.0
    buyer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    buyer.wallet.balances = {"EURC": 100000.0}
    buyer.currency_zone = "USD"  # cross-zone vs. EURC's EUR zone -> fx tax applies
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0
    for tx in settled:
        assert tx.currency_symbol == "EURC"
        assert tx.fx_tax_paid > 0.0
        # A EUR peg of 5.0 vs. USD's 1.0 means USD->EURC divides by 5 --
        # the converted native-unit paid_value must be well under what the
        # raw pre-conversion USD number would have been (up to ~250 for
        # baseline's marked-up goods).
        assert tx.paid_value < 60.0
        # fx_tax_paid must be a percentage of the SAME native-unit
        # paid_value that gets debited alongside it in settle() -- not a
        # percentage of the pre-conversion USD number.
        assert tx.fx_tax_paid == pytest.approx(tx.paid_value * 0.0002, rel=1e-9)


def test_run_timestep_llm_path_hallucination_detection_stays_unit_consistent_for_gold_pegged_currency():
    """detect_hallucination(expected_value, paid_value, ...) compares a USD
    listing.true_price against the LLM's own raw per-round proposed price
    (NegotiationAction.price) -- both USD-scale numbers, by convention, and
    this comparison happens mid-negotiation, before Transaction.paid_value
    is ever converted to native units. Confirms that a model paying exactly
    the fair (USD-scale) price for a PAXG-settled trade is classified
    ACCURATE (~0% error), not a spurious ~100% "underpayment" artifact that
    would appear if this comparison were ever fed a native-unit (post
    -conversion) paid_value against a USD expected_value.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    env.goods = [Good(name="test_good", category="test", base_price_usd=100.0)]
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")
    buyer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    seller = next(a for a in env.agents.values() if a.agent_class == "seller")
    buyer.wallet.balances = {"PAXG": 1000.0}

    asking_price = seller.asking_price(true_price(env.goods[0]))  # 110.0

    client = mock_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", currency="PAXG", price=asking_price),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", currency="PAXG", price=asking_price),
        }
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    hallucinations = [r.hallucination for r in result.llm_decisions if r.hallucination is not None]
    assert hallucinations
    for h in hallucinations:
        assert h.direction == HallucinationDirection.ACCURATE
        assert h.percentage_error == pytest.approx(0.0, abs=1e-6)


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


def test_simulation_runner_module_has_no_top_level_httpx_import():
    """Fix 1 (Critical), static check: sandbox/sandbox_launcher.py's E2B
    provisioning installs only `pydantic sqlalchemy pyyaml python-dotenv
    pandas` (no httpx) and then imports simulation_runner for a purely
    deterministic (use_llm=False) run. Any module-level `import httpx`
    line in simulation_runner.py would break that path even though it
    never touches the LLM code. This inspects the file's own module-level
    AST (not the live, possibly-already-cached sys.modules) so it can't be
    fooled by other test files having already imported httpx first.
    """
    source = (REPO_ROOT / "src" / "simulation" / "simulation_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_modules = {"httpx"}

    for node in tree.body:  # module level only -- deliberately not ast.walk
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, (
                    f"module-level `import {alias.name}` reintroduces the sandbox-breaking hard import"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules, (
                f"module-level `from {node.module} import ...` reintroduces the sandbox-breaking hard import"
            )


def test_simulation_runner_use_llm_false_works_even_if_httpx_cannot_be_imported(monkeypatch):
    """Fix 1 (Critical), behavioral check: reproduce the actual sandbox
    failure mode by making a fresh `import httpx` raise, then confirming a
    freshly (re-)imported simulation_runner module still works end-to-end for a
    deterministic run. `sys.modules["httpx"] = None` is the standard trick
    for forcing ImportError on `import httpx` (see Python docs on
    sys.modules); monkeypatch restores the original entries automatically.
    """
    monkeypatch.setitem(sys.modules, "httpx", None)
    monkeypatch.delitem(sys.modules, "src.simulation.simulation_runner", raising=False)
    # Also clear timestep since it's imported by simulation_runner
    monkeypatch.delitem(sys.modules, "src.simulation.timestep", raising=False)

    module = importlib.import_module("src.simulation.simulation_runner")

    config = module.SimulationConfig(
        agent_mix={"consumer": 2, "merchant": 2},
        num_days=2,
        scenario="baseline",
        random_seed=42,
    )
    result = module.SimulationRunner().run(config)  # use_llm=False (default)

    assert len(result.timesteps) == 2
    all_transactions = [tx for step in result.timesteps for tx in step.transactions]
    assert isinstance(all_transactions, list)


# ---------------------------------------------------------------------------
# Task 9: live Polygon price wiring + CurrencyHistory/MacroHistory wiring gap
# ---------------------------------------------------------------------------


def _capturing_openrouter_client(model_responses: dict[str, dict], captured_prompts: list[str]) -> httpx.Client:
    """Like tests.llm_test_helpers.mock_openrouter_client, but additionally
    appends each request's rendered prompt text into `captured_prompts` so a
    test can assert on the *literal prompt text* render_prompt produced --
    not just that some function was called. This reads
    body["messages"][0]["content"], the exact field src.llm.llm_router
    .call_model sends the rendered prompt as (`messages = [{"role": "user",
    "content": prompt}]`), so whatever this captures is byte-for-byte what
    the model would have seen.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_prompts.append(body["messages"][0]["content"])
        model_id = body["model"]
        if model_id not in model_responses:
            return httpx.Response(404, json={"error": f"no mocked response for model {model_id!r}"})
        content = json.dumps(model_responses[model_id])
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))


def test_run_timestep_llm_path_wires_currency_history_into_rendered_prompt():
    """Step 0: `build_currency_history`/`build_macro_history` (Task 8) must
    actually reach the rendered prompt text sent to the model -- not just be
    computed and dropped on the floor, which is the exact gap Task 8's
    review found. Fires a DEPEG_EVENT for USDC on day 0 -- recorded into
    env.event_log at the very top of run_timestep, before the buyer loop
    runs -- so build_currency_history("USDC", day=0) sees it and produces a
    `recent_events` entry ("Day 0: depeg_event (magnitude 0.05)"); the
    prompt actually sent to the (mocked) model must contain that literal
    text, proving currency_history was threaded through
    build_decision_context -> render_prompt, not merely constructed.
    """
    env = _build_env_with_shocks(
        [ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.05, target_currency="USDC")],
        {"consumer": 1, "merchant": 1},
    )
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")

    captured_prompts: list[str] = []
    client = _capturing_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", currency="USDC", price=90.0),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", currency="USDC", price=90.0),
        },
        captured_prompts,
    )
    rng = random.Random(0)

    run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    assert captured_prompts
    assert any("depeg_event" in prompt and "magnitude 0.05" in prompt for prompt in captured_prompts)


def test_run_timestep_with_polygon_client_populates_live_price_snapshots():
    """`polygon_client` wiring: when supplied, run_timestep fetches one
    LivePriceSnapshot per tradable currency's reference ticker once per day
    and threads it into every build_decision_context call for that day --
    verified the same way as the history test above, via the literal prompt
    text sent to the (mocked) model, not just that fetch_live_price was
    invoked.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")
    polygon_client = mock_polygon_client({"X:USDCUSD": 1.0002})

    captured_prompts: list[str] = []
    openrouter_client = _capturing_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", currency="USDC", price=90.0),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", currency="USDC", price=90.0),
        },
        captured_prompts,
    )
    rng = random.Random(0)

    run_timestep(
        env,
        day=0,
        rng=rng,
        use_llm=True,
        openrouter_client=openrouter_client,
        polygon_client=polygon_client,
    )

    assert captured_prompts
    assert any("1.0002" in prompt for prompt in captured_prompts)


def test_run_timestep_without_polygon_client_leaves_live_price_snapshots_empty():
    """Default (polygon_client=None, the existing signature's default)
    behavior must stay unchanged: no live price section populated for any
    currency, matching build_decision_context's own default of {}.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")

    captured_prompts: list[str] = []
    client = _capturing_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", currency="USDC", price=90.0),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", currency="USDC", price=90.0),
        },
        captured_prompts,
    )
    rng = random.Random(0)

    run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    assert captured_prompts
    assert all("(no live price data available)" in prompt for prompt in captured_prompts)


def test_build_from_population_uses_full_universe_when_currencies_is_none():
    from src.agents.population import generate_agent_population

    population = generate_agent_population(seed=0, model_candidates=["vendor/model"])

    env = Environment.build_from_population("baseline", population)

    assert len(env.currencies) == 9  # full real universe


def test_build_from_population_uses_supplied_currencies_when_given():
    from src.agents.population import generate_agent_population
    from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS

    population = generate_agent_population(seed=0, model_candidates=["vendor/model"])
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_governance"]

    env = Environment.build_from_population(
        "baseline", population, currencies={option_a.symbol: option_a, option_b.symbol: option_b}
    )

    assert len(env.currencies) == 2


# ---------------------------------------------------------------------------
# Review of Task 13 (d8f6568): mid-negotiation ACCEPT funds check compared
# USD-scale decision.price against native-unit wallet_balances directly.
# `_make_llm_decide_closure`'s buyer_wallet_balances (passed to
# adapt_decision -> validate_decision as the funds-check input) is native
# units of each currency (straight from agent.wallet.balances), but
# decision.price is USD-scale by the negotiation convention Task 13
# established. For a gold-pegged currency (~2400 USD/unit), a realistic
# small native-unit balance -- comfortably sufficient in USD terms -- was
# spuriously rejected as "insufficient funds" because e.g. 1.0 (native PAXG)
# < 110.0 (USD-scale price), even though 1.0 PAXG is worth ~2400 USD.
# ---------------------------------------------------------------------------


def test_run_timestep_llm_path_accepts_realistic_gold_balance_against_usd_scale_price():
    """Buyer holds ONLY a realistic, modest 1.0 PAXG balance (~2400 USD) --
    comfortably enough to cover a ~110 USD trade, but the pre-fix raw
    comparison (1.0 available < 110.0 USD-scale price) would spuriously
    reject the ACCEPT as "insufficient funds" and end the negotiation in a
    synthetic WALK_AWAY with zero settlements. Confirms the funds check now
    converts wallet_balances to USD before comparing against decision.price,
    so the trade settles instead.
    """
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    env.goods = [Good(name="test_good", category="test", base_price_usd=100.0)]
    _assign_models(env, "test-vendor/buyer-model", "test-vendor/seller-model")
    buyer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    seller = next(a for a in env.agents.values() if a.agent_class == "seller")
    buyer.wallet.balances = {"PAXG": 1.0}  # ~2400 USD equivalent -- realistic, not "ample"

    asking_price = seller.asking_price(true_price(env.goods[0]))  # 100.0 * 1.10 = 110.0

    client = mock_openrouter_client(
        {
            "test-vendor/buyer-model": _decision_json(action="OFFER", currency="PAXG", price=asking_price),
            "test-vendor/seller-model": _decision_json(action="ACCEPT", currency="PAXG", price=asking_price),
        }
    )
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled) > 0, (
        "expected the ~110 USD trade to settle against a realistic 1.0 PAXG "
        "(~2400 USD) balance -- got 0 settlements, meaning the funds check "
        "is still comparing USD-scale price against native-unit balance"
    )
    for tx in settled:
        assert tx.currency_symbol == "PAXG"
    # No negotiation should have failed with the stale "insufficient funds"
    # reasoning that the raw native-vs-USD comparison bug produced (the
    # actual pre-fix failure_reason was
    # "still invalid after one correction: Insufficient funds").
    for record in result.llm_decisions:
        assert "Insufficient funds" not in (record.failure_reason or "")
