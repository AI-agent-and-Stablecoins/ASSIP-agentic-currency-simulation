"""Runs one simulation day: the 13-step rule-based lifecycle (steps 1-13 of
project_instructions.md section 6; LLM reasoning at step 6 is skipped since
Phase 1 has no LLM calls; step 14 -- persistence -- is the caller's job so
this stays testable without a database).

Phase 3 adds an opt-in `use_llm=True` path (see `run_timestep`'s docstring)
that replaces the deterministic `choose_currency_and_chain` + `negotiate()`
call with a real per-agent LLM decision and full LLM-vs-LLM negotiation.

`httpx` and the `httpx`-dependent slices of `src.llm` (agent_reasoning,
llm_router, market_intelligence) are deliberately NOT imported at module
level: `sandbox/sandbox_launcher.py` provisions its E2B sandbox with only
`pydantic sqlalchemy pyyaml python-dotenv pandas` (no `httpx`) and then
imports `simulation_runner` -> this module for a purely deterministic
(use_llm=False) run. A module-level `import httpx` here would break that
path even though it never touches the LLM code. Instead, those imports are
function-local, executed only inside the functions/branches that actually
need them (`decide_single_model`, and `run_timestep`'s `use_llm=True`
branch) -- the same convention `src.agents.base_agent.BaseAgent
.build_llm_context()` already uses for exactly this reason. `from __future__
import annotations` keeps the `httpx.Client`/`AgentDecisionContext` type
hints working without a module-level import (annotations are never
evaluated at runtime unless something calls `typing.get_type_hints` on
these functions, which nothing here does).
"""

from __future__ import annotations

import json
import random
from typing import Callable

from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.agents.buyer_agent import BuyerAgent
from src.agents.seller_agent import SellerAgent
from src.agents.wealth import advance_price_index
from src.blockchain.routing_engine import CurrencyChainOption, generate_candidates
from src.economy.fx_dynamics import advance_eur_usd_rate
from src.economy.fx_tax import compute_fx_tax, load_fx_params
from src.economy.shocks import ShockEvent, ShockType, apply_currency_shock, apply_shock
from src.llm.decision_adapter import DecisionValidationError, NegotiationAction, adapt_decision
from src.llm.decision_schema import Decision, DecisionAction
from src.llm.decision_to_transaction import build_transaction_from_negotiation
from src.llm.hallucination_detector import HallucinationResult, detect_hallucination
from src.market.pricing_engine import true_price
from src.negotiation.conversation_history import ConversationLog
from src.negotiation.llm_negotiation_engine import NegotiationSession, run_llm_negotiation
from src.negotiation.negotiation_engine import negotiate
from src.simulation.environment import Environment
from src.simulation.scheduler import agent_activation_order
from src.transactions.settlement import settle
from src.transactions.transaction import Transaction, TransactionStatus
from src.transactions.validation import validate_transaction


class LLMDecisionRecord(BaseModel):
    """One `decide_single_model` call's outcome (success or total failure).

    Deliberately a thin, additive shape -- not a 1:1 mirror of
    `database.repository.LLMDecisionLogEntry`/`HallucinationLogEntry` --
    since several of those records' fields (simulation_id, prompt_version,
    scenario, domestic_or_cross_border, governance_prompt_enabled) are
    run/experiment-level concerns that run_timestep has no opinion about;
    the persistence task is expected to fill those in from its own caller
    context while pulling the rest (agent/model/decision/hallucination
    fields) straight from here.

    `rendered_prompt` (the actual prompt text sent to the model, captured in
    `decide_single_model` right after `render_prompt` runs) IS carried here,
    despite the "run-level concern" framing above -- it isn't a
    run/experiment-level concern, it's per-decision data this module already
    computes and would otherwise silently drop. `database/repository.py`'s
    `_llm_decision_log_entry` derives `rendered_prompt_hash`/`system_prompt`
    from it (a Task 11 review fix: it previously hashed `reasoning` -- the
    model's OUTPUT -- instead, which produced a real-looking but wrong hash).
    """

    agent_id: str
    agent_type: str
    risk_profile: str
    utility_type: str
    requested_model: str
    actual_model: str
    success: bool
    correction_attempts: int = 0
    failure_reason: str | None = None
    negotiation_id: str | None = None
    round: int | None = None
    action: str | None = None
    currency_symbol: str | None = None
    chain_name: str | None = None
    amount: float | None = None
    price: float | None = None
    reasoning: str | None = None
    rendered_prompt: str | None = None
    hallucination: HallucinationResult | None = None
    spread_optimal_currency: str | None = None
    spread_optimal_chain: str | None = None
    gas_optimal_currency: str | None = None
    gas_optimal_chain: str | None = None


class TimestepResult(BaseModel):
    day: int
    transactions: list[Transaction] = Field(default_factory=list)
    negotiations: list[ConversationLog] = Field(default_factory=list)
    fired_shocks: list[ShockEvent] = Field(default_factory=list)
    memory_events: list[tuple[str, str, str]] = Field(default_factory=list)
    # Populated only when use_llm=True. Kept separate from `negotiations`
    # (which is typed for the rule-based ConversationLog) and from
    # `transactions` (shared/unconditional) since NegotiationSession carries
    # richer per-round LLM data a future persistence task may want.
    llm_decisions: list[LLMDecisionRecord] = Field(default_factory=list)
    llm_negotiations: list[NegotiationSession] = Field(default_factory=list)


_SHOCK_MEMORY_LABELS = {
    ShockType.DEPEG_EVENT: "Depeg",
    ShockType.GOVERNANCE_DOWNGRADE: "GovernanceDowngrade",
    ShockType.LIQUIDITY_CRUNCH: "LiquidityCrunch",
    ShockType.REGULATORY_ENFORCEMENT: "RegulatoryEnforcement",
}


def decide_single_model(
    agent_class: str,
    context: AgentDecisionContext,
    model_id: str,
    client: httpx.Client,
    supported_currencies: set[str],
    supported_chains: set[str],
    payer_wallet_balances: dict[str, float] | None = None,
    telemetry: dict | None = None,
) -> NegotiationAction | None:
    """Single-model decision helper for Phase 3's fixed per-agent model
    assignment.

    Bypasses `src.llm.agent_reasoning.decide()`'s shared-roster/policy
    resolution and `call_with_fallback_chain`'s multi-model substitution --
    both assume Phase 2's shared-model-comparison design, whereas Phase 3
    assigns exactly one fixed model per agent (see
    `src.agents.population.generate_agent_population`).

    Contract: render_prompt -> call_model(single model_id) -> adapt_decision,
    narrowed to the exact candidates offered this round (supported_currencies/
    supported_chains), not the full currency/chain universe. On
    DecisionValidationError, one corrected re-prompt is sent (the prompt plus
    a correction message, matching `decide()`'s existing correction-message
    wording); if that is still invalid, or the model call itself fails
    (ModelCallFailedError/AuthenticationError) at any point, this returns
    None. There is no deterministic fallback: a hard failure must surface as
    "no decision" so the caller can skip the transaction, not silently
    substitute rule-based behavior.

    `payer_wallet_balances`, if given, is the balance dict `adapt_decision`
    checks an ACCEPT against for funds-sufficiency. Defaults to `context
    .agent.wallet_balances` (this side's own wallet) when omitted, which is
    only correct when `context` belongs to the buyer -- the buyer is always
    the one paying, so a seller-side call MUST pass the buyer's balances
    explicitly (see `_make_llm_decide_closure`, which always threads the
    buyer's wallet through for both sides); otherwise a seller ACCEPTing a
    currency it happens not to hold itself -- irrelevant, since it isn't the
    payer -- would be spuriously rejected as "insufficient funds".

    `telemetry`, if given a dict, is populated in place with
    `correction_attempts` (0 or 1) and `failure_reason` (None on success) so
    callers that need that detail for logging (see run_timestep's
    `LLMDecisionRecord`) don't have to change this function's primary
    `NegotiationAction | None` return contract.
    """
    from src.llm.agent_reasoning import render_prompt
    from src.llm.llm_router import AuthenticationError, ModelCallFailedError, call_model

    if telemetry is not None:
        telemetry.setdefault("correction_attempts", 0)
        telemetry.setdefault("failure_reason", None)

    funds_check_balances = payer_wallet_balances if payer_wallet_balances is not None else context.agent.wallet_balances

    schema_json = json.dumps(Decision.model_json_schema())
    prompt = render_prompt(agent_class, context, schema_json)
    if telemetry is not None:
        # Captured unconditionally (before the model call, which may fail) so
        # the caller can always recover the exact text this decision was
        # made from -- see LLMDecisionRecord.rendered_prompt's docstring.
        # Deliberately the ORIGINAL rendered prompt, not the corrected
        # re-prompt below: this decision's provenance is "what render_prompt
        # produced for this round", one value per decision, matching
        # prompt_version's own one-per-decision granularity.
        telemetry["rendered_prompt"] = prompt

    try:
        decision = call_model(prompt, model_id, client)
    except (ModelCallFailedError, AuthenticationError) as exc:
        if telemetry is not None:
            telemetry["failure_reason"] = f"model call failed: {exc}"
        return None

    try:
        return adapt_decision(decision, supported_currencies, supported_chains, funds_check_balances)
    except DecisionValidationError as exc:
        if telemetry is not None:
            telemetry["correction_attempts"] = 1
        corrected_prompt = (
            f"{prompt}\n\nYour previous proposal was economically invalid: {exc.reason}. "
            "Respond again with a corrected JSON decision matching the schema."
        )
        try:
            corrected_decision = call_model(corrected_prompt, model_id, client)
        except (ModelCallFailedError, AuthenticationError) as call_exc:
            if telemetry is not None:
                telemetry["failure_reason"] = f"correction re-prompt call failed: {call_exc}"
            return None

        try:
            return adapt_decision(corrected_decision, supported_currencies, supported_chains, funds_check_balances)
        except DecisionValidationError as final_exc:
            if telemetry is not None:
                telemetry["failure_reason"] = f"still invalid after one correction: {final_exc.reason}"
            return None


def _spread_and_gas_optimal(candidates: list[CurrencyChainOption]) -> tuple[str, str, str, str]:
    """Identifies which candidate this round was spread-optimal (highest
    liquidity_score -- the codebase's bid-ask-spread proxy, per Plan 2's
    design spec) vs. gas-optimal (lowest gas_fee). Used by Plan 5's H2
    tradeoff-sample design (see docs/superpowers/specs/
    2026-08-02-phase3-plan5-econometrics-design.md Sec 2) to identify
    whether a genuine spread-vs-gas tradeoff existed that round.
    """
    spread_optimal = max(candidates, key=lambda c: c.liquidity_score)
    gas_optimal = min(candidates, key=lambda c: c.gas_fee)
    return (
        spread_optimal.currency_symbol,
        spread_optimal.chain_name,
        gas_optimal.currency_symbol,
        gas_optimal.chain_name,
    )


def _make_llm_decide_closure(
    agent: BaseAgent,
    agent_class: str,
    context: AgentDecisionContext,
    model_id: str,
    client: httpx.Client,
    supported_currencies: set[str],
    supported_chains: set[str],
    listing_true_price: float,
    decision_log: list[LLMDecisionRecord],
    buyer_wallet_balances: dict[str, float],
    spread_optimal_currency: str,
    spread_optimal_chain: str,
    gas_optimal_currency: str,
    gas_optimal_chain: str,
) -> Callable[[NegotiationSession], NegotiationAction]:
    """Builds one side's `buyer_decide`/`seller_decide` closure for
    `run_llm_negotiation`. Each call: (1) rebuilds `context.conversation_history`
    from scratch out of `session.conversation_history` (the full,
    both-sides-in-order offer log `NegotiationSession` already keeps) so this
    agent sees every prior offer -- its own as well as the opponent's, not
    just the opponent's last move -- (2) calls `decide_single_model` passing
    `buyer_wallet_balances` as the funds-check balances regardless of which
    side this closure is for (the buyer is always the payer, so an ACCEPT's
    funds check must never be run against the seller's own wallet), (3)
    records one `LLMDecisionRecord` (success or failure, with a
    `detect_hallucination` call whenever a price was actually produced), and
    (4) on total failure, returns a synthetic WALK_AWAY action so
    `run_llm_negotiation`'s `Callable[..., NegotiationAction]` contract stays
    intact -- WALK_AWAY is a neutral "no decision" signal, not a
    rule-based substitute, and it terminates the negotiation so
    `build_transaction_from_negotiation` returns None for this pair.
    """

    def _decide(session: NegotiationSession) -> NegotiationAction:
        context.conversation_history = [
            f"Round {offer.round}: {offer.agent_id} {offer.action.value} at price {offer.price} "
            f"{offer.currency_symbol} on {offer.chain_name}. Reasoning: {offer.reasoning}"
            for offer in session.conversation_history
        ]

        telemetry: dict = {}
        action = decide_single_model(
            agent_class,
            context,
            model_id,
            client,
            supported_currencies,
            supported_chains,
            payer_wallet_balances=buyer_wallet_balances,
            telemetry=telemetry,
        )

        hallucination: HallucinationResult | None = None
        if action is not None:
            hallucination = detect_hallucination(
                listing_true_price,
                action.price,
                currency_symbol=action.currency_symbol,
                chain_name=action.chain_name,
                requested_model=model_id,
                actual_model=model_id,
                agent_type=agent.agent_class,
                risk_profile=agent.risk_profile,
            )

        decision_log.append(
            LLMDecisionRecord(
                agent_id=agent.agent_id,
                agent_type=agent.agent_class,
                risk_profile=agent.risk_profile,
                utility_type=agent.utility_type,
                requested_model=model_id,
                actual_model=model_id,
                success=action is not None,
                correction_attempts=telemetry.get("correction_attempts", 0),
                failure_reason=telemetry.get("failure_reason"),
                negotiation_id=session.negotiation_id,
                round=session.current_round,
                action=action.action.value if action is not None else None,
                currency_symbol=action.currency_symbol if action is not None else None,
                chain_name=action.chain_name if action is not None else None,
                amount=action.amount if action is not None else None,
                price=action.price if action is not None else None,
                reasoning=action.reasoning if action is not None else None,
                rendered_prompt=telemetry.get("rendered_prompt"),
                hallucination=hallucination,
                spread_optimal_currency=spread_optimal_currency,
                spread_optimal_chain=spread_optimal_chain,
                gas_optimal_currency=gas_optimal_currency,
                gas_optimal_chain=gas_optimal_chain,
            )
        )

        if action is None:
            return NegotiationAction(
                action=DecisionAction.WALK_AWAY,
                price=0.0,
                amount=0.0,
                currency_symbol="",
                chain_name="",
                reasoning=f"LLM decision failed: {telemetry.get('failure_reason') or 'unknown failure'}",
            )
        return action

    return _decide


def _polygon_ticker(symbol: str) -> str:
    """Derive this currency's Polygon crypto-aggregate reference ticker.

    No currency config declares a ticker field (checked `CurrencyConfig` and
    its subclasses in src/currencies/*.py -- none exists), so this task
    derives one. Every configured currency (configs/currencies/*.yaml) is
    itself a crypto token -- USD-pegged (USDC, USDT, DAI, FDUSD,
    Tokenized_Deposits), EUR-pegged (EURC, EURT), and gold-backed (PAXG,
    XAUT) alike -- so all get the same Polygon crypto-aggregate ticker shape
    scripts/calibrate_currency_configs.py already uses for the USD-pegged
    and gold-backed cases (USD_STABLECOIN_TICKERS/GOLD_TOKEN_TICKERS there:
    "X:USDCUSD", "X:PAXGUSD", "X:XAUTUSD", ...): `X:{symbol}USD`, matching
    `fetch_live_price`'s own documented example and
    tests/test_market_intelligence.py's fixtures ("X:USDCUSD").

    A peg-conditional scheme (forex `C:EURUSD` for EUR, `C:XAUUSD` for gold)
    was considered, since that calibration script also defines
    FOREX_TICKERS -- but those are reference *peg* rates used to seed
    macro_state.peg_reference_rates, not per-token prices; that same script
    still fetches PAXG's and XAUT's own market price via `X:{symbol}USD`,
    not a forex ticker. Reusing that exact convention for EURC/EURT keeps
    this mapping uniform rather than inventing an unproven forex-ticker
    guess for tokens that trade as crypto assets, not as currency pairs.
    `fetch_live_price` already degrades gracefully (price=None,
    unavailable_reason set) if Polygon has no data for a derived ticker, so
    a currency this guess doesn't fit for is a safe no-data default, not a
    crash.
    """
    return f"X:{symbol}USD"


def run_timestep(
    env: Environment,
    day: int,
    rng: random.Random,
    max_negotiation_rounds: int = 10,
    agreement_tolerance: float = 0.01,
    concession_rate: float = 0.3,
    use_llm: bool = False,
    openrouter_client: httpx.Client | None = None,
    polygon_client: httpx.Client | None = None,
) -> TimestepResult:
    """Run one simulation day.

    When use_llm=False (the default), behavior is byte-for-byte identical to
    the original deterministic simulation: `buyer.choose_currency_and_chain`
    followed by the rule-based `negotiate()`.

    When use_llm=True, each buyer/seller/good pairing instead runs a real
    per-agent LLM decision (`decide_single_model`) and a full LLM-vs-LLM
    negotiation (`run_llm_negotiation`) in place of that deterministic
    step -- `openrouter_client` is required in that case (raises ValueError
    if None), as is every agent having a non-None `assigned_model`
    (raises ValueError naming the offending agent(s) otherwise -- silently
    calling `decide_single_model(model_id=None, ...)` would just produce a
    quiet zero-transaction run via synthetic WALK_AWAYs, which is a much
    worse failure mode than an immediate loud error). Every other lifecycle
    step (shock application, marketplace listing, memory recording) is
    shared and unconditional.

    `polygon_client`, if given (only meaningful when use_llm=True), is used
    to fetch one `LivePriceSnapshot` per currency in `env.currencies` --
    via `market_intelligence.fetch_live_price` and this module's
    `_polygon_ticker` mapping -- exactly once at the start of the day, not
    per-agent or per-negotiation-round, matching the market-data-fetch
    principle already established for that endpoint. When `polygon_client`
    is None (the default), `live_price_snapshots` stays empty, same as
    before this parameter existed. `currency_history`/`macro_history`
    (`src.economy.history_builder`) are likewise built once per day here --
    for every currency in `env.currencies` (always a superset of any single
    round's narrower candidate set, since `generate_candidates` only ever
    offers currencies from that same universe) and once for the whole
    macroeconomy respectively -- rather than being rebuilt per agent or per
    buyer/good pairing, since none of `trust_ledger`/`event_log`/
    `macro_state` changes within a day once Steps 1-2 above have run.
    """
    if use_llm and openrouter_client is None:
        raise ValueError("openrouter_client is required when use_llm=True")

    if use_llm:
        unassigned = [agent_id for agent_id, agent in env.agents.items() if agent.assigned_model is None]
        if unassigned:
            raise ValueError(
                "use_llm=True requires every agent to have an assigned_model, but the following "
                f"agent(s) have assigned_model=None: {unassigned}"
            )

    # Loaded once per timestep (cheap, deterministic YAML read) and reused for
    # every transaction below -- both the deterministic and LLM-driven paths
    # apply the same cross-border FX conversion tax (Task 6).
    fx_params = load_fx_params()

    # Steps 1-2: update macroeconomic state, currency attributes, and prices
    # from any shocks due today. Ambient daily EUR/USD noise (ASSIP Plan 5
    # whole-branch review Fix C1) is applied first, every day regardless of
    # whether a scheduled fx_rate_shock also fires -- it models real-world
    # day-to-day FX movement, while shocks model deliberate discrete events;
    # the two compose multiplicatively.
    env.macro_state = advance_eur_usd_rate(env.macro_state, day)
    due_shocks = env.event_queue.pop_due(day)
    for shock in due_shocks:
        env.event_log.record(shock)
        env.macro_state = apply_shock(env.macro_state, shock)
        env.currencies = apply_currency_shock(env.currencies, shock)
    env.refresh_exchange_rates()

    # Advance trust/peg/liquidity dynamics once per day, regardless of
    # whether any shock fired today (a quiet day is a valid input that
    # still drives recovery toward baseline).
    env.trust_ledger.update(due_shocks, env.currencies)

    # Fix 5 (Task 11 review): env.price_index previously had zero callers
    # anywhere, so it stayed at its constructed 1.0 forever and
    # real_purchasing_power never actually reflected inflation -- only
    # nominal wallet changes. Advance it here, once per day, right alongside
    # trust_ledger.update -- this is core day-loop-owned simulation state
    # (evolves whether or not persistence happens today), not
    # persistence-owned state, matching how trust_ledger/event_log are
    # already treated. Uses env.macro_state.inflation AFTER this day's
    # shocks have been applied above, same timing trust_ledger.update uses.
    env.price_index = advance_price_index(env.price_index, env.macro_state.inflation)

    # LLM-path environment-level context, built/fetched exactly once per
    # day (not per agent, not per buyer/good pairing below) -- live prices,
    # currency history, and macro history are all facts about the day, not
    # about any one decision. Function-local imports: these pull in httpx
    # (via src.llm.market_intelligence) and src.llm.agent_reasoning (via
    # src.economy.history_builder, which imports it for the
    # CurrencyHistory/MacroHistory types) -- see this file's module
    # docstring for why those must not be module-level imports.
    live_price_snapshots = {}
    currency_history = {}
    macro_history = None
    if use_llm:
        from src.economy.history_builder import build_currency_history, build_macro_history
        from src.llm.market_intelligence import fetch_live_price

        # env.currencies is always a superset of any single round's
        # narrower candidate set (generate_candidates only ever offers
        # currencies drawn from this same universe), so building/fetching
        # for every currency here -- once -- correctly covers every
        # buyer/good pairing's build_decision_context call below without
        # rebuilding anything per agent. build_decision_context itself
        # already narrows both dicts down to the round's actual candidates
        # (see its relevant_snapshots/relevant_history filtering).
        currency_history = {
            symbol: build_currency_history(env.trust_ledger, env.event_log, symbol, day)
            for symbol in env.currencies
        }
        macro_history = build_macro_history(env, day)

        if polygon_client is not None:
            live_price_snapshots = {
                symbol: fetch_live_price(_polygon_ticker(symbol), polygon_client) for symbol in env.currencies
            }

    result = TimestepResult(day=day, fired_shocks=due_shocks)

    for shock in due_shocks:
        if shock.target_currency is None or shock.type not in _SHOCK_MEMORY_LABELS:
            continue
        label = _SHOCK_MEMORY_LABELS[shock.type]
        for agent in env.agents.values():
            if agent.wallet.balances.get(shock.target_currency, 0.0) > 0:
                event_text = (
                    f"Day {day}: {shock.target_currency} {label.lower()} "
                    f"(magnitude {shock.magnitude})."
                )
                agent.memory.record_narrative(event_text)
                result.memory_events.append((agent.agent_id, label, event_text))

    env.marketplace.clear_listings()

    sellers = [a for a in env.agents.values() if isinstance(a, SellerAgent)]
    buyers = {a.agent_id: a for a in env.agents.values() if isinstance(a, BuyerAgent)}

    for seller in sellers:
        for good in env.goods:
            price = true_price(good)
            asking = seller.asking_price(price)
            env.marketplace.post_listing(seller.agent_id, good, asking)

    # Step 3: select active agents (buyers act each day; sellers already listed above).
    active_buyers = agent_activation_order(buyers, day, rng)

    for buyer in active_buyers:
        for good in env.goods:
            # Step 4: agent observes the environment (available listings).
            listings = env.marketplace.find_counterparties(good.name, exclude_agent_id=buyer.agent_id)
            if not listings:
                continue
            listing = listings[0]
            seller = env.agents[listing.seller_id]

            # Steps 5, 7-8: compute utility, choose currency and blockchain.
            candidates = generate_candidates(
                buyer.wallet.balances,
                env.currencies,
                env.chains,
                env.liquidity_pools,
                trust_ledger=env.trust_ledger,
            )
            if not candidates:
                continue

            spread_optimal_currency, spread_optimal_chain, gas_optimal_currency, gas_optimal_chain = (
                _spread_and_gas_optimal(candidates)
            )

            if use_llm:
                # Steps 6-9 (LLM path): each side gets its own AgentDecisionContext
                # (own assigned_model/risk_aversion/currency_zone/wallet), and a
                # full LLM-vs-LLM negotiation replaces the deterministic
                # choose_currency_and_chain + negotiate() call. supported_
                # currencies/chains are narrowed to exactly what was offered
                # this round -- not the full universe -- per the plan's
                # anti-hallucination tightening.
                #
                # Local imports: these pull in httpx (via src.llm.agent_reasoning /
                # src.llm.market_intelligence), so they must not be module-level --
                # see this file's module docstring for why.
                from src.llm.agent_reasoning import TransactionContext, build_decision_context
                from src.llm.market_intelligence import load_currency_profile

                supported_currencies = {c.currency_symbol for c in candidates}
                supported_chains = {c.chain_name for c in candidates}
                currency_profiles = {
                    symbol: profile
                    for symbol in supported_currencies
                    if (profile := load_currency_profile(symbol)) is not None
                }
                # NOTE: this is buyer-vs-seller currency-zone mismatch, known
                # before any settlement currency is chosen -- a related but
                # distinct concept from Task 6's FX conversion tax "cross-border"
                # check, which will compare the settlement currency's zone
                # against the buyer's currency_zone (computed later, once a
                # currency is actually agreed). Do not conflate the two names.
                counterparty_cross_zone = (
                    buyer.currency_zone is not None
                    and seller.currency_zone is not None
                    and buyer.currency_zone != seller.currency_zone
                )
                transaction_context = TransactionContext(
                    is_cross_border=counterparty_cross_zone,
                    origin_currency=buyer.currency_zone if counterparty_cross_zone else None,
                    destination_currency=seller.currency_zone if counterparty_cross_zone else None,
                )

                buyer_context = build_decision_context(
                    buyer.build_llm_context(),
                    candidates,
                    currency_profiles,
                    env.macro_state,
                    env.macro_state,
                    transaction_context,
                    live_price_snapshots=live_price_snapshots,
                    currency_history=currency_history,
                    macro_history=macro_history,
                )
                seller_context = build_decision_context(
                    seller.build_llm_context(),
                    candidates,
                    currency_profiles,
                    env.macro_state,
                    env.macro_state,
                    transaction_context,
                    live_price_snapshots=live_price_snapshots,
                    currency_history=currency_history,
                    macro_history=macro_history,
                )

                # Both closures check ACCEPT's funds validity against the buyer's
                # wallet, never the seller's -- the buyer is always the payer, so
                # a seller ACCEPTing a currency it doesn't itself hold is
                # irrelevant to whether the trade is affordable.
                #
                # decision.price is USD-scale (the negotiation convention this
                # fix chain established -- see d8f6568), but
                # buyer_context.agent.wallet_balances is native currency units
                # (used as-is for the prompt's own wallet display, which must
                # stay untouched). Converting to USD here only affects the
                # funds-CHECK input: without it, a gold-pegged buyer with a
                # realistic e.g. {"PAXG": 1.0} balance would be rejected as
                # "insufficient funds" against a ~100 USD price, since
                # 1.0 < 100.0 even though 1.0 PAXG is worth ~2400 USD.
                buyer_wallet_balances_usd = {
                    symbol: env.exchange_rates.convert(balance, symbol, "USD")
                    for symbol, balance in buyer_context.agent.wallet_balances.items()
                }
                buyer_decide = _make_llm_decide_closure(
                    buyer,
                    "buyer",
                    buyer_context,
                    buyer.assigned_model,
                    openrouter_client,
                    supported_currencies,
                    supported_chains,
                    listing.true_price,
                    result.llm_decisions,
                    buyer_wallet_balances=buyer_wallet_balances_usd,
                    spread_optimal_currency=spread_optimal_currency,
                    spread_optimal_chain=spread_optimal_chain,
                    gas_optimal_currency=gas_optimal_currency,
                    gas_optimal_chain=gas_optimal_chain,
                )
                seller_decide = _make_llm_decide_closure(
                    seller,
                    "seller",
                    seller_context,
                    seller.assigned_model,
                    openrouter_client,
                    supported_currencies,
                    supported_chains,
                    listing.true_price,
                    result.llm_decisions,
                    buyer_wallet_balances=buyer_wallet_balances_usd,
                    spread_optimal_currency=spread_optimal_currency,
                    spread_optimal_chain=spread_optimal_chain,
                    gas_optimal_currency=gas_optimal_currency,
                    gas_optimal_chain=gas_optimal_chain,
                )

                session = run_llm_negotiation(
                    buyer.agent_id,
                    seller.agent_id,
                    buyer_decide,
                    seller_decide,
                    max_rounds=max_negotiation_rounds,
                )
                result.llm_negotiations.append(session)

                tx = build_transaction_from_negotiation(
                    session, candidates, buyer.agent_id, seller.agent_id, good.name, day
                )
                if tx is None:
                    continue

                # Task 4's carry-forward: build_transaction_from_negotiation
                # stubs expected_value from the negotiation's own opening
                # offer (a negotiation anchor, not ground truth). Overwrite it
                # with the real deterministic fair value computed above (the
                # same listing.true_price the rule-based path already uses)
                # before this Transaction is used for settlement,
                # persistence, or hallucination detection.
                tx.expected_value = listing.true_price

                # Task 13: build_transaction_from_negotiation stores the
                # LLM's raw negotiated price number verbatim -- USD-scale, by
                # the same convention the deterministic path below uses (and
                # that the tx.expected_value overwrite right above already
                # relies on: listing.true_price is a USD number). settle()/
                # validate_transaction, however, treat tx.paid_value as a
                # literal NATIVE-UNIT amount of tx.currency_symbol. Without
                # this conversion, any non-USD-pegged settlement currency
                # (e.g. gold-backed PAXG/XAUT at ~2400 USD/unit) would try to
                # debit the raw USD number's worth of *units* -- ~2400x too
                # much -- and always fail "insufficient funds".
                tx.paid_value = env.exchange_rates.convert(tx.paid_value, "USD", tx.currency_symbol)

                # Task 6: cross-border FX conversion tax -- compared against
                # the settlement currency's zone (USD/EUR; gold-backed
                # currencies are zone-neutral) vs. the buyer's own
                # currency_zone (None for legacy count-based agents, which
                # never pay this tax). Computed on the final accepted price,
                # same as the deterministic path below. Must run AFTER the
                # USD->native-unit conversion above: settle() debits
                # `tx.paid_value + tx.fx_tax_paid` together in
                # tx.currency_symbol native units, so fx_tax_paid has to be a
                # percentage of the same native-unit paid_value, not of the
                # pre-conversion USD number.
                tx.fx_tax_paid = compute_fx_tax(
                    tx.paid_value, env.currencies[tx.currency_symbol], buyer.currency_zone, fx_params.fx_tax_rate
                )

                # Step 10: validate.
                validation = validate_transaction(tx, buyer.wallet, env.currencies)
                if not validation.is_valid:
                    tx.status = TransactionStatus.FAILED
                    result.transactions.append(tx)
                    continue

                # Step 11: settle payment.
                settle(tx, buyer.wallet, seller.wallet)
                env.ledger.record(tx)
                result.transactions.append(tx)

                # Step 12: update memory/preferences.
                success = tx.status == TransactionStatus.SETTLED
                buyer.update_memory(tx.currency_symbol, success)
                seller.update_memory(tx.currency_symbol, success)
            else:
                chosen = buyer.choose_currency_and_chain(candidates)

                # Step 9: negotiate.
                buyer_open = buyer.opening_offer_price(listing.true_price)
                seller_open = seller.asking_price(listing.true_price)
                agreed_price, log = negotiate(
                    buyer_opening_price=buyer_open,
                    seller_opening_price=seller_open,
                    currency_symbol=chosen.currency_symbol,
                    chain_name=chosen.chain_name,
                    true_price=listing.true_price,
                    supported_currencies=set(env.currencies.keys()),
                    max_rounds=max_negotiation_rounds,
                    agreement_tolerance=agreement_tolerance,
                    concession_rate=concession_rate,
                )
                result.negotiations.append(log)
                if agreed_price is None:
                    continue

                # Task 13: negotiate() (like the LLM path) produces a
                # USD-scale price -- true_price()/asking_price()/
                # opening_offer_price() are all computed in USD. Convert to
                # the settlement currency's native units before this becomes
                # Transaction.paid_value, which settle()/validate_transaction
                # treat as a literal native-unit amount of chosen
                # .currency_symbol. Without this, any non-USD-pegged
                # currency (e.g. gold-backed PAXG/XAUT) would try to debit
                # the raw USD number's worth of *units* instead of its
                # actual USD-equivalent worth.
                native_paid_value = env.exchange_rates.convert(agreed_price, "USD", chosen.currency_symbol)

                # Task 6: cross-border FX conversion tax (see the LLM-path
                # comment above for the full rationale) -- computed here on
                # the same env.currencies[chosen.currency_symbol]/
                # buyer.currency_zone/fx_params.fx_tax_rate inputs so both
                # paths apply identical FX-tax semantics before settlement.
                # Must run AFTER the USD->native-unit conversion above, for
                # the same unit-consistency reason given in the LLM path.
                fx_tax_paid = compute_fx_tax(
                    native_paid_value, env.currencies[chosen.currency_symbol], buyer.currency_zone, fx_params.fx_tax_rate
                )

                tx = Transaction(
                    buyer_id=buyer.agent_id,
                    seller_id=seller.agent_id,
                    good_name=good.name,
                    currency_symbol=chosen.currency_symbol,
                    chain_name=chosen.chain_name,
                    gas_fee=chosen.gas_fee,
                    expected_value=listing.true_price,
                    paid_value=native_paid_value,
                    timestep=day,
                    fx_tax_paid=fx_tax_paid,
                )

                # Step 10: validate.
                validation = validate_transaction(tx, buyer.wallet, env.currencies)
                if not validation.is_valid:
                    tx.status = TransactionStatus.FAILED
                    result.transactions.append(tx)
                    continue

                # Step 11: settle payment.
                settle(tx, buyer.wallet, seller.wallet)
                env.ledger.record(tx)
                result.transactions.append(tx)

                # Step 12: update memory/preferences.
                success = tx.status == TransactionStatus.SETTLED
                buyer.update_memory(chosen.currency_symbol, success)
                seller.update_memory(chosen.currency_symbol, success)

    # Step 13 (recording metrics) reads env.ledger/result after the fact --
    # see metrics/*.py, which query the ledger rather than being called inline here.
    return result
