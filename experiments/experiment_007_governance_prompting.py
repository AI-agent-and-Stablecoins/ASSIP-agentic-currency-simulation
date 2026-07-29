# experiments/experiment_007_governance_prompting.py
"""Experiment 007: does governance-prompting shift USDC vs. USDT selection?

Hypothesis 3 (docs/superpowers/specs/2026-07-22-phase2-llm-negotiation-layer-design.md
§1, §12): the more an agent is prompted to reason about governance/compliance,
the more it should favor the better-governed stablecoin (USDC) over the more
liquid but less transparent one (USDT).

Design: 5 pinned models (model_comparison routing policy -- no cross-model
substitution, so "model" stays a clean experimental factor, not confounded
with reliability) x 2 conditions (baseline vs. governance-emphasized
prompt). Agent profile, risk parameters, market state, available currencies,
and the transaction opportunity are held constant across every cell.

Dependent variables are tiered by evidential strength, per the design doc:
primary = observed currency selection; secondary = negotiation
outcome/hallucination rate; exploratory = whether reported_reasoning
mentions governance (never treated as proof of *why* a choice was made).
"""

import json
import os

import httpx
from dotenv import load_dotenv

from database.repository import LLMDecisionLogEntry, LLMDecisionRepository
from src.agents.agent_factory import build_agent, load_agent_profiles
from src.blockchain.chain import load_chain_universe
from src.blockchain.routing_engine import generate_candidates
from src.currencies.currency import load_currency_universe
from src.economy.macro_state import MacroState
from src.llm.agent_reasoning import AgentDecisionContext, TransactionContext, build_decision_context, render_prompt, PROMPT_VERSIONS, hash_rendered_prompt
from src.llm.decision_schema import Decision, DecisionAction
from src.llm.hallucination_detector import detect_hallucination
from src.llm.llm_router import ModelCallFailedError, build_openrouter_client, call_model, load_model_roster
from src.llm.market_intelligence import load_currency_profile
from src.utils.constants import REPO_ROOT
from src.utils.helpers import generate_id

load_dotenv(REPO_ROOT / ".env")

GOOD_TRUE_PRICE = 100.0
_COMPARED_CURRENCIES = ("USDC", "USDT")


def _build_context(governance_prompt_enabled: bool) -> AgentDecisionContext:
    """Held constant across every cell: the "consumer" agent profile (fixed
    risk parameters), the currency/chain universe, and the transaction
    opportunity. Only governance_prompt_enabled varies here -- model varies
    in run_cell."""
    profiles = load_agent_profiles()
    agent = build_agent(profiles["consumer"])
    # The "consumer" profile (configs/agent_profiles/consumer.yaml) only holds
    # USDC by default -- generate_candidates only offers currencies the wallet
    # already holds a positive balance of, so without this the agent would
    # never see USDT as a candidate and the governance-prompting comparison
    # would be degenerate (USDC selected 100% of the time regardless of
    # condition). Mutate the locally-built agent instance only -- the shared
    # YAML profile (used elsewhere, e.g. tests/test_agents.py) is untouched.
    agent.wallet.balances["USDT"] = 1000.0
    currencies = load_currency_universe()
    chains = load_chain_universe()

    candidates = [
        option
        for option in generate_candidates(agent.wallet.balances, currencies, chains)
        if option.currency_symbol in _COMPARED_CURRENCIES
    ]
    currency_profiles = {
        symbol: profile for symbol in _COMPARED_CURRENCIES if (profile := load_currency_profile(symbol)) is not None
    }
    macro = MacroState()

    return build_decision_context(
        agent.build_llm_context(),
        candidates,
        currency_profiles,
        macro,
        macro,
        TransactionContext(is_cross_border=False),
        governance_prompt_enabled=governance_prompt_enabled,
    )


def run_cell(
    model_id: str,
    governance_prompt_enabled: bool,
    client: httpx.Client,
    repository: LLMDecisionRepository | None = None,
) -> dict:
    """Runs one (model, condition) cell. Returns a plain dict rather than a
    pydantic model since this is a script-level result row, not a value
    passed between typed interfaces. If a repository is supplied, also
    persists the decision -- callers that only want the printed table (e.g.
    tests/test_experiment_007_live.py) may omit it."""
    context = _build_context(governance_prompt_enabled)
    schema_json = json.dumps(Decision.model_json_schema())
    prompt = render_prompt("buyer", context, schema_json)

    try:
        decision = call_model(prompt, model_id, client)
    except ModelCallFailedError as exc:
        return {
            "model_id": model_id,
            "governance_prompt_enabled": governance_prompt_enabled,
            "excluded": True,
            "exclusion_reason": exc.reason,
        }

    hallucination = None
    if decision.action in (DecisionAction.OFFER, DecisionAction.COUNTER_OFFER, DecisionAction.ACCEPT):
        hallucination = detect_hallucination(
            GOOD_TRUE_PRICE, decision.price, currency_symbol=decision.proposed_currency, actual_model=model_id
        )

    if repository is not None:
        repository.record(
            LLMDecisionLogEntry(
                decision_id=generate_id("dec"),
                simulation_id="experiment_007_governance_prompting",
                timestep=0,
                agent_id=context.agent.agent_id,
                agent_type=context.agent.agent_class,
                requested_model=model_id,
                actual_model=model_id,
                fallback_used=False,
                fallback_reason=None,
                model_attempts=[model_id],
                prompt_version=PROMPT_VERSIONS["buyer"],
                rendered_prompt_hash=hash_rendered_prompt(prompt),
                system_prompt=prompt,
                action=decision.action.value,
                currency=decision.proposed_currency,
                chain=decision.proposed_chain,
                amount=decision.amount,
                price=decision.price,
                reported_reasoning=decision.reasoning,
                negotiation_id=None,
                round=0,
                risk_profile=context.agent.risk_profile,
                utility_type=context.agent.utility_type,
                utility_parameters={"risk_aversion": context.agent.risk_aversion, "eis": context.agent.eis},
                scenario="experiment_007_governance_prompting",
                domestic_or_cross_border="cross_border" if context.transaction_context.is_cross_border else "domestic",
                governance_prompt_enabled=governance_prompt_enabled,
            )
        )

    return {
        "model_id": model_id,
        "governance_prompt_enabled": governance_prompt_enabled,
        "excluded": False,
        "selected_currency": decision.proposed_currency,
        "action": decision.action.value,
        "price": decision.price,
        "reported_reasoning": decision.reasoning,
        "hallucination_direction": hallucination.direction.value if hallucination else None,
    }


def _print_results_table(results: list[dict]) -> None:
    print(f"{'model':<30} {'condition':<12} {'currency':<8} {'action':<14} {'price':>8}  reasoning")
    for row in results:
        if row["excluded"]:
            print(f"{row['model_id']:<30} EXCLUDED: {row['exclusion_reason']}")
            continue
        condition = "governance" if row["governance_prompt_enabled"] else "baseline"
        print(
            f"{row['model_id']:<30} {condition:<12} {row['selected_currency']:<8} {row['action']:<14} "
            f"{row['price']:>8.2f}  {row['reported_reasoning'][:60]}"
        )

    included = [r for r in results if not r["excluded"]]
    usdc_baseline = sum(1 for r in included if not r["governance_prompt_enabled"] and r["selected_currency"] == "USDC")
    usdc_governance = sum(1 for r in included if r["governance_prompt_enabled"] and r["selected_currency"] == "USDC")
    baseline_total = sum(1 for r in included if not r["governance_prompt_enabled"])
    governance_total = sum(1 for r in included if r["governance_prompt_enabled"])
    print(
        f"\nPrimary outcome -- USDC selection rate: "
        f"baseline {usdc_baseline}/{baseline_total}, governance-emphasized {usdc_governance}/{governance_total}"
    )
    excluded_models = [r["model_id"] for r in results if r["excluded"]]
    if excluded_models:
        print(f"Excluded (no substitution): {excluded_models}")


def main() -> None:
    from database.session import create_all_tables, new_session

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set -- see .env.example")

    roster = load_model_roster()
    client = build_openrouter_client(api_key)
    pinned_models = [roster.resolve(label) for label in roster.routing_policies.model_comparison.pinned_models]

    create_all_tables()
    session = new_session()
    repository = LLMDecisionRepository(session)

    results = [
        run_cell(model_id, governance_prompt_enabled, client, repository)
        for governance_prompt_enabled in (False, True)
        for model_id in pinned_models
    ]
    session.commit()

    _print_results_table(results)


if __name__ == "__main__":
    main()
