"""The equivalence/indifference-search framework, per
docs/superpowers/specs/2026-08-14-equivalence-framework-design.md. Every
comparison fixes one real currency's (or its assigned chain's) value at its
real config value (Y) and searches the other's corresponding value (X)
via binary search against the switch-elicitation question -- X-Y is the
reported compensation. When varied_field == "gas_fee", the modification
target is varied_currency's assigned chain (its
src.economy.hypothesis_scenarios.HYPOTHESIS_CHAIN_PINS entry), not the
currency itself -- gas_fee lives on ChainConfig, not CurrencyConfig.
"""

from dataclasses import dataclass

import httpx

from src.agents.population import CARA_ELIGIBLE_ROLES, RISK_AVERSION_COHORTS
from src.economy.hypothesis_scenarios import HYPOTHESIS_CHAIN_PINS
from src.llm.llm_router import call_model_for_switch
from src.llm.switch_elicitation import render_switch_prompt
from src.simulation.environment import Environment

_CURRENCY_TRAIT_FIELDS = ("governance_score", "liquidity_score", "peg_error")

# Whether a HIGHER varied_value makes the varied currency more attractive.
# liquidity_score/governance_score: higher is better. peg_error/gas_fee:
# lower is better -- the search direction in _agent_indifference_point must
# invert for these two fields, or the binary search converges to a bound
# instead of the agent's real threshold.
_HIGHER_IS_BETTER: dict[str, bool] = {
    "liquidity_score": True,
    "governance_score": True,
    "peg_error": False,
    "gas_fee": False,
}

# peg_error/gas_fee real-world differences (~1e-4 to ~1) are finer-grained
# relative to their [0, 0.05]/[0, 5.0] search bounds than liquidity_score/
# governance_score's differences are relative to [0.0, 1.0], so they get
# extra rounds for comparable resolution.
_SEARCH_ROUNDS_BY_FIELD: dict[str, int] = {
    "liquidity_score": 7,
    "governance_score": 7,
    "peg_error": 10,
    "gas_fee": 10,
}


@dataclass(frozen=True)
class EquivalenceComparison:
    hypothesis: str
    fixed_currency: str
    varied_currency: str
    varied_field: str
    bounds: tuple[float, float]


EQUIVALENCE_COMPARISONS: dict[str, list[EquivalenceComparison]] = {
    "H3": [EquivalenceComparison("H3", "USDT", "TDUSD", "liquidity_score", (0.0, 1.0))],
    "H4": [EquivalenceComparison("H4", "USDT", "DAI", "peg_error", (0.0, 0.05))],
    "H5": [EquivalenceComparison("H5", "USDT", "USDC", "gas_fee", (0.0, 5.0))],
    "H6": [EquivalenceComparison("H6", "EURC", "USDC", "liquidity_score", (0.0, 1.0))],
    "H7": [EquivalenceComparison("H7", "EURT", "USDC", "peg_error", (0.0, 0.05))],
    "H8": [EquivalenceComparison("H8", "EURC", "USDC", "gas_fee", (0.0, 5.0))],
    "H9": [EquivalenceComparison("H9", "TDUSD", "USDT", "peg_error", (0.0, 0.05))],
    "H10": [EquivalenceComparison("H10", "TDUSD", "USDT", "gas_fee", (0.0, 5.0))],
    "H11": [EquivalenceComparison("H11", "DAI", "TDUSD", "gas_fee", (0.0, 5.0))],
    "H2": [
        EquivalenceComparison("H2", "USDT", "EURC", "governance_score", (0.0, 1.0)),
        EquivalenceComparison("H2", "USDT", "PAXG", "governance_score", (0.0, 1.0)),
    ],
}


def _real_traits(env: Environment, symbol: str, hypothesis: str) -> dict[str, float]:
    """The real, current values of a currency's characteristics -- the
    other (non-varied) traits shown to the agent for context, and the
    source of a fixed currency's real reference value (Y)."""
    currency = env.currencies[symbol]
    traits = {field: getattr(currency, field) for field in _CURRENCY_TRAIT_FIELDS}
    chain_name = HYPOTHESIS_CHAIN_PINS.get(hypothesis, {}).get(symbol)
    if chain_name is not None:
        traits["gas_fee"] = env.chains[chain_name].gas_fee
    return traits


def _agent_indifference_point(
    agent,
    comparison: EquivalenceComparison,
    fixed_traits: dict[str, float],
    varied_other_traits: dict[str, float],
    model_id: str,
    client: httpx.Client,
) -> float:
    low, high = comparison.bounds
    higher_is_better = _HIGHER_IS_BETTER[comparison.varied_field]
    rounds = _SEARCH_ROUNDS_BY_FIELD[comparison.varied_field]
    agent_context = agent.build_llm_context()

    for _ in range(rounds):
        midpoint = (low + high) / 2
        prompt = render_switch_prompt(
            agent_context,
            fixed_symbol=comparison.fixed_currency,
            fixed_traits=fixed_traits,
            varied_symbol=comparison.varied_currency,
            varied_field=comparison.varied_field,
            varied_value=midpoint,
            varied_other_traits=varied_other_traits,
        )
        decision = call_model_for_switch(prompt, model_id, client)
        # For a higher-is-better field, "would switch" at the midpoint means
        # the agent's threshold is at or below the midpoint, so narrow the
        # upper bound. For a lower-is-better field the relationship inverts:
        # "would switch" means the midpoint is still cheap/small enough to be
        # attractive, so the threshold is at or above the midpoint instead.
        narrow_upper_bound = decision.will_switch if higher_is_better else not decision.will_switch
        if narrow_upper_bound:
            high = midpoint
        else:
            low = midpoint

    return (low + high) / 2


def cohort_indifference_points(
    env: Environment, comparison: EquivalenceComparison, model_id: str, client: httpx.Client
) -> dict[float, float]:
    if comparison.fixed_currency not in env.currencies or comparison.varied_currency not in env.currencies:
        raise ValueError(
            f"env.currencies {sorted(env.currencies)} does not contain both of "
            f"{comparison.hypothesis}'s comparison currencies "
            f"({comparison.fixed_currency!r}, {comparison.varied_currency!r})"
        )

    fixed_traits = _real_traits(env, comparison.fixed_currency, comparison.hypothesis)
    if comparison.varied_field not in fixed_traits:
        raise ValueError(
            f"{comparison.hypothesis}'s varied_field {comparison.varied_field!r} is not resolvable for "
            f"fixed_currency {comparison.fixed_currency!r} -- gas_fee requires a HYPOTHESIS_CHAIN_PINS "
            f"entry for that currency under {comparison.hypothesis!r}"
        )
    fixed_value = fixed_traits[comparison.varied_field]
    varied_traits = _real_traits(env, comparison.varied_currency, comparison.hypothesis)
    varied_other_traits = {
        field: value for field, value in varied_traits.items() if field != comparison.varied_field
    }

    cohort_sums: dict[float, float] = {cohort: 0.0 for cohort in RISK_AVERSION_COHORTS}
    cohort_counts: dict[float, int] = {cohort: 0 for cohort in RISK_AVERSION_COHORTS}

    for agent in env.agents.values():
        if agent.profile_name not in CARA_ELIGIBLE_ROLES:
            continue
        cohort = min(RISK_AVERSION_COHORTS, key=lambda c: abs(c - agent.risk_aversion))
        indifference_point = _agent_indifference_point(
            agent, comparison, fixed_traits, varied_other_traits, model_id, client
        )
        compensation = indifference_point - fixed_value
        cohort_sums[cohort] += compensation
        cohort_counts[cohort] += 1

    return {
        cohort: cohort_sums[cohort] / count
        for cohort, count in cohort_counts.items()
        if count > 0
    }
