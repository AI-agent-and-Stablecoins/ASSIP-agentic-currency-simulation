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

_SEARCH_ROUNDS = 7


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


def _fixed_value(env: Environment, comparison: EquivalenceComparison) -> float:
    currency = env.currencies[comparison.fixed_currency]
    if comparison.varied_field == "gas_fee":
        chain_name = HYPOTHESIS_CHAIN_PINS[comparison.hypothesis][comparison.fixed_currency]
        return env.chains[chain_name].gas_fee
    return getattr(currency, comparison.varied_field)


def _agent_indifference_point(
    agent, env: Environment, comparison: EquivalenceComparison, fixed_value: float, model_id: str, client: httpx.Client
) -> float:
    low, high = comparison.bounds
    agent_context = agent.build_llm_context()

    for _ in range(_SEARCH_ROUNDS):
        midpoint = (low + high) / 2
        prompt = render_switch_prompt(
            agent_context,
            fixed_symbol=comparison.fixed_currency,
            fixed_field=comparison.varied_field,
            fixed_value=fixed_value,
            varied_symbol=comparison.varied_currency,
            varied_field=comparison.varied_field,
            varied_value=midpoint,
        )
        decision = call_model_for_switch(prompt, model_id, client)
        if decision.will_switch:
            high = midpoint
        else:
            low = midpoint

    return (low + high) / 2


def cohort_indifference_points(
    env: Environment, comparison: EquivalenceComparison, model_id: str, client: httpx.Client
) -> dict[float, float]:
    fixed_value = _fixed_value(env, comparison)

    cohort_sums: dict[float, float] = {cohort: 0.0 for cohort in RISK_AVERSION_COHORTS}
    cohort_counts: dict[float, int] = {cohort: 0 for cohort in RISK_AVERSION_COHORTS}

    for agent in env.agents.values():
        if agent.profile_name not in CARA_ELIGIBLE_ROLES:
            continue
        cohort = min(RISK_AVERSION_COHORTS, key=lambda c: abs(c - agent.risk_aversion))
        indifference_point = _agent_indifference_point(agent, env, comparison, fixed_value, model_id, client)
        compensation = indifference_point - fixed_value
        cohort_sums[cohort] += compensation
        cohort_counts[cohort] += 1

    return {
        cohort: cohort_sums[cohort] / count
        for cohort, count in cohort_counts.items()
        if count > 0
    }
