"""Discrete-level switch search for the synthetic-coin track, per
docs/superpowers/specs/2026-08-15-synthetic-coin-track-design.md §6.

Mirrors src/economy/equivalence_framework.py's
_agent_indifference_point/cohort_indifference_points in role and shape, but
replaces the real-coin track's continuous 7-10-round binary search with a
direct discrete-level elicitation: since a synthetic dimension only ever
takes 3 (or 2) possible values, there is no need to binary-search a
continuous range -- instead, ask the switch question once per level (at
most len(comparison.levels) calls, short-circuiting as soon as the agent
says it would switch) and report the transition point among those discrete
answers. This module is intentionally self-contained: it does not import
equivalence_framework.py's _HIGHER_IS_BETTER or _real_traits, so the two
tracks' search mechanisms never share mutable state.
"""

from dataclasses import dataclass

import httpx

from src.agents.population import CARA_ELIGIBLE_ROLES, RISK_AVERSION_COHORTS
from src.llm.llm_router import call_model_for_switch
from src.llm.switch_elicitation import render_switch_prompt
from src.simulation.environment import Environment

# Whether a HIGHER varied_value makes the varied currency more attractive --
# matches equivalence_framework.py's convention exactly for the two
# overlapping field names (peg_error/gas_fee), plus the two new fields this
# track needs (governance_score/bid_ask_spread). Kept as a separate,
# module-local dict rather than importing equivalence_framework's, per this
# module's "self-contained" design.
_HIGHER_IS_BETTER: dict[str, bool] = {
    "governance_score": True,
    "bid_ask_spread": False,
    "peg_error": False,
    "gas_fee": False,
}


@dataclass(frozen=True)
class SyntheticEquivalenceComparison:
    hypothesis: str
    fixed_currency: str
    varied_currency: str
    varied_field: str  # "governance_score" | "bid_ask_spread" | "peg_error" | "gas_fee"
    levels: tuple[float, ...]  # ordered by VALUE (low to high), not attractiveness


def _real_traits(env: Environment, symbol: str) -> dict[str, float]:
    """The current values of a synthetic currency's characteristics -- the
    other (non-varied) traits shown to the agent for context, and the
    source of a fixed currency's real reference value. Unlike
    equivalence_framework._real_traits (which needs HYPOTHESIS_CHAIN_PINS to
    resolve gas_fee via a chain lookup), this track's currencies always
    carry bid_ask_spread directly as a CurrencyConfig field, and gas_fee is
    resolved via env.currency_chain_pins (set by whatever caller built this
    Environment), not a hypothesis-specific pin table this module would
    otherwise need to import.
    """
    currency = env.currencies[symbol]
    traits = {
        "governance_score": currency.governance_score,
        "bid_ask_spread": currency.bid_ask_spread,
        "peg_error": currency.peg_error,
    }
    chain_name = env.currency_chain_pins.get(symbol)
    if chain_name is not None:
        traits["gas_fee"] = env.chains[chain_name].gas_fee
    return traits


def _agent_discrete_switch_point(
    agent,
    comparison: SyntheticEquivalenceComparison,
    fixed_traits: dict[str, float],
    varied_other_traits: dict[str, float],
    client: httpx.Client,
) -> float:
    """Same role as equivalence_framework._agent_indifference_point, but asks
    the switch question at each of comparison.levels directly (one call per
    level, not a multi-round binary search), iterating from LEAST to MOST
    attractive per _HIGHER_IS_BETTER's direction, and returns the first
    level at which the agent says will_switch=True -- or the most attractive
    level tested if the agent never switches, honestly reporting that
    boundary rather than extrapolating past it.
    """
    agent_context = agent.build_llm_context()
    if agent_context.assigned_model is None:
        raise ValueError(
            "cohort_discrete_switch_points requires every CARA-eligible agent to have an "
            f"assigned_model, but agent {agent.agent_id!r} has assigned_model=None"
        )

    higher_is_better = _HIGHER_IS_BETTER[comparison.varied_field]
    # comparison.levels is ordered by VALUE (low to high); order it here by
    # ATTRACTIVENESS (least to most) so the first will_switch=True really is
    # the lowest-attractiveness threshold at which the agent would switch.
    ordered_levels = comparison.levels if higher_is_better else tuple(reversed(comparison.levels))

    for level in ordered_levels:
        prompt = render_switch_prompt(
            agent_context,
            fixed_symbol=comparison.fixed_currency,
            fixed_traits=fixed_traits,
            varied_symbol=comparison.varied_currency,
            varied_field=comparison.varied_field,
            varied_value=level,
            varied_other_traits=varied_other_traits,
        )
        decision = call_model_for_switch(prompt, agent_context.assigned_model, client)
        if decision.will_switch:
            return level

    # Never switched at any tested level -- report the most attractive
    # level actually tested, rather than extrapolating past it.
    return ordered_levels[-1]


def cohort_discrete_switch_points(
    env: Environment, comparison: SyntheticEquivalenceComparison, client: httpx.Client
) -> dict[float, float]:
    if comparison.fixed_currency not in env.currencies or comparison.varied_currency not in env.currencies:
        raise ValueError(
            f"env.currencies {sorted(env.currencies)} does not contain both of "
            f"{comparison.hypothesis}'s comparison currencies "
            f"({comparison.fixed_currency!r}, {comparison.varied_currency!r})"
        )

    fixed_traits = _real_traits(env, comparison.fixed_currency)
    if fixed_traits.get(comparison.varied_field) is None:
        raise ValueError(
            f"{comparison.hypothesis}'s varied_field {comparison.varied_field!r} is not resolvable for "
            f"fixed_currency {comparison.fixed_currency!r} -- gas_fee requires an env.currency_chain_pins "
            f"entry for that currency"
        )
    fixed_value = fixed_traits[comparison.varied_field]
    varied_traits = _real_traits(env, comparison.varied_currency)
    varied_other_traits = {
        field: value for field, value in varied_traits.items() if field != comparison.varied_field
    }

    cohort_sums: dict[float, float] = {cohort: 0.0 for cohort in RISK_AVERSION_COHORTS}
    cohort_counts: dict[float, int] = {cohort: 0 for cohort in RISK_AVERSION_COHORTS}

    for agent in env.agents.values():
        if agent.profile_name not in CARA_ELIGIBLE_ROLES:
            continue
        cohort = min(RISK_AVERSION_COHORTS, key=lambda c: abs(c - agent.risk_aversion))
        switch_point = _agent_discrete_switch_point(
            agent, comparison, fixed_traits, varied_other_traits, client
        )
        compensation = switch_point - fixed_value
        cohort_sums[cohort] += compensation
        cohort_counts[cohort] += 1

    return {
        cohort: cohort_sums[cohort] / count
        for cohort, count in cohort_counts.items()
        if count > 0
    }
