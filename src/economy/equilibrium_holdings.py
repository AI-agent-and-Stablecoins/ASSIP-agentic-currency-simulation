"""Reads a completed hypothesis-sim's final agent wallets and computes
mean %-of-wealth-per-currency by risk-aversion cohort -- the "equilibrium
holdings" table type from docs/superpowers/specs/2026-08-14-equilibrium-
holdings-design.md (e.g. H1's "50% USD, 30% Euro, 20% gold" example).
Operates on a live Environment object, not persisted database records --
no runner/persistence layer exists yet for hypothesis-sims (see that
spec's §3).
"""

from src.agents.population import CARA_ELIGIBLE_ROLES, RISK_AVERSION_COHORTS
from src.simulation.environment import Environment


def holdings_by_cohort(env: Environment) -> dict[float, dict[str, float]]:
    cohort_pct_sums: dict[float, dict[str, float]] = {cohort: {} for cohort in RISK_AVERSION_COHORTS}
    cohort_agent_counts: dict[float, int] = {cohort: 0 for cohort in RISK_AVERSION_COHORTS}

    for agent in env.agents.values():
        if agent.profile_name not in CARA_ELIGIBLE_ROLES:
            continue
        total_usd = agent.wallet.total_value_usd(env.exchange_rates)
        if total_usd <= 0:
            continue

        cohort = min(RISK_AVERSION_COHORTS, key=lambda c: abs(c - agent.risk_aversion))
        cohort_agent_counts[cohort] += 1
        for symbol in env.currencies:
            balance = agent.wallet.balances.get(symbol, 0.0)
            pct = env.exchange_rates.convert(balance, symbol, "USD") / total_usd
            cohort_pct_sums[cohort][symbol] = cohort_pct_sums[cohort].get(symbol, 0.0) + pct

    result: dict[float, dict[str, float]] = {}
    for cohort, count in cohort_agent_counts.items():
        if count == 0:
            continue
        result[cohort] = {symbol: total / count for symbol, total in cohort_pct_sums[cohort].items()}
    return result
