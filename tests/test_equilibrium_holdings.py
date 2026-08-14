import random

import pytest

from src.agents.population import generate_hypothesis_population
from src.currencies.currency import load_currency_universe
from src.economy.equilibrium_holdings import holdings_by_cohort
from src.economy.hypothesis_scenarios import HYPOTHESIS_CURRENCIES
from src.economy.macro_state import MacroState
from src.economy.wallet_seeding import seed_restricted_wallets
from src.simulation.environment import Environment
from src.simulation.timestep import run_timestep


def _h1_env(utility_type="crra", seed=0):
    real_currencies = load_currency_universe()
    restricted = {symbol: real_currencies[symbol] for symbol in HYPOTHESIS_CURRENCIES["H1"]}
    population = generate_hypothesis_population(seed, ["vendor/model"], utility_type)
    env = Environment.build_from_population("baseline", population, currencies=restricted)
    seed_restricted_wallets(env.agents, restricted, real_currencies, MacroState().peg_reference_rates)
    return env


def test_freshly_seeded_wallets_split_evenly_across_h1s_three_currencies():
    env = _h1_env()

    result = holdings_by_cohort(env)

    assert set(result.keys()) == {0.0, 2.0, 4.0, 6.0}
    for cohort_pcts in result.values():
        assert set(cohort_pcts.keys()) == {"USDC", "EURC", "PAXG"}
        for pct in cohort_pcts.values():
            assert pct == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_computes_the_correct_arithmetic_mean_across_a_cohort():
    env = _h1_env()
    cohort_agents = [
        a for a in env.agents.values()
        if a.profile_name in ("consumer", "bank", "investor") and a.risk_aversion == 0.0
    ]
    assert len(cohort_agents) >= 2
    cohort_agents[0].wallet.balances = {"USDC": 100.0, "EURC": 0.0, "PAXG": 0.0}
    cohort_agents[1].wallet.balances = {"USDC": 0.0, "EURC": 100.0, "PAXG": 0.0}
    for agent in cohort_agents[2:]:
        agent.wallet.balances = {"USDC": 0.0, "EURC": 100.0, "PAXG": 0.0}

    result = holdings_by_cohort(env)

    n = len(cohort_agents)
    assert result[0.0]["USDC"] == pytest.approx(1.0 / n)
    assert result[0.0]["EURC"] == pytest.approx((n - 1) / n)
    assert result[0.0]["PAXG"] == pytest.approx(0.0)


def test_cara_zero_substitute_buckets_into_the_0_0_cohort_key():
    env = _h1_env(utility_type="cara")

    result = holdings_by_cohort(env)

    assert 0.0 in result
    assert 1e-4 not in result


def test_a_bankrupt_agent_is_excluded_from_its_cohorts_average():
    env = _h1_env()
    cohort_agents = [
        a for a in env.agents.values()
        if a.profile_name in ("consumer", "bank", "investor") and a.risk_aversion == 0.0
    ]
    bankrupt = cohort_agents[0]
    bankrupt.wallet.balances = {"USDC": 0.0, "EURC": 0.0, "PAXG": 0.0}
    for agent in cohort_agents[1:]:
        agent.wallet.balances = {"USDC": 100.0, "EURC": 0.0, "PAXG": 0.0}

    result = holdings_by_cohort(env)

    assert result[0.0]["USDC"] == pytest.approx(1.0)


def test_h1_end_to_end_percentages_sum_to_one_per_cohort():
    """Plumbing test: proves the real 3-currency H1 restriction, real
    population, real wallet seeding, and a real (deterministic-path,
    fast) run_timestep loop all compose correctly through
    holdings_by_cohort. Uses the deterministic path purely for test
    speed -- per docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-
    pivot-design.md's binding decision, a REAL hypothesis-sim measuring
    genuine cohort differentiation must use use_llm=True; this test
    only proves the measurement math/plumbing, not cohort behavior."""
    env = _h1_env()
    rng = random.Random(0)
    for day in range(5):
        run_timestep(env, day, rng)

    result = holdings_by_cohort(env)

    assert len(result) > 0
    for cohort_pcts in result.values():
        assert sum(cohort_pcts.values()) == pytest.approx(1.0, rel=1e-6)
