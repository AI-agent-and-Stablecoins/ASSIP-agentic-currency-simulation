from unittest.mock import patch

import pytest

from src.agents.population import CARA_ELIGIBLE_ROLES, RISK_AVERSION_COHORTS, generate_hypothesis_population
from src.currencies.currency import load_currency_universe
from src.economy.equivalence_framework import (
    EQUIVALENCE_COMPARISONS,
    EquivalenceComparison,
    cohort_indifference_points,
)
from src.economy.hypothesis_scenarios import HYPOTHESIS_CURRENCIES
from src.economy.macro_state import MacroState
from src.economy.wallet_seeding import seed_restricted_wallets
from src.simulation.environment import Environment
from tests.llm_test_helpers import mock_switch_threshold_client


def test_h2_has_exactly_two_comparisons():
    assert len(EQUIVALENCE_COMPARISONS["H2"]) == 2
    varied = {c.varied_currency for c in EQUIVALENCE_COMPARISONS["H2"]}
    assert varied == {"EURC", "PAXG"}


def test_every_other_hypothesis_has_exactly_one_comparison():
    for hypothesis in ("H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"):
        assert len(EQUIVALENCE_COMPARISONS[hypothesis]) == 1


def test_h1_has_no_comparisons():
    assert "H1" not in EQUIVALENCE_COMPARISONS


def test_h3_fixes_usdt_liquidity_and_varies_tdusd_liquidity():
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    assert comparison.fixed_currency == "USDT"
    assert comparison.varied_currency == "TDUSD"
    assert comparison.varied_field == "liquidity_score"
    assert comparison.bounds == (0.0, 1.0)


def test_gas_fee_comparisons_have_the_gas_fee_bounds():
    for hypothesis in ("H5", "H8", "H10", "H11"):
        comparison = EQUIVALENCE_COMPARISONS[hypothesis][0]
        assert comparison.varied_field == "gas_fee"
        assert comparison.bounds == (0.0, 5.0)


def test_h2_comparisons_vary_governance_score():
    for comparison in EQUIVALENCE_COMPARISONS["H2"]:
        assert comparison.varied_field == "governance_score"
        assert comparison.fixed_currency == "USDT"


def _hypothesis_env(hypothesis):
    real_currencies = load_currency_universe()
    restricted = {symbol: real_currencies[symbol] for symbol in HYPOTHESIS_CURRENCIES[hypothesis]}
    population = generate_hypothesis_population(0, ["vendor/model"], "crra")
    env = Environment.build_from_population("baseline", population, currencies=restricted)
    seed_restricted_wallets(env.agents, restricted, real_currencies, MacroState().peg_reference_rates)
    return env


def test_binary_search_converges_to_a_known_threshold_for_a_higher_is_better_field():
    # H3 varies liquidity_score, a higher-is-better field: the search must
    # converge toward the true threshold, not saturate to a bound, which is
    # exactly the bug a constant-response mock would fail to catch.
    env = _hypothesis_env("H3")
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    threshold = 0.62
    client = mock_switch_threshold_client("liquidity_score", threshold, higher_is_better=True)

    result = cohort_indifference_points(env, comparison, "vendor/model", client)

    fixed_value = load_currency_universe()[comparison.fixed_currency].liquidity_score
    expected = threshold - fixed_value
    assert set(result.keys()) <= {0.0, 2.0, 4.0, 6.0}
    for value in result.values():
        assert value == pytest.approx(expected, abs=0.01)


def test_binary_search_converges_to_a_known_threshold_for_a_lower_is_better_field():
    # H10 varies gas_fee, a lower-is-better field, where the pre-fix
    # direction logic was inverted and always saturated to a bound
    # regardless of the agent's real threshold.
    env = _hypothesis_env("H10")
    comparison = EQUIVALENCE_COMPARISONS["H10"][0]
    threshold = 2.5
    client = mock_switch_threshold_client("gas_fee", threshold, higher_is_better=False)

    result = cohort_indifference_points(env, comparison, "vendor/model", client)

    fixed_value = env.chains["solana"].gas_fee  # H10's fixed reference: TDUSD on Solana
    expected = threshold - fixed_value
    assert set(result.keys()) <= {0.0, 2.0, 4.0, 6.0}
    for value in result.values():
        assert value == pytest.approx(expected, abs=0.05)


def test_cohort_mean_is_the_average_of_individual_agent_indifference_points():
    env = _hypothesis_env("H3")
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    fixed_value = load_currency_universe()[comparison.fixed_currency].liquidity_score

    cara_agents = [agent for agent in env.agents.values() if agent.profile_name in CARA_ELIGIBLE_ROLES]
    assert len(cara_agents) >= 2
    fake_points = {
        agent.agent_id: fixed_value + 0.1 + 0.2 * (index % 2) for index, agent in enumerate(cara_agents)
    }

    def fake_indifference_point(agent, comparison, fixed_traits, varied_other_traits, model_id, client):
        return fake_points[agent.agent_id]

    with patch(
        "src.economy.equivalence_framework._agent_indifference_point", side_effect=fake_indifference_point
    ):
        result = cohort_indifference_points(env, comparison, "vendor/model", None)

    for cohort in result:
        members = [
            agent
            for agent in cara_agents
            if min(RISK_AVERSION_COHORTS, key=lambda c: abs(c - agent.risk_aversion)) == cohort
        ]
        expected = sum(fake_points[agent.agent_id] for agent in members) / len(members) - fixed_value
        assert result[cohort] == pytest.approx(expected)


def test_cohort_indifference_points_rejects_an_env_missing_a_comparisons_currency():
    env = _hypothesis_env("H3")  # restricted to TDUSD/USDT only -- no DAI
    mismatched = EquivalenceComparison("H4", "USDT", "DAI", "peg_error", (0.0, 0.05))

    with pytest.raises(ValueError):
        cohort_indifference_points(env, mismatched, "vendor/model", None)
