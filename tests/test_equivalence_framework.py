import pytest

from src.agents.population import generate_hypothesis_population
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
from tests.llm_test_helpers import mock_openrouter_client


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


def _h3_env():
    real_currencies = load_currency_universe()
    restricted = {symbol: real_currencies[symbol] for symbol in HYPOTHESIS_CURRENCIES["H3"]}
    population = generate_hypothesis_population(0, ["vendor/model"], "crra")
    env = Environment.build_from_population("baseline", population, currencies=restricted)
    seed_restricted_wallets(env.agents, restricted, real_currencies, MacroState().peg_reference_rates)
    return env


def test_binary_search_converges_toward_a_known_threshold():
    env = _h3_env()
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    # Every agent always answers "will_switch: True" -- each round narrows
    # `high` toward `low`, so the search converges near bounds[0].
    client = mock_openrouter_client({"vendor/model": {"will_switch": True, "reasoning": "test"}})

    result = cohort_indifference_points(env, comparison, "vendor/model", client)

    assert set(result.keys()) <= {0.0, 2.0, 4.0, 6.0}
    # cohort_indifference_points reports (indifference_point - fixed_value);
    # fixed_value here is USDT's real liquidity_score (0.98), and the
    # indifference point converges near bounds[0]=0.0, so the result is
    # a large negative number.
    for value in result.values():
        assert value < -0.9


def test_cohort_mean_is_the_average_of_individual_agent_indifference_points():
    from src.currencies.currency import load_currency_universe

    env = _h3_env()
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    fixed_value = load_currency_universe()[comparison.fixed_currency].liquidity_score
    client = mock_openrouter_client({"vendor/model": {"will_switch": False, "reasoning": "test"}})

    result = cohort_indifference_points(env, comparison, "vendor/model", client)

    # Every agent answering "no" the whole search pushes every trial toward
    # the upper bound -- the final per-agent indifference point converges
    # near bounds[1], so the reported (indifference_point - fixed_value)
    # converges near (bounds[1] - fixed_value).
    expected = comparison.bounds[1] - fixed_value
    for value in result.values():
        assert value == pytest.approx(expected, abs=0.05)
