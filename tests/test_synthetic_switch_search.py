import json
import re
from unittest.mock import patch

import httpx
import pytest

from src.agents.population import CARA_ELIGIBLE_ROLES, RISK_AVERSION_COHORTS, generate_hypothesis_population
from src.currencies.stablecoin import StablecoinConfig
from src.economy.synthetic_switch_search import (
    SyntheticEquivalenceComparison,
    _agent_discrete_switch_point,
    cohort_discrete_switch_points,
)
from src.llm.llm_router import OPENROUTER_BASE_URL
from src.simulation.environment import Environment
from tests.llm_test_helpers import mock_switch_threshold_client

_MODEL_ID = "vendor/model"


def _synthetic_currency(symbol: str) -> StablecoinConfig:
    return StablecoinConfig(
        symbol=symbol,
        peg="USD",
        governance_score=1.0,
        liquidity_score=0.5,
        peg_error=0.004,
        issuer_risk=0.10,
        genius_compliant=True,
        bid_ask_spread=0.0005,
        redemption_mechanism="on-chain burn/mint",
    )


def _synthetic_test_env() -> Environment:
    currencies = {
        "SYN_FIXED": _synthetic_currency("SYN_FIXED"),
        "SYN_VARIED": _synthetic_currency("SYN_VARIED"),
    }
    population = generate_hypothesis_population(0, [_MODEL_ID], "crra")
    return Environment.build_from_population("baseline", population, currencies=currencies)


def _first_cara_agent(env: Environment):
    for agent in env.agents.values():
        if agent.profile_name in CARA_ELIGIBLE_ROLES:
            return agent
    raise AssertionError("no CARA-eligible agent in test population")


def _counting_switch_client(field: str, threshold: float, higher_is_better: bool, model_id: str = _MODEL_ID):
    """Same request/response shape as mock_switch_threshold_client, but also
    counts how many times the handler (i.e. call_model_for_switch) fires --
    used to prove the discrete search stops within len(comparison.levels)
    calls, unlike the continuous 7-10-round binary search it replaces.
    """
    pattern = re.compile(rf"Coin B \([^)]*\):[^\n]*{re.escape(field)}=(-?[\d.]+)")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        body = json.loads(request.content)
        assert body["model"] == model_id
        prompt = body["messages"][0]["content"]
        match = pattern.search(prompt)
        assert match, f"could not find {field!r} in the Coin B block of prompt:\n{prompt}"
        varied_value = float(match.group(1))
        will_switch = varied_value >= threshold if higher_is_better else varied_value <= threshold
        content = json.dumps({"will_switch": will_switch, "reasoning": "test"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    return client, calls


def test_agent_discrete_switch_point_picks_correct_level_for_higher_is_better_field():
    env = _synthetic_test_env()
    agent = _first_cara_agent(env)
    comparison = SyntheticEquivalenceComparison(
        hypothesis="TEST", fixed_currency="SYN_FIXED", varied_currency="SYN_VARIED",
        varied_field="governance_score", levels=(0.0, 1.0),
    )
    client = mock_switch_threshold_client("governance_score", threshold=0.5, higher_is_better=True)
    fixed_traits = {"governance_score": 1.0, "bid_ask_spread": 0.0005, "peg_error": 0.004}
    varied_other_traits = {"bid_ask_spread": 0.0005, "peg_error": 0.004}

    result = _agent_discrete_switch_point(agent, comparison, fixed_traits, varied_other_traits, client)

    assert result == 1.0


def test_agent_discrete_switch_point_picks_correct_level_for_lower_is_better_field():
    env = _synthetic_test_env()
    agent = _first_cara_agent(env)
    comparison = SyntheticEquivalenceComparison(
        hypothesis="TEST", fixed_currency="SYN_FIXED", varied_currency="SYN_VARIED",
        varied_field="bid_ask_spread", levels=(0.0001, 0.0005, 0.0010),
    )
    client = mock_switch_threshold_client("bid_ask_spread", threshold=0.0005, higher_is_better=False)
    fixed_traits = {"governance_score": 1.0, "bid_ask_spread": 0.0005, "peg_error": 0.004}
    varied_other_traits = {"governance_score": 1.0, "peg_error": 0.004}

    result = _agent_discrete_switch_point(agent, comparison, fixed_traits, varied_other_traits, client)

    # Least-to-most-attractive order for a lower-is-better field is
    # (0.0010, 0.0005, 0.0001); the agent first says will_switch=True at
    # 0.0005 (<= threshold), so that -- not the most extreme level -- is
    # the correct reported threshold.
    assert result == 0.0005


def test_agent_discrete_switch_point_reports_most_attractive_level_when_agent_never_switches():
    env = _synthetic_test_env()
    agent = _first_cara_agent(env)
    comparison = SyntheticEquivalenceComparison(
        hypothesis="TEST", fixed_currency="SYN_FIXED", varied_currency="SYN_VARIED",
        varied_field="bid_ask_spread", levels=(0.0001, 0.0005, 0.0010),
    )
    # threshold below every tested level -> will_switch is always False.
    client, calls = _counting_switch_client("bid_ask_spread", threshold=-1.0, higher_is_better=False)
    fixed_traits = {"governance_score": 1.0, "bid_ask_spread": 0.0005, "peg_error": 0.004}
    varied_other_traits = {"governance_score": 1.0, "peg_error": 0.004}

    result = _agent_discrete_switch_point(agent, comparison, fixed_traits, varied_other_traits, client)

    assert result == 0.0001  # the most attractive (lowest-spread) level actually tested
    assert calls["count"] == len(comparison.levels)


def test_discrete_search_never_exceeds_one_call_per_level_per_agent():
    env = _synthetic_test_env()
    agent = _first_cara_agent(env)
    comparison = SyntheticEquivalenceComparison(
        hypothesis="TEST", fixed_currency="SYN_FIXED", varied_currency="SYN_VARIED",
        varied_field="bid_ask_spread", levels=(0.0001, 0.0005, 0.0010),
    )
    client, calls = _counting_switch_client("bid_ask_spread", threshold=0.0005, higher_is_better=False)
    fixed_traits = {"governance_score": 1.0, "bid_ask_spread": 0.0005, "peg_error": 0.004}
    varied_other_traits = {"governance_score": 1.0, "peg_error": 0.004}

    _agent_discrete_switch_point(agent, comparison, fixed_traits, varied_other_traits, client)

    # ordered_levels (lower-is-better, so descending) = (0.0010, 0.0005, 0.0001).
    # threshold=0.0005 means will_switch flips to True on the SECOND call
    # (0.0005 <= 0.0005), before the third level is ever asked -- proving the
    # search genuinely short-circuits rather than always scanning every
    # level (a non-short-circuiting implementation would make 3 calls here,
    # not 2), and that the discrete search (2-3 calls) is cheaper than the
    # continuous binary search's 7-10 rounds.
    assert calls["count"] == 2


def test_cohort_mean_is_the_average_of_individual_agent_discrete_switch_points():
    env = _synthetic_test_env()
    comparison = SyntheticEquivalenceComparison(
        hypothesis="TEST", fixed_currency="SYN_FIXED", varied_currency="SYN_VARIED",
        varied_field="bid_ask_spread", levels=(0.0001, 0.0005, 0.0010),
    )
    fixed_value = env.currencies["SYN_FIXED"].bid_ask_spread

    cara_agents = [agent for agent in env.agents.values() if agent.profile_name in CARA_ELIGIBLE_ROLES]
    assert len(cara_agents) >= 2
    fake_points = {
        agent.agent_id: fixed_value + 0.0001 + 0.0002 * (index % 2) for index, agent in enumerate(cara_agents)
    }

    def fake_switch_point(agent, comparison, fixed_traits, varied_other_traits, client):
        return fake_points[agent.agent_id]

    with patch(
        "src.economy.synthetic_switch_search._agent_discrete_switch_point", side_effect=fake_switch_point
    ):
        result = cohort_discrete_switch_points(env, comparison, None)

    for cohort in result:
        members = [
            agent
            for agent in cara_agents
            if min(RISK_AVERSION_COHORTS, key=lambda c: abs(c - agent.risk_aversion)) == cohort
        ]
        expected = sum(fake_points[agent.agent_id] for agent in members) / len(members) - fixed_value
        assert result[cohort] == pytest.approx(expected)


def test_cohort_discrete_switch_points_rejects_an_env_missing_a_comparisons_currency():
    env = _synthetic_test_env()  # only SYN_FIXED/SYN_VARIED -- no SYN_OTHER
    mismatched = SyntheticEquivalenceComparison(
        hypothesis="TEST", fixed_currency="SYN_FIXED", varied_currency="SYN_OTHER",
        varied_field="bid_ask_spread", levels=(0.0001, 0.0005, 0.0010),
    )

    with pytest.raises(ValueError):
        cohort_discrete_switch_points(env, mismatched, None)
