import pytest

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.economy.income import HOME_CURRENCY_BY_ZONE, pay_income


def _consumer(currency_zone="USD"):
    agent = build_agent(load_agent_profiles()["consumer"])
    agent.currency_zone = currency_zone
    return agent


def test_home_currency_by_zone_maps_usd_and_eur():
    assert HOME_CURRENCY_BY_ZONE == {"USD": "USDC", "EUR": "EURC"}


def test_pay_income_deposits_into_usd_zone_buyers_usdc_on_payday():
    agent = _consumer("USD")
    before = agent.wallet.balances["USDC"]

    result = pay_income(agent, day=7)

    assert result == ("USDC", 250.0)
    assert agent.wallet.balances["USDC"] == pytest.approx(before + 250.0)


def test_pay_income_deposits_into_eur_zone_buyers_eurc_on_payday():
    agent = _consumer("EUR")
    before = agent.wallet.balances["EURC"]

    result = pay_income(agent, day=7)

    assert result == ("EURC", 250.0)
    assert agent.wallet.balances["EURC"] == pytest.approx(before + 250.0)


def test_pay_income_is_a_no_op_on_day_zero():
    agent = _consumer("USD")
    before = dict(agent.wallet.balances)

    result = pay_income(agent, day=0)

    assert result is None
    assert agent.wallet.balances == before


def test_pay_income_is_a_no_op_between_paydays():
    agent = _consumer("USD")
    before = dict(agent.wallet.balances)

    result = pay_income(agent, day=3)

    assert result is None
    assert agent.wallet.balances == before


def test_pay_income_fires_again_on_the_second_payday():
    agent = _consumer("USD")

    result = pay_income(agent, day=14)

    assert result == ("USDC", 250.0)


def test_pay_income_is_a_no_op_for_a_profile_without_income_configured():
    agent = build_agent(load_agent_profiles()["merchant"])

    result = pay_income(agent, day=7)

    assert result is None


def test_pay_income_is_a_no_op_for_an_unrecognized_currency_zone():
    agent = _consumer(currency_zone=None)

    result = pay_income(agent, day=7)

    assert result is None


def test_pay_income_appends_a_narrative_memory_event():
    agent = _consumer("USD")

    pay_income(agent, day=7)

    assert agent.memory.narrative_events == ["Day 7: received 250.0 USDC income."]
