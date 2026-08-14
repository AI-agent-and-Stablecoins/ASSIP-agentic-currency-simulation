import pytest

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.currencies.currency import load_currency_universe
from src.currencies.exchange_rates import ExchangeRateTable
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.economy.income import HOME_CURRENCY_BY_ZONE, pay_income
from src.economy.macro_state import MacroState


def _consumer(currency_zone="USD"):
    agent = build_agent(load_agent_profiles()["consumer"])
    agent.currency_zone = currency_zone
    return agent


def _real_currencies_and_rates():
    currencies = load_currency_universe()
    return currencies, ExchangeRateTable(currencies, MacroState().peg_reference_rates)


def _sandbox_currencies_and_rates(pair_key):
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[pair_key]
    currencies = {option_a.symbol: option_a, option_b.symbol: option_b}
    return currencies, ExchangeRateTable(currencies, MacroState().peg_reference_rates)


def test_home_currency_by_zone_maps_usd_and_eur():
    assert HOME_CURRENCY_BY_ZONE == {"USD": "USDC", "EUR": "EURC"}


def test_pay_income_deposits_into_usd_zone_buyers_usdc_on_payday():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()
    before = agent.wallet.balances["USDC"]

    result = pay_income(agent, 7, currencies, rates)

    assert result == {"USDC": 250.0}
    assert agent.wallet.balances["USDC"] == pytest.approx(before + 250.0)


def test_pay_income_deposits_into_eur_zone_buyers_eurc_on_payday():
    agent = _consumer("EUR")
    currencies, rates = _real_currencies_and_rates()
    before = agent.wallet.balances["EURC"]
    expected_amount = 250.0 / 1.08  # EUR peg_reference_rate, MacroState() default

    result = pay_income(agent, 7, currencies, rates)

    assert result == pytest.approx({"EURC": expected_amount})
    assert agent.wallet.balances["EURC"] == pytest.approx(before + expected_amount)


def test_pay_income_is_a_no_op_on_day_zero():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()
    before = dict(agent.wallet.balances)

    result = pay_income(agent, 0, currencies, rates)

    assert result is None
    assert agent.wallet.balances == before


def test_pay_income_is_a_no_op_between_paydays():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()
    before = dict(agent.wallet.balances)

    result = pay_income(agent, 3, currencies, rates)

    assert result is None
    assert agent.wallet.balances == before


def test_pay_income_fires_again_on_the_second_payday():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()

    result = pay_income(agent, 14, currencies, rates)

    assert result == {"USDC": 250.0}


def test_pay_income_is_a_no_op_for_a_profile_without_income_configured():
    agent = build_agent(load_agent_profiles()["merchant"])
    currencies, rates = _real_currencies_and_rates()

    result = pay_income(agent, 7, currencies, rates)

    assert result is None


def test_pay_income_is_a_no_op_for_an_unrecognized_currency_zone():
    agent = _consumer(currency_zone=None)
    currencies, rates = _real_currencies_and_rates()

    result = pay_income(agent, 7, currencies, rates)

    assert result is None


def test_pay_income_appends_a_narrative_memory_event():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()

    pay_income(agent, 7, currencies, rates)

    assert agent.memory.narrative_events == ["Day 7: received 250.0 USDC income."]


def test_pay_income_splits_evenly_across_a_same_peg_sandbox_pair_matching_the_buyers_zone():
    agent = _consumer("USD")
    currencies, rates = _sandbox_currencies_and_rates("liquidity_vs_governance")

    result = pay_income(agent, 7, currencies, rates)

    assert set(result.keys()) == {"SBX1_HILIQ_LOGOV", "SBX1_HIGOV_LOLIQ"}
    assert result["SBX1_HILIQ_LOGOV"] == pytest.approx(125.0)
    assert result["SBX1_HIGOV_LOLIQ"] == pytest.approx(125.0)
    assert agent.wallet.balances["SBX1_HILIQ_LOGOV"] == pytest.approx(125.0)
    assert agent.wallet.balances["SBX1_HIGOV_LOLIQ"] == pytest.approx(125.0)


def test_pay_income_splits_across_the_whole_pair_when_no_currency_matches_the_buyers_zone():
    agent = _consumer("EUR")
    currencies, rates = _sandbox_currencies_and_rates("liquidity_vs_governance")

    result = pay_income(agent, 7, currencies, rates)

    assert set(result.keys()) == {"SBX1_HILIQ_LOGOV", "SBX1_HIGOV_LOLIQ"}
    assert result["SBX1_HILIQ_LOGOV"] == pytest.approx(125.0)
    assert result["SBX1_HIGOV_LOLIQ"] == pytest.approx(125.0)


def test_pay_income_pays_only_the_zone_matching_side_of_an_asset_backing_pair():
    agent = _consumer("USD")
    currencies, rates = _sandbox_currencies_and_rates("asset_backing_vs_liquidity")

    result = pay_income(agent, 7, currencies, rates)

    assert result == {"SBX4_STABLE_HILIQ": pytest.approx(250.0)}
    assert "SBX4_GOLD_LOLIQ" not in agent.wallet.balances


def test_pay_income_narrative_describes_a_multi_currency_split():
    agent = _consumer("USD")
    currencies, rates = _sandbox_currencies_and_rates("liquidity_vs_governance")

    pay_income(agent, 7, currencies, rates)

    narrative = agent.memory.narrative_events[0]
    assert "SBX1_HILIQ_LOGOV" in narrative
    assert "SBX1_HIGOV_LOLIQ" in narrative
    assert " + " in narrative
