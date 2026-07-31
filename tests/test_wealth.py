import pytest

from src.agents.wallet import Wallet
from src.agents.wealth import advance_price_index, real_purchasing_power
from src.simulation.environment import Environment


def _build_env() -> Environment:
    return Environment.build("baseline", {"consumer": 2, "merchant": 2})


def test_real_purchasing_power_divides_nominal_value_by_price_index():
    env = _build_env()
    wallet = Wallet(balances={"USDC": 1000.0})

    result = real_purchasing_power(wallet, env.exchange_rates, price_index=1.0)

    assert result == pytest.approx(wallet.total_value_usd(env.exchange_rates))


def test_real_purchasing_power_shrinks_as_price_index_rises():
    env = _build_env()
    wallet = Wallet(balances={"USDC": 1000.0})

    at_baseline = real_purchasing_power(wallet, env.exchange_rates, price_index=1.0)
    after_inflation = real_purchasing_power(wallet, env.exchange_rates, price_index=1.1)

    assert after_inflation < at_baseline


def test_advance_price_index_compounds_daily_inflation():
    index = advance_price_index(1.0, inflation_rate=0.02)
    assert index == pytest.approx(1.02)

    index_2 = advance_price_index(index, inflation_rate=0.02)
    assert index_2 == pytest.approx(1.02 * 1.02)
