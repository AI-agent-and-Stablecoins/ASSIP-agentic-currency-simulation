import pytest

from src.agents.wallet import Wallet
from src.agents.wealth import advance_price_index, real_purchasing_power
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
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


def test_advance_price_index_converts_annual_rate_to_daily_equivalent():
    """`inflation_rate` in scenario configs is an ANNUAL rate. One day's
    compounding must apply the daily-equivalent rate
    ((1 + annual)**(1/365) - 1), not the annual rate directly -- otherwise
    a 2%/year "baseline" scenario compounds to ~1349x over a 365-day run.
    """
    index = advance_price_index(1.0, annual_inflation_rate=0.02)
    daily_rate = (1.02) ** (1 / 365) - 1
    assert index == pytest.approx(1.0 + daily_rate)
    assert index == pytest.approx(1.0000542, rel=1e-4)

    index_2 = advance_price_index(index, annual_inflation_rate=0.02)
    assert index_2 == pytest.approx((1 + daily_rate) ** 2)


def test_advance_price_index_365_daily_compoundings_reconstruct_annual_rate():
    """Compounding the daily-equivalent rate 365 times should return
    (approximately) to the original annual rate, proving the annual ->
    daily conversion is correct."""
    index = 1.0
    for _ in range(365):
        index = advance_price_index(index, annual_inflation_rate=0.02)

    assert index == pytest.approx(1.02, rel=1e-6)


@pytest.mark.parametrize("pair_name", list(SANDBOX_CURRENCY_PAIRS.keys()))
def test_real_purchasing_power_does_not_crash_for_sandbox_currency_wallets(pair_name):
    """Task 11 review Fix 4: the 6 factor-isolation sandboxes (Task 10) hold
    synthetic CurrencyConfig pairs with made-up symbols (e.g.
    SBX1_HILIQ_LOGOV, never a real-universe symbol). Traced
    ExchangeRateTable._peg_value (src/currencies/exchange_rates.py):
    it resolves a wallet's currency symbol to a peg reference rate via
    `currency.peg` ("USD" or "XAU" for every sandbox currency), never the
    raw synthetic symbol itself -- and MacroState's default
    peg_reference_rates already has "USD"/"EUR"/"XAU" entries (see
    src/economy/macro_state.py). So real_purchasing_power -- and therefore
    persist_full_timestep, which calls it once per agent every day --
    does NOT raise KeyError for a sandbox-currency wallet. This was a
    review false-positive for the peg-lookup path specifically; the dead
    `if "ethereum"/"solana" in env.chains` guards flagged alongside it were
    removed in database/repository.py instead, since they protected
    against nothing real (env.chains is always the full universe).
    """
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[pair_name]
    currencies = {option_a.symbol: option_a, option_b.symbol: option_b}
    env = Environment.build_from_population("baseline", [], currencies=currencies)
    wallet = Wallet(balances={option_a.symbol: 500.0})

    value = real_purchasing_power(wallet, env.exchange_rates, env.price_index)

    expected = env.exchange_rates.convert(500.0, option_a.symbol, "USD") / env.price_index
    assert value == pytest.approx(expected)
    assert value > 0.0
