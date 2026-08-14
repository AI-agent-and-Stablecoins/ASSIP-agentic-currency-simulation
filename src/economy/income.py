"""Weekly income for buyer agents: without this, buyers only ever spend from
their fixed initial_wallet and the economy permanently runs dry a few days
into any multi-week run (see docs/superpowers/specs/2026-08-14-buyer-income-
mechanism-design.md). Only buyer profiles that opt in via
income_per_period/income_period_days in configs/agent_profiles/*.yaml are
paid; every other role's fields stay None, so this is a no-op for them.

Payment currency resolution (see the spec's §7 amendment): the master/real
cell always pays into the buyer's exact home currency (USDC/EURC). Sandbox
cells restrict the environment's currency universe to 2 synthetic symbols
that often share the same peg (see src/currencies/sandbox_currencies.py) --
paying only one side of such a pair would bias the very comparison the
sandbox exists to run, so income there splits evenly by USD value across
every zone-matching currency, or the whole universe if none match, mirroring
src/simulation/matrix_runner.py's _seed_sandbox_wallets.
"""

from src.agents.base_agent import BaseAgent
from src.currencies.currency import CurrencyConfig
from src.currencies.exchange_rates import ExchangeRateTable
from src.economy.fx_tax import currency_zone_of

HOME_CURRENCY_BY_ZONE = {"USD": "USDC", "EUR": "EURC"}


def pay_income(
    agent: BaseAgent,
    day: int,
    currencies: dict[str, CurrencyConfig],
    exchange_rates: ExchangeRateTable,
) -> dict[str, float] | None:
    if agent.income_per_period is None or agent.income_period_days is None:
        return None
    if day == 0 or day % agent.income_period_days != 0:
        return None

    home_symbol = HOME_CURRENCY_BY_ZONE.get(agent.currency_zone)
    if home_symbol is None:
        return None

    if home_symbol in currencies:
        targets = [home_symbol]
    else:
        targets = [
            symbol for symbol, currency in currencies.items() if currency_zone_of(currency) == agent.currency_zone
        ] or list(currencies.keys())

    share_usd = agent.income_per_period / len(targets)
    paid: dict[str, float] = {}
    for symbol in targets:
        amount = exchange_rates.convert(share_usd, "USD", symbol)
        agent.wallet.deposit(symbol, amount)
        paid[symbol] = amount

    paid_desc = " + ".join(f"{amount} {symbol}" for symbol, amount in paid.items())
    agent.memory.record_narrative(f"Day {day}: received {paid_desc} income.")
    return paid
