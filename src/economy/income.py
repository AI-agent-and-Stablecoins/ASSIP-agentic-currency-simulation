"""Weekly income for buyer agents: without this, buyers only ever spend from
their fixed initial_wallet and the economy permanently runs dry a few days
into any multi-week run (see docs/superpowers/specs/2026-08-14-buyer-income-
mechanism-design.md). Only buyer profiles that opt in via
income_per_period/income_period_days in configs/agent_profiles/*.yaml are
paid; every other role's fields stay None, so this is a no-op for them.
"""

from src.agents.base_agent import BaseAgent

HOME_CURRENCY_BY_ZONE = {"USD": "USDC", "EUR": "EURC"}


def pay_income(agent: BaseAgent, day: int) -> tuple[str, float] | None:
    if agent.income_per_period is None or agent.income_period_days is None:
        return None
    if day == 0 or day % agent.income_period_days != 0:
        return None
    currency = HOME_CURRENCY_BY_ZONE.get(agent.currency_zone)
    if currency is None:
        return None

    agent.wallet.deposit(currency, agent.income_per_period)
    agent.memory.record_narrative(f"Day {day}: received {agent.income_per_period} {currency} income.")
    return (currency, agent.income_per_period)
