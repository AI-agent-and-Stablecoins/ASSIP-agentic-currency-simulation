"""Tracks inequality and concentration of wealth across agents."""

from src.currencies.exchange_rates import ExchangeRateTable


def gini_coefficient(values: list[float]) -> float:
    sorted_values = sorted(v for v in values if v >= 0)
    n = len(sorted_values)
    total = sum(sorted_values)
    if n == 0 or total == 0:
        return 0.0
    weighted_sum = sum(i * v for i, v in enumerate(sorted_values, start=1))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def wealth_distribution(agent_wealths_usd: dict[str, float]) -> dict[str, float]:
    values = list(agent_wealths_usd.values())
    return {
        "gini": gini_coefficient(values),
        "total_wealth_usd": sum(values),
        "num_agents": float(len(values)),
    }


def wealth_distribution_from_agents(agents: dict, exchange_rates: ExchangeRateTable) -> dict[str, float]:
    wealths = {agent_id: agent.wallet.total_value_usd(exchange_rates) for agent_id, agent in agents.items()}
    return wealth_distribution(wealths)
