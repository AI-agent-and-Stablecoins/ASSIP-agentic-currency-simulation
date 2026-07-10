"""Determines which agents act, and in what order, on a given day."""

import random

from src.agents.base_agent import BaseAgent


def agent_activation_order(agents: dict[str, BaseAgent], day: int, rng: random.Random) -> list[BaseAgent]:
    """Every agent acts once per day; order is shuffled with the run's seeded rng."""
    ordered = list(agents.values())
    rng.shuffle(ordered)
    return ordered
