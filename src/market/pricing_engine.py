"""Determines the ground-truth price of a good.

This is critical because hallucination detection (src/llm/hallucination_detector.py)
depends on knowing the actual value a paid price is compared against. Phase 1
pricing is fully deterministic -- no LLM valuation error is introduced here.
"""

from src.market.goods import Good
from src.market.supply_demand import adjust_price


def true_price(good: Good, supply: float = 1.0, demand: float = 1.0) -> float:
    return adjust_price(good.base_price_usd, supply, demand)
