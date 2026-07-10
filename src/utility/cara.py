"""Constant Absolute Risk Aversion utility.

Useful for institutional agents, banks, and large merchants whose risk
aversion doesn't scale with their wealth the way CRRA assumes.
"""

import math

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction


class CARAUtility(UtilityFunction):
    def __init__(self, risk_aversion: float):
        if risk_aversion == 0:
            raise ValueError("risk_aversion must be nonzero for CARA utility")
        self.risk_aversion = risk_aversion

    def evaluate(self, option: CurrencyChainOption, wealth: float = 1.0, **kwargs: float) -> float:
        safety_multiplier = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
        effective_wealth = wealth * safety_multiplier - option.gas_fee
        a = self.risk_aversion
        return -math.exp(-a * effective_wealth) / a
