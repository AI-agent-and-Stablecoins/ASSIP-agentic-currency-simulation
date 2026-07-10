"""Constant Relative Risk Aversion utility.

Used for hypotheses like "more risk-averse agents prefer USD over Euro
stablecoins." CRRA is defined over wealth; an option's governance/liquidity
quality and peg stability are treated as an effective-wealth multiplier, so
higher risk_aversion increasingly rewards safer options.
"""

import math

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction


class CRRAUtility(UtilityFunction):
    def __init__(self, risk_aversion: float):
        self.risk_aversion = risk_aversion

    def evaluate(self, option: CurrencyChainOption, wealth: float = 1.0, **kwargs: float) -> float:
        safety_multiplier = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
        effective_wealth = max(wealth * safety_multiplier, 1e-9)
        gamma = self.risk_aversion
        if abs(gamma - 1.0) < 1e-9:
            return math.log(effective_wealth)
        return (effective_wealth ** (1 - gamma)) / (1 - gamma)
