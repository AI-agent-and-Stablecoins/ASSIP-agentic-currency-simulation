"""Risk-neutral utility: the un-shaped baseline the other utility functions
(CRRA, CARA, the EZ-inspired proxy) are measured against.

Deliberately linear -- no curvature, no risk preference. Evaluates raw net
economic payoff (safety-adjusted wealth minus gas fee) so that any deviation
CRRA/CARA/EZ show from this baseline is attributable to their risk-shaping,
not to a difference in what "payoff" means.
"""

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction


class RiskNeutralUtility(UtilityFunction):
    def evaluate(self, option: CurrencyChainOption, wealth: float = 1.0, **kwargs: float) -> float:
        safety_multiplier = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
        return wealth * safety_multiplier - option.gas_fee
