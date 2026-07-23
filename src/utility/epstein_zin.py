"""EZ-inspired static utility proxy.

This is not an Epstein-Zin utility function and should not be interpreted as
one in empirical results. It is an EZ-inspired static proxy designed to test
whether separately parameterizing risk aversion and an EIS-inspired
fee-sensitivity parameter changes choice behavior relative to CRRA. True
Epstein-Zin utility is recursive over a consumption stream and requires a
continuation value from future timesteps; this simulation's decision is a
single-period per-transaction choice, so no continuation value exists here.

`risk_aversion` (gamma) shapes curvature over the safety-adjusted payoff --
the same role it plays in CRRA. The constructor parameter `eis` (psi) is kept
under that name for compatibility with project_instructions.md's CRRA/CARA/EZ
framing, but the derived quantity `1 / eis` is named
`eis_inspired_fee_sensitivity` below, not `fee_sensitivity` alone: using
psi to scale a gas-fee penalty is a behavioral modeling choice, not a
derivation from formal EZ preferences over consumption.
"""

import math

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction


class EpsteinZinProxyUtility(UtilityFunction):
    def __init__(self, risk_aversion: float, eis: float):
        if eis <= 0:
            raise ValueError("eis must be positive for the EZ-inspired proxy")
        self.risk_aversion = risk_aversion
        self.eis = eis

    def evaluate(self, option: CurrencyChainOption, wealth: float = 1.0, **kwargs: float) -> float:
        safety = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
        eis_inspired_fee_sensitivity = 1.0 / self.eis
        effective_wealth = max(wealth * safety - option.gas_fee * eis_inspired_fee_sensitivity, 1e-9)

        gamma = self.risk_aversion
        if abs(gamma - 1.0) < 1e-9:
            return math.log(effective_wealth)
        return (effective_wealth ** (1 - gamma)) / (1 - gamma)
