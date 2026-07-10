"""Primary utility function: weighted combination of governance, liquidity,
gas fees, peg volatility, and compliance. Weights come from agent profile
config (configs/agent_profiles/*.yaml) -- never hardcoded, so experiments
can vary them per agent.
"""

from pydantic import BaseModel, Field

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction


class MultiAttributeWeights(BaseModel):
    governance_weight: float = 0.25
    liquidity_weight: float = 0.25
    gas_fee_weight: float = Field(default=0.2)
    volatility_weight: float = Field(default=0.2)
    compliance_weight: float = Field(default=0.1)


class MultiAttributeUtility(UtilityFunction):
    def __init__(self, weights: MultiAttributeWeights):
        self.weights = weights

    def evaluate(self, option: CurrencyChainOption, **kwargs: float) -> float:
        w = self.weights
        return (
            w.governance_weight * option.governance_score
            + w.liquidity_weight * option.liquidity_score
            - w.gas_fee_weight * option.gas_fee
            - w.volatility_weight * option.peg_error
            + w.compliance_weight * (1.0 if option.genius_compliant else 0.0)
        )
