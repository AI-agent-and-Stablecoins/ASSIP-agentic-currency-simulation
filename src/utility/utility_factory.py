"""Assigns a concrete utility function from an agent profile's declared type.

Takes plain scalar/weights arguments rather than an AgentProfileConfig object
so src/utility has no dependency on src/agents -- agents call into utility,
not the reverse.
"""

from src.utility.base import UtilityFunction
from src.utility.cara import CARAUtility
from src.utility.crra import CRRAUtility
from src.utility.epstein_zin import EpsteinZinProxyUtility
from src.utility.multi_attribute import MultiAttributeUtility, MultiAttributeWeights
from src.utility.risk_neutral import RiskNeutralUtility


def build_utility_function(
    utility_type: str,
    risk_aversion: float | None = None,
    weights: MultiAttributeWeights | None = None,
    eis: float | None = None,
) -> UtilityFunction:
    if utility_type == "crra":
        if risk_aversion is None:
            raise ValueError("CRRA utility requires risk_aversion")
        return CRRAUtility(risk_aversion)
    if utility_type == "cara":
        if risk_aversion is None:
            raise ValueError("CARA utility requires risk_aversion")
        return CARAUtility(risk_aversion)
    if utility_type == "multi_attribute":
        return MultiAttributeUtility(weights or MultiAttributeWeights())
    if utility_type == "risk_neutral":
        return RiskNeutralUtility()
    if utility_type == "epstein_zin_proxy":
        if risk_aversion is None:
            raise ValueError("epstein_zin_proxy utility requires risk_aversion")
        if eis is None:
            raise ValueError("epstein_zin_proxy utility requires eis")
        return EpsteinZinProxyUtility(risk_aversion, eis)
    raise ValueError(f"Unknown utility_type: {utility_type}")
