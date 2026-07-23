"""Compares expected value vs paid value to quantify over/underpayment.

Pure math, no LLM dependency -- this becomes meaningful once Phase 2 LLM
agents make pricing decisions that can diverge from the market's true price.
Phase 1 rule-based agents never call this in the live simulation loop since
they compute prices deterministically.
"""

from enum import Enum

from pydantic import BaseModel


def overpayment_pct(expected: float, paid: float) -> float:
    """Positive = overpaid, negative = underpaid, 0 = paid exactly the expected value."""
    if expected <= 0:
        raise ValueError("expected value must be positive")
    return (paid - expected) / expected * 100.0


class HallucinationDirection(str, Enum):
    OVERPAYMENT = "OVERPAYMENT"
    UNDERPAYMENT = "UNDERPAYMENT"
    ACCURATE = "ACCURATE"


class HallucinationResult(BaseModel):
    expected_value: float
    paid_value: float
    absolute_error: float
    percentage_error: float
    direction: HallucinationDirection
    currency_symbol: str | None = None
    chain_name: str | None = None
    requested_model: str | None = None
    actual_model: str | None = None
    agent_type: str | None = None
    risk_profile: str | None = None
    economic_scenario: str | None = None


def detect_hallucination(
    expected_value: float,
    paid_value: float,
    hallucination_threshold: float = 0.20,
    currency_symbol: str | None = None,
    chain_name: str | None = None,
    requested_model: str | None = None,
    actual_model: str | None = None,
    agent_type: str | None = None,
    risk_profile: str | None = None,
    economic_scenario: str | None = None,
) -> HallucinationResult:
    """Classifies a settled transaction's pricing error, on top of (not
    instead of) the existing signed overpayment_pct -- see the design doc §9
    for why overpayment_pct's contract is not renegotiable."""
    signed_pct = overpayment_pct(expected_value, paid_value)
    percentage_error = abs(signed_pct)

    if percentage_error < hallucination_threshold * 100.0:
        direction = HallucinationDirection.ACCURATE
    elif signed_pct > 0:
        direction = HallucinationDirection.OVERPAYMENT
    else:
        direction = HallucinationDirection.UNDERPAYMENT

    return HallucinationResult(
        expected_value=expected_value,
        paid_value=paid_value,
        absolute_error=abs(paid_value - expected_value),
        percentage_error=percentage_error,
        direction=direction,
        currency_symbol=currency_symbol,
        chain_name=chain_name,
        requested_model=requested_model,
        actual_model=actual_model,
        agent_type=agent_type,
        risk_profile=risk_profile,
        economic_scenario=economic_scenario,
    )
