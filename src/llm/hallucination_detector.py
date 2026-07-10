"""Compares expected value vs paid value to quantify over/underpayment.

Pure math, no LLM dependency -- this becomes meaningful once Phase 2 LLM
agents make pricing decisions that can diverge from the market's true price.
Phase 1 rule-based agents never call this in the live simulation loop since
they compute prices deterministically.
"""


def overpayment_pct(expected: float, paid: float) -> float:
    """Positive = overpaid, negative = underpaid, 0 = paid exactly the expected value."""
    if expected <= 0:
        raise ValueError("expected value must be positive")
    return (paid - expected) / expected * 100.0
