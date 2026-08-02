"""Generic, economically-neutral helper functions."""

import uuid


def generate_id(prefix: str) -> str:
    """Generate a unique identifier like 'agent-3f9c1a2b4e5d6f708192a3b4c5d6e7f8'."""
    return f"{prefix}-{uuid.uuid4().hex}"


def round_currency(value: float, ndigits: int = 6) -> float:
    """Round a monetary amount to avoid floating-point dust accumulating in ledgers."""
    return round(value, ndigits)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))
