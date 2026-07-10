"""Issuer risk scoring based on issuer size and track record."""

from src.utils.helpers import clamp


def issuer_risk_score(issuer_size_usd_billions: float, track_record_years: float) -> float:
    """Larger, longer-tenured issuers score lower risk. Returns a value in [0, 1]."""
    size_component = 1.0 / (1.0 + issuer_size_usd_billions / 10.0)
    tenure_component = 1.0 / (1.0 + track_record_years / 5.0)
    return clamp(0.5 * size_component + 0.5 * tenure_component, 0.0, 1.0)
