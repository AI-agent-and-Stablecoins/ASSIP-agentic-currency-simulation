"""Combines reserves, transparency, compliance, and issuer risk into one governance score.

configs/currencies/*.yaml carry a pre-computed governance_score used directly
by agent utility functions. This module exists so that score can be derived
and cross-checked from its underlying components, and so Phase 3 shocks can
recompute governance scores dynamically when reserve composition or
compliance status changes.
"""

from pydantic import BaseModel

from src.governance.compliance import ComplianceStatus
from src.governance.reserve_models import ReserveComposition
from src.governance.transparency import TransparencyRating, transparency_score

_RESERVE_ASSET_QUALITY: dict[str, float] = {
    "treasuries": 1.0,
    "cash": 1.0,
    "commercial_paper": 0.7,
    "gold": 0.8,
    "bank_deposits": 0.6,
}


class GovernanceWeights(BaseModel):
    reserve_quality_weight: float = 0.4
    transparency_weight: float = 0.3
    compliance_weight: float = 0.2
    issuer_risk_weight: float = 0.1


def reserve_quality_score(reserves: ReserveComposition) -> float:
    return (
        reserves.treasuries * _RESERVE_ASSET_QUALITY["treasuries"]
        + reserves.cash * _RESERVE_ASSET_QUALITY["cash"]
        + reserves.commercial_paper * _RESERVE_ASSET_QUALITY["commercial_paper"]
        + reserves.gold * _RESERVE_ASSET_QUALITY["gold"]
        + reserves.bank_deposits * _RESERVE_ASSET_QUALITY["bank_deposits"]
    )


def compute_governance_score(
    reserves: ReserveComposition,
    transparency: TransparencyRating,
    compliance: ComplianceStatus,
    issuer_risk: float,
    weights: GovernanceWeights | None = None,
) -> float:
    weights = weights or GovernanceWeights()
    score = (
        weights.reserve_quality_weight * reserve_quality_score(reserves)
        + weights.transparency_weight * transparency_score(transparency)
        + weights.compliance_weight * (1.0 if compliance.genius_act_compliant else 0.0)
        + weights.issuer_risk_weight * (1.0 - issuer_risk)
    )
    return max(0.0, min(1.0, score))
