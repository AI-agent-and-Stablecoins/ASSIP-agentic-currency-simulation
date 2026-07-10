"""Regulatory compliance status (e.g. GENIUS Act) for a currency issuer."""

from pydantic import BaseModel


class ComplianceStatus(BaseModel):
    genius_act_compliant: bool
    jurisdiction: str
