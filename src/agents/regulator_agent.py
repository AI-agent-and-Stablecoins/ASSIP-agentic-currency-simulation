"""Regulator agent: passive observer in Phase 1.

Records compliance-related observations but does not act on them --
enforcement/intervention logic belongs to Phase 3 (economic shocks).
"""

from src.agents.base_agent import BaseAgent

_COMPLIANCE_MEMORY_KEY = "__compliance_observations__"


class RegulatorAgent(BaseAgent):
    def observe_compliance(self, genius_compliant: bool) -> None:
        self.memory.record(_COMPLIANCE_MEMORY_KEY, genius_compliant)
