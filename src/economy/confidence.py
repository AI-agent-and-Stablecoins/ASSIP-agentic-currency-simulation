"""Aggregate confidence index, eroded by inflation and (starting Phase 2) hallucinations."""

from src.economy.macro_state import MacroState


def update_confidence(state: MacroState, recent_hallucination_rate: float = 0.0) -> float:
    """Phase 1 has no LLM decisions, so recent_hallucination_rate is always 0
    unless a caller supplies otherwise -- this is the Phase 2 extension point."""
    stability_term = 1.0 - min(state.inflation, 1.0)
    return max(0.0, min(1.0, stability_term - recent_hallucination_rate))
