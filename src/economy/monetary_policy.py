"""Interest-rate / peg-defense policy hooks.

Mostly a thin pass-through in Phase 1 -- active policy responses to shocks
are a Phase 3 extension.
"""

from src.economy.macro_state import MacroState


def set_interest_rate(state: MacroState, rate: float) -> MacroState:
    updated = state.model_copy(deep=True)
    updated.interest_rate = rate
    return updated
