"""Updates purchasing power via the inflation rate."""

from src.economy.macro_state import MacroState


def apply_inflation(state: MacroState, rate: float) -> MacroState:
    updated = state.model_copy(deep=True)
    updated.inflation += rate
    return updated
