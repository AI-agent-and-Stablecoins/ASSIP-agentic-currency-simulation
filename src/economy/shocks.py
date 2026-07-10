"""Scenario definitions and the shocks (inflation, bank failures, gold rallies,
fee spikes) they trigger against the macro state."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from src.economy.macro_state import MacroState
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT


class ShockType(str, Enum):
    INFLATION = "inflation"
    BANK_FAILURE = "bank_failure"
    GOLD_RALLY = "gold_rally"
    FEE_SPIKE = "fee_spike"


class ShockEvent(BaseModel):
    day: int
    type: ShockType
    magnitude: float


class ScenarioConfig(BaseModel):
    name: str
    initial_state: MacroState
    shocks: list[ShockEvent] = Field(default_factory=list)
    duration_days: int


def load_scenario(name: str, config_dir: Path = CONFIG_ROOT / "scenarios") -> ScenarioConfig:
    return load_yaml_as(config_dir / f"{name}.yaml", ScenarioConfig)


def apply_shock(state: MacroState, shock: ShockEvent) -> MacroState:
    updated = state.model_copy(deep=True)
    if shock.type == ShockType.INFLATION:
        updated.inflation += shock.magnitude
    elif shock.type == ShockType.GOLD_RALLY:
        updated.gold_price *= 1 + shock.magnitude
        updated.peg_reference_rates["XAU"] = updated.gold_price
    elif shock.type == ShockType.BANK_FAILURE:
        updated.confidence_index = max(0.0, updated.confidence_index - shock.magnitude)
    elif shock.type == ShockType.FEE_SPIKE:
        # Fee spikes mutate blockchain gas_fee configs directly (src/blockchain),
        # not macro state -- the caller applies this shock at that layer.
        pass
    return updated
