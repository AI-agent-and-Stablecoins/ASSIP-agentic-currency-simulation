"""Scenario definitions and the shocks (inflation, bank failures, gold rallies,
fee spikes) they trigger against the macro state."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from src.currencies.currency import CurrencyConfig
from src.economy.macro_state import MacroState
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT
from src.utils.helpers import clamp


class ShockType(str, Enum):
    INFLATION = "inflation"
    BANK_FAILURE = "bank_failure"
    GOLD_RALLY = "gold_rally"
    FEE_SPIKE = "fee_spike"
    REGULATORY_ENFORCEMENT = "regulatory_enforcement"
    LIQUIDITY_CRUNCH = "liquidity_crunch"
    GOVERNANCE_DOWNGRADE = "governance_downgrade"
    DEPEG_EVENT = "depeg_event"
    CRISIS_WARNING = "crisis_warning"
    FX_VOLATILITY_SHOCK = "fx_volatility_shock"
    FX_RATE_SHOCK = "fx_rate_shock"
    CAPITAL_CONTROLS = "capital_controls"


class ShockEvent(BaseModel):
    day: int
    type: ShockType
    magnitude: float = Field(ge=0.0)
    target_currency: str | None = None
    target_issuer: str | None = None
    decay_days: int | None = None
    """Currently informational only -- TrustLedger's offset channels decay
    via the single global TrustParams.lambda_recover formula (design spec
    Sec 3.2's deliberate choice over a second, per-shock decay mechanism),
    not this field. Reserved for a future per-shock override if one is
    ever added."""


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
    elif shock.type == ShockType.CRISIS_WARNING:
        updated.confidence_index = max(0.0, updated.confidence_index - shock.magnitude)
    elif shock.type == ShockType.FX_RATE_SHOCK:
        updated.peg_reference_rates["EUR"] = updated.peg_reference_rates["EUR"] * (1 + shock.magnitude)
    return updated


def apply_currency_shock(
    currencies: dict[str, CurrencyConfig], shock: ShockEvent
) -> dict[str, CurrencyConfig]:
    """Permanent, structural currency-attribute mutations only
    (governance_downgrade, regulatory_enforcement's issuer_risk component).
    Temporary/decaying effects (depeg_event, liquidity_crunch, ...) are
    handled by TrustLedger's offset channels (src/economy/trust.py), not
    here -- see docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md
    Sec 2-3 for why these are two different mechanisms."""
    if shock.target_currency is None or shock.target_currency not in currencies:
        return currencies

    updated = dict(currencies)
    target = updated[shock.target_currency]

    if shock.type == ShockType.GOVERNANCE_DOWNGRADE:
        new_score = clamp(target.governance_score - shock.magnitude, 0.0, 1.0)
        updated[shock.target_currency] = target.model_copy(update={"governance_score": new_score})
    elif shock.type == ShockType.REGULATORY_ENFORCEMENT:
        new_risk = clamp(target.issuer_risk + shock.magnitude, 0.0, 1.0)
        updated[shock.target_currency] = target.model_copy(update={"issuer_risk": new_risk})

    return updated
