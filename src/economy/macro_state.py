"""Current state of the macroeconomy, including FX/gold reference rates
used by src/currencies/exchange_rates.py."""

from pydantic import BaseModel, Field


class MacroState(BaseModel):
    inflation: float = 0.0
    interest_rate: float = 0.0
    gold_price: float = 2400.0
    confidence_index: float = 1.0
    peg_reference_rates: dict[str, float] = Field(
        default_factory=lambda: {"USD": 1.0, "EUR": 1.08, "XAU": 2400.0}
    )
