"""Cell specs for the synthetic-coin hypothesis track, per
docs/superpowers/specs/2026-08-15-synthetic-coin-track-design.md §8.

Unlike `src/economy/hypothesis_scenarios.py`'s `HypothesisCellSpec` (which
names real currencies loaded from configs/currencies/*.yaml), each
`SyntheticHypothesisCellSpec` carries its own fully-controlled
`CurrencyConfig` instances and chain pins straight from
`src.currencies.synthetic_hypothesis_currencies.build_synthetic_hypothesis_currencies`
-- there is no cross-border or event-shock variant for this track, only one
baseline cell per hypothesis.
"""

from dataclasses import dataclass

from src.currencies.currency import CurrencyConfig
from src.currencies.synthetic_hypothesis_currencies import (
    SYNTHETIC_DIMENSION_PAIRS,
    build_synthetic_hypothesis_currencies,
)


@dataclass(frozen=True)
class SyntheticHypothesisCellSpec:
    hypothesis: str
    currencies: dict[str, CurrencyConfig]
    chain_pins: dict[str, str]

    @property
    def key(self) -> str:
        return f"{self.hypothesis}_synthetic"


def build_synthetic_hypothesis_cell_specs() -> list[SyntheticHypothesisCellSpec]:
    """One spec per hypothesis (H1-H11), baseline-only -- no cross-border or
    event variants for this track."""
    specs = []
    for hypothesis in SYNTHETIC_DIMENSION_PAIRS:
        currencies, chain_pins = build_synthetic_hypothesis_currencies(hypothesis)
        specs.append(
            SyntheticHypothesisCellSpec(
                hypothesis=hypothesis,
                currencies=currencies,
                chain_pins=chain_pins,
            )
        )
    return specs
