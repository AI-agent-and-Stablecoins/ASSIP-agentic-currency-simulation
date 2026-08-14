"""The equivalence/indifference-search framework, per
docs/superpowers/specs/2026-08-14-equivalence-framework-design.md. Every
comparison fixes one real currency's (or its assigned chain's) value at its
real config value (Y) and searches the other's corresponding value (X)
via binary search against the switch-elicitation question -- X-Y is the
reported compensation. When varied_field == "gas_fee", the modification
target is varied_currency's assigned chain (its
src.economy.hypothesis_scenarios.HYPOTHESIS_CHAIN_PINS entry), not the
currency itself -- gas_fee lives on ChainConfig, not CurrencyConfig.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EquivalenceComparison:
    hypothesis: str
    fixed_currency: str
    varied_currency: str
    varied_field: str
    bounds: tuple[float, float]


EQUIVALENCE_COMPARISONS: dict[str, list[EquivalenceComparison]] = {
    "H3": [EquivalenceComparison("H3", "USDT", "TDUSD", "liquidity_score", (0.0, 1.0))],
    "H4": [EquivalenceComparison("H4", "USDT", "DAI", "peg_error", (0.0, 0.05))],
    "H5": [EquivalenceComparison("H5", "USDT", "USDC", "gas_fee", (0.0, 5.0))],
    "H6": [EquivalenceComparison("H6", "EURC", "USDC", "liquidity_score", (0.0, 1.0))],
    "H7": [EquivalenceComparison("H7", "EURT", "USDC", "peg_error", (0.0, 0.05))],
    "H8": [EquivalenceComparison("H8", "EURC", "USDC", "gas_fee", (0.0, 5.0))],
    "H9": [EquivalenceComparison("H9", "TDUSD", "USDT", "peg_error", (0.0, 0.05))],
    "H10": [EquivalenceComparison("H10", "TDUSD", "USDT", "gas_fee", (0.0, 5.0))],
    "H11": [EquivalenceComparison("H11", "DAI", "TDUSD", "gas_fee", (0.0, 5.0))],
    "H2": [
        EquivalenceComparison("H2", "USDT", "EURC", "governance_score", (0.0, 1.0)),
        EquivalenceComparison("H2", "USDT", "PAXG", "governance_score", (0.0, 1.0)),
    ],
}
