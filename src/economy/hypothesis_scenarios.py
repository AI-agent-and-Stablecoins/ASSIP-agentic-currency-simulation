"""The 11 new hypotheses' sandbox definitions, per
docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-pivot-design.md.
Every hypothesis uses only real currencies (configs/currencies/*.yaml) --
no synthetic currencies, per that spec's explicit user decision. Gas-fee
hypotheses (H5, H8, H10, H11) additionally pin each currency to one real
chain via generate_candidates' currency_chain_pins, so the "better" trait
is always the one that costs more gas -- otherwise an agent could pick the
better currency on the cheaper chain and the tradeoff the hypothesis exists
to test would never actually be forced.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HypothesisCellSpec:
    hypothesis: str
    currencies: tuple[str, ...]
    chain_pins: dict[str, str] | None = None
    cross_border: bool = False
    event_shock: str | None = None
    event_target_currency: str | None = None


HYPOTHESIS_CURRENCIES: dict[str, tuple[str, ...]] = {
    "H1": ("USDC", "EURC", "PAXG"),
    "H2": ("USDC", "USDT", "EURC", "EURT", "PAXG", "XAUT"),
    "H3": ("TDUSD", "USDT"),
    "H4": ("DAI", "USDT"),
    "H5": ("USDC", "USDT"),
    "H6": ("USDC", "EURC"),
    "H7": ("USDC", "EURT"),
    "H8": ("USDC", "EURC"),
    "H9": ("TDUSD", "USDT"),
    "H10": ("USDT", "TDUSD"),
    "H11": ("TDUSD", "DAI"),
}

HYPOTHESIS_CHAIN_PINS: dict[str, dict[str, str]] = {
    "H5": {"USDC": "ethereum", "USDT": "solana"},
    "H8": {"USDC": "solana", "EURC": "ethereum"},
    "H10": {"USDT": "ethereum", "TDUSD": "solana"},
    "H11": {"TDUSD": "ethereum", "DAI": "solana"},
}

CROSS_BORDER_HYPOTHESES = ("H1", "H2", "H6", "H7", "H8")

EVENT_BASED_HYPOTHESES = ("H1", "H2", "H4", "H9")

EVENT_TARGET_CURRENCY: dict[str, str] = {
    "H1": "USDC",
    "H2": "USDT",
    "H4": "DAI",
    "H9": "USDT",
}


def build_hypothesis_cell_specs() -> list[HypothesisCellSpec]:
    specs: list[HypothesisCellSpec] = []

    for hypothesis, currencies in HYPOTHESIS_CURRENCIES.items():
        specs.append(
            HypothesisCellSpec(
                hypothesis=hypothesis,
                currencies=currencies,
                chain_pins=HYPOTHESIS_CHAIN_PINS.get(hypothesis),
            )
        )

    for hypothesis in CROSS_BORDER_HYPOTHESES:
        specs.append(
            HypothesisCellSpec(
                hypothesis=hypothesis,
                currencies=HYPOTHESIS_CURRENCIES[hypothesis],
                chain_pins=HYPOTHESIS_CHAIN_PINS.get(hypothesis),
                cross_border=True,
            )
        )

    for hypothesis in EVENT_BASED_HYPOTHESES:
        for shock in ("depeg", "banking_crisis"):
            specs.append(
                HypothesisCellSpec(
                    hypothesis=hypothesis,
                    currencies=HYPOTHESIS_CURRENCIES[hypothesis],
                    chain_pins=HYPOTHESIS_CHAIN_PINS.get(hypothesis),
                    event_shock=shock,
                    event_target_currency=EVENT_TARGET_CURRENCY[hypothesis],
                )
            )

    return specs
