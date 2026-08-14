"""The 11 new hypotheses' sandbox definitions, per
docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-pivot-design.md.
Every hypothesis uses only real currencies (configs/currencies/*.yaml) --
no synthetic currencies, per that spec's explicit user decision. Gas-fee
hypotheses (H5, H8, H10, H11) additionally pin each currency to one real
chain via generate_candidates' currency_chain_pins, so the "better" trait
is always the one that costs more gas -- otherwise an agent could pick the
better currency on the cheaper chain and the tradeoff the hypothesis exists
to test would never actually be forced.

Every one of these hypothesis-sims MUST run with run_timestep's use_llm=True
(the real per-agent LLM decision path), never the rule-based deterministic
path. CRRAUtility, CARAUtility, and EpsteinZinProxyUtility (src/utility/) are
each a strictly monotone transform of a single effective-wealth scalar, so
choose_best's argmax on the deterministic path is IDENTICAL for every risk-
aversion cohort -- risk-aversion has zero effect on which currency/chain a
deterministic-path agent picks. Only the LLM path actually varies its answer
by the risk_aversion value printed into its own prompt context. Running a
hypothesis-sim deterministically would silently produce four identical
cohort rows with no error anywhere.
"""

from dataclasses import dataclass

from src.economy.shocks import ShockType


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
    # H1/H2 have no single "worse" tradeoff side (H1 is 3-way medium of
    # exchange alone; H2 crosses governance with 3-way medium of exchange),
    # unlike H4/H9's pairwise tradeoffs -- USDC/USDT are the shock target
    # for the banking-crisis flight-to-safety narrative these two were
    # included for (the currency an agent flees FROM, not gold, the natural
    # flight-TO asset in H1's own triple).
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
        for shock in (ShockType.DEPEG_EVENT.value, ShockType.BANK_FAILURE.value):
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
