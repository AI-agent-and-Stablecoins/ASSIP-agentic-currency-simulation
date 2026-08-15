"""Synthetic 5-dimension currency grid for the second, fully-controlled
hypothesis track, per `docs/superpowers/specs/2026-08-15-synthetic-coin-track-design.md`.

Unlike `src/economy/hypothesis_scenarios.py` (which picks real stablecoins
that happen to differ mostly along one tested dimension -- an approximation),
this module builds coins with EXACTLY the attribute values the spec
specifies, isolating the tested dimension(s) precisely and holding every
other dimension at one fixed, neutral value across an entire hypothesis's
grid. This mirrors `src/currencies/sandbox_currencies.py`'s pattern (fully
controlled synthetic `StablecoinConfig`/`GoldBackedConfig` instances), just
generalized from a 2-coin pair to an N-coin cross-product grid.

Nothing here is loaded from YAML and nothing here is registered with
`load_chain_universe()` -- the synthetic chains defined below stay
completely separate from the real chain universe used by the real-coin
track and the master simulation.
"""

import itertools

from src.blockchain.chain import ChainConfig
from src.currencies.currency import AssetClass, CurrencyConfig
from src.currencies.gold_token import GoldBackedConfig
from src.currencies.stablecoin import StablecoinConfig

# ---------------------------------------------------------------------------
# Task 2: synthetic gas-fee chains (plain Python, never YAML, never
# registered with load_chain_universe()).
# ---------------------------------------------------------------------------

SYNTHETIC_GAS_LEVELS = (0.01, 0.05, 0.10)

SYNTHETIC_CHAINS: dict[str, ChainConfig] = {
    "synthetic_gas_low": ChainConfig(name="synthetic_gas_low", throughput=1000.0, gas_fee=0.01, finality_seconds=5.0),
    "synthetic_gas_mid": ChainConfig(name="synthetic_gas_mid", throughput=1000.0, gas_fee=0.05, finality_seconds=5.0),
    "synthetic_gas_high": ChainConfig(
        name="synthetic_gas_high", throughput=1000.0, gas_fee=0.10, finality_seconds=5.0
    ),
}

# gas_fee level -> the synthetic chain whose ChainConfig.gas_fee matches it.
_CHAIN_BY_GAS_FEE: dict[float, str] = {
    chain.gas_fee: name for name, chain in SYNTHETIC_CHAINS.items()
}

# ---------------------------------------------------------------------------
# Task 3: the 5-dimension grid + per-hypothesis coin builder.
# ---------------------------------------------------------------------------

GOVERNANCE_LEVELS = (0.0, 1.0)  # low, high
MEDIUM_LEVELS = ("USD", "EUR", "XAU")
BID_ASK_SPREAD_LEVELS = (0.0001, 0.0005, 0.0010)  # 0.01% / 0.05% / 0.10%
VOLATILITY_LEVELS = (0.001, 0.004, 0.008)  # 0.1% / 0.4% / 0.8% -- maps to peg_error
GAS_FEE_LEVELS = SYNTHETIC_GAS_LEVELS

SYNTHETIC_DIMENSION_PAIRS: dict[str, tuple[str, ...]] = {
    "H1": ("medium",),
    "H2": ("governance", "medium"),
    "H3": ("governance", "liquidity"),
    "H4": ("governance", "volatility"),
    "H5": ("governance", "gas_fee"),
    "H6": ("medium", "liquidity"),
    "H7": ("medium", "volatility"),
    "H8": ("medium", "gas_fee"),
    "H9": ("liquidity", "volatility"),
    "H10": ("liquidity", "gas_fee"),
    "H11": ("volatility", "gas_fee"),
}

# Neutral fixed value for a dimension when it's NOT one of a hypothesis's
# tested pair -- a defensible, symmetric midpoint/compliant-default choice
# per spec §3, held identical across every hypothesis's grid.
NEUTRAL_FIXED_VALUES: dict[str, float | str] = {
    "governance": 1.0,
    "medium": "USD",
    "liquidity": 0.0005,
    "volatility": 0.004,
    "gas_fee": 0.05,
}

# Every dimension's full level tuple, keyed by the same dimension names used
# in SYNTHETIC_DIMENSION_PAIRS/NEUTRAL_FIXED_VALUES.
_LEVELS_BY_DIMENSION: dict[str, tuple] = {
    "governance": GOVERNANCE_LEVELS,
    "medium": MEDIUM_LEVELS,
    "liquidity": BID_ASK_SPREAD_LEVELS,
    "volatility": VOLATILITY_LEVELS,
    "gas_fee": GAS_FEE_LEVELS,
}

# issuer_risk and liquidity_score (the OLD abstract 0-1 liquidity field, not
# this track's new literal bid_ask_spread) are never tested by any of the 11
# hypotheses -- held at one shared neutral constant across every coin in
# every hypothesis's grid, matching sandbox_currencies.py's convention.
_SYNTHETIC_ISSUER_RISK = 0.10
_SYNTHETIC_LIQUIDITY_SCORE = 0.90

_SYNTHETIC_REDEMPTION = "Synthetic sandbox currency for factor-isolation testing (redemption mechanism placeholder)"
_SYNTHETIC_CUSTODIAN = "Synthetic Sandbox Custodian (factor-isolation testing placeholder)"
_SYNTHETIC_GOLD_RESERVE_OZ = 100_000.0


def build_synthetic_hypothesis_currencies(hypothesis: str) -> tuple[dict[str, CurrencyConfig], dict[str, str]]:
    """Returns (currencies, chain_pins) for `hypothesis`'s full cross-product
    grid (per spec §3's table), holding every untested dimension at
    NEUTRAL_FIXED_VALUES. chain_pins maps every currency symbol in the grid
    to its assigned synthetic chain name -- gas fee is always determined,
    whether tested (pinned per-combination) or held neutral (pinned to the
    chain matching NEUTRAL_FIXED_VALUES["gas_fee"])."""
    tested_dimensions = SYNTHETIC_DIMENSION_PAIRS[hypothesis]
    level_tuples = [_LEVELS_BY_DIMENSION[dimension] for dimension in tested_dimensions]

    currencies: dict[str, CurrencyConfig] = {}
    chain_pins: dict[str, str] = {}

    for index, combination in enumerate(itertools.product(*level_tuples)):
        values: dict[str, float | str] = dict(NEUTRAL_FIXED_VALUES)
        values.update(dict(zip(tested_dimensions, combination)))

        symbol = f"SYN_{hypothesis}_{index:02d}"
        peg = str(values["medium"])
        governance_score = float(values["governance"])
        genius_compliant = governance_score >= 1.0
        bid_ask_spread = float(values["liquidity"])
        peg_error = float(values["volatility"])
        gas_fee = float(values["gas_fee"])

        common_kwargs = dict(
            symbol=symbol,
            peg=peg,
            governance_score=governance_score,
            liquidity_score=_SYNTHETIC_LIQUIDITY_SCORE,
            peg_error=peg_error,
            issuer_risk=_SYNTHETIC_ISSUER_RISK,
            genius_compliant=genius_compliant,
            bid_ask_spread=bid_ask_spread,
        )

        currency: CurrencyConfig
        if peg == "XAU":
            currency = GoldBackedConfig(
                asset_class=AssetClass.GOLD_BACKED,
                gold_reserve_oz=_SYNTHETIC_GOLD_RESERVE_OZ,
                custodian=_SYNTHETIC_CUSTODIAN,
                **common_kwargs,
            )
        else:
            currency = StablecoinConfig(
                asset_class=AssetClass.STABLECOIN,
                redemption_mechanism=_SYNTHETIC_REDEMPTION,
                **common_kwargs,
            )

        currencies[symbol] = currency
        chain_pins[symbol] = _CHAIN_BY_GAS_FEE[gas_fee]

    return currencies, chain_pins
