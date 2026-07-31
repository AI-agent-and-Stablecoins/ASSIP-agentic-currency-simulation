"""Synthetic per-sandbox CurrencyConfig pairs for the 6 factor-isolation sandboxes.

Per the Phase 3 Plan 4 design spec Sec 6.2: the real 9-currency universe is
internally consistent (better governance correlates with better peg
stability throughout), so it cannot express a deliberate one-dimension
tradeoff (e.g. "compliant, PegError=0.02" vs "non-compliant, PegError=0.00").
These 12 synthetic CurrencyConfig instances (6 Option-A/Option-B pairs) are
built directly, matching the design spec's Sec 6.2 table numbers exactly,
isolating exactly the named dimension(s) per sandbox while holding every
other field constant at a shared neutral value within that pair.

These are sandbox-scoped only -- never loaded from YAML, never part of the
persisted real universe (see `load_currency_universe`).
"""

from src.currencies.currency import AssetClass, CurrencyConfig
from src.currencies.gold_token import GoldBackedConfig
from src.currencies.stablecoin import StablecoinConfig
from src.currencies.tokenized_deposit import TokenizedDepositConfig

_SYNTHETIC_REDEMPTION = "Synthetic sandbox currency for factor-isolation testing (redemption mechanism placeholder)"
_SYNTHETIC_CUSTODIAN = "Synthetic Sandbox Custodian (factor-isolation testing placeholder)"
_SYNTHETIC_BANK = "Synthetic Sandbox Bank (factor-isolation testing placeholder)"

# Sandbox 1: Liquidity vs. Governance -- isolates governance_score, liquidity_score.
# Held constant: peg_error, issuer_risk, asset_class (stablecoin), peg (USD).
_LIQUIDITY_VS_GOVERNANCE = (
    StablecoinConfig(
        symbol="SBX1_HILIQ_LOGOV",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.55,
        liquidity_score=0.99,
        peg_error=0.001,
        issuer_risk=0.15,
        genius_compliant=False,
        redemption_mechanism=_SYNTHETIC_REDEMPTION,
    ),
    StablecoinConfig(
        symbol="SBX1_HIGOV_LOLIQ",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.95,
        liquidity_score=0.90,
        peg_error=0.001,
        issuer_risk=0.15,
        genius_compliant=True,
        redemption_mechanism=_SYNTHETIC_REDEMPTION,
    ),
)

# Sandbox 2: Governance vs. Stability -- isolates governance_score, peg_error.
# Held constant: liquidity_score, issuer_risk, asset_class (stablecoin), peg (USD).
_GOVERNANCE_VS_STABILITY = (
    StablecoinConfig(
        symbol="SBX2_HIGOV_LOSTAB",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.95,
        liquidity_score=0.85,
        peg_error=0.02,
        issuer_risk=0.15,
        genius_compliant=True,
        redemption_mechanism=_SYNTHETIC_REDEMPTION,
    ),
    StablecoinConfig(
        symbol="SBX2_LOGOV_HISTAB",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.55,
        liquidity_score=0.85,
        peg_error=0.0001,
        issuer_risk=0.15,
        genius_compliant=False,
        redemption_mechanism=_SYNTHETIC_REDEMPTION,
    ),
)

# Sandbox 3: Liquidity vs. Stability -- isolates liquidity_score, peg_error.
# Held constant: governance_score, issuer_risk, genius_compliant, asset_class (stablecoin), peg (USD).
_LIQUIDITY_VS_STABILITY = (
    StablecoinConfig(
        symbol="SBX3_HILIQ_LOSTAB",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.75,
        liquidity_score=0.99,
        peg_error=0.04,
        issuer_risk=0.15,
        genius_compliant=True,
        redemption_mechanism=_SYNTHETIC_REDEMPTION,
    ),
    StablecoinConfig(
        symbol="SBX3_LOLIQ_HISTAB",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.75,
        liquidity_score=0.75,
        peg_error=0.0001,
        issuer_risk=0.15,
        genius_compliant=True,
        redemption_mechanism=_SYNTHETIC_REDEMPTION,
    ),
)

# Sandbox 4: Asset Backing vs. Liquidity -- isolates asset_class, liquidity_score.
# Held constant: governance_score, peg_error, issuer_risk, genius_compliant.
_ASSET_BACKING_VS_LIQUIDITY = (
    GoldBackedConfig(
        symbol="SBX4_GOLD_LOLIQ",
        asset_class=AssetClass.GOLD_BACKED,
        peg="XAU",
        governance_score=0.80,
        liquidity_score=0.70,
        peg_error=0.01,
        issuer_risk=0.15,
        genius_compliant=True,
        gold_reserve_oz=100_000.0,
        custodian=_SYNTHETIC_CUSTODIAN,
    ),
    StablecoinConfig(
        symbol="SBX4_STABLE_HILIQ",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.80,
        liquidity_score=0.99,
        peg_error=0.01,
        issuer_risk=0.15,
        genius_compliant=True,
        redemption_mechanism=_SYNTHETIC_REDEMPTION,
    ),
)

# Sandbox 5: Asset Backing vs. Stability -- isolates asset_class, peg_error.
# Held constant: governance_score, liquidity_score, issuer_risk, genius_compliant.
_ASSET_BACKING_VS_STABILITY = (
    GoldBackedConfig(
        symbol="SBX5_GOLD_LOSTAB",
        asset_class=AssetClass.GOLD_BACKED,
        peg="XAU",
        governance_score=0.80,
        liquidity_score=0.80,
        peg_error=0.015,
        issuer_risk=0.15,
        genius_compliant=True,
        gold_reserve_oz=100_000.0,
        custodian=_SYNTHETIC_CUSTODIAN,
    ),
    TokenizedDepositConfig(
        symbol="SBX5_DEPOSIT_HISTAB",
        asset_class=AssetClass.TOKENIZED_DEPOSIT,
        peg="USD",
        governance_score=0.80,
        liquidity_score=0.80,
        peg_error=0.0001,
        issuer_risk=0.15,
        genius_compliant=True,
        issuing_bank=_SYNTHETIC_BANK,
        fdic_insured=True,
    ),
)

# Sandbox 6: Asset Backing vs. Governance -- isolates asset_class, issuer_risk/governance_score.
# Held constant: liquidity_score, peg_error, genius_compliant.
_ASSET_BACKING_VS_GOVERNANCE = (
    TokenizedDepositConfig(
        symbol="SBX6_DEPOSIT_BANKRISK",
        asset_class=AssetClass.TOKENIZED_DEPOSIT,
        peg="USD",
        governance_score=0.75,
        liquidity_score=0.85,
        peg_error=0.01,
        issuer_risk=0.25,
        genius_compliant=False,
        issuing_bank=_SYNTHETIC_BANK,
        fdic_insured=True,
    ),
    StablecoinConfig(
        symbol="SBX6_STABLE_ALGO",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.70,
        liquidity_score=0.85,
        peg_error=0.01,
        issuer_risk=0.20,
        genius_compliant=False,
        redemption_mechanism=_SYNTHETIC_REDEMPTION,
    ),
)

SANDBOX_CURRENCY_PAIRS: dict[str, tuple[CurrencyConfig, CurrencyConfig]] = {
    "liquidity_vs_governance": _LIQUIDITY_VS_GOVERNANCE,
    "governance_vs_stability": _GOVERNANCE_VS_STABILITY,
    "liquidity_vs_stability": _LIQUIDITY_VS_STABILITY,
    "asset_backing_vs_liquidity": _ASSET_BACKING_VS_LIQUIDITY,
    "asset_backing_vs_stability": _ASSET_BACKING_VS_STABILITY,
    "asset_backing_vs_governance": _ASSET_BACKING_VS_GOVERNANCE,
}
