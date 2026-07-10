"""Base currency configuration shared by every digital medium of exchange.

Stablecoins, gold-backed tokens, and tokenized deposits all share the same
attribute surface (governance, liquidity, peg stability, issuer risk,
compliance) so agent utility functions can compare them uniformly. Phase 1
is stateless -- a Currency is exactly its validated config, with no mutable
runtime fields -- so there is no separate runtime wrapper class.
"""

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.utils.constants import CONFIG_ROOT


class AssetClass(str, Enum):
    STABLECOIN = "stablecoin"
    GOLD_BACKED = "gold_backed"
    TOKENIZED_DEPOSIT = "tokenized_deposit"


class CurrencyConfig(BaseModel):
    symbol: str
    asset_class: AssetClass
    peg: str | None = None
    governance_score: float = Field(ge=0.0, le=1.0)
    liquidity_score: float = Field(ge=0.0, le=1.0)
    peg_error: float = Field(ge=0.0)
    issuer_risk: float = Field(ge=0.0, le=1.0)
    genius_compliant: bool


# Phase 1: a "Currency" is just its validated config. Kept as an alias so
# call sites can say `Currency` without caring that there's no extra state yet.
Currency = CurrencyConfig


def load_currency_universe(config_dir: Path = CONFIG_ROOT / "currencies") -> dict[str, CurrencyConfig]:
    """Load every currencies/*.yaml file, dispatching to the right subclass by asset_class."""
    from src.currencies.gold_token import GoldBackedConfig
    from src.currencies.stablecoin import StablecoinConfig
    from src.currencies.tokenized_deposit import TokenizedDepositConfig

    dispatch: dict[AssetClass, type[CurrencyConfig]] = {
        AssetClass.STABLECOIN: StablecoinConfig,
        AssetClass.GOLD_BACKED: GoldBackedConfig,
        AssetClass.TOKENIZED_DEPOSIT: TokenizedDepositConfig,
    }

    universe: dict[str, CurrencyConfig] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        asset_class = AssetClass(raw["asset_class"])
        model = dispatch[asset_class]
        config = model.model_validate(raw)
        universe[config.symbol] = config
    return universe
