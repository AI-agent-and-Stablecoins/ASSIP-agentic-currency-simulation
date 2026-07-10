"""Gold-backed token configuration."""

from typing import Literal

from pydantic import Field, field_validator

from src.currencies.currency import AssetClass, CurrencyConfig


class GoldBackedConfig(CurrencyConfig):
    asset_class: Literal[AssetClass.GOLD_BACKED] = AssetClass.GOLD_BACKED
    gold_reserve_oz: float = Field(gt=0.0)
    custodian: str

    @field_validator("peg")
    @classmethod
    def peg_required(cls, value: str | None) -> str:
        if not value:
            raise ValueError("Gold-backed tokens must declare a peg (typically XAU)")
        return value
