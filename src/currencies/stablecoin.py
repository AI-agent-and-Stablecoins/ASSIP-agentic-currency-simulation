"""Fiat-backed stablecoin configuration."""

from typing import Literal

from pydantic import field_validator

from src.currencies.currency import AssetClass, CurrencyConfig


class StablecoinConfig(CurrencyConfig):
    asset_class: Literal[AssetClass.STABLECOIN] = AssetClass.STABLECOIN
    redemption_mechanism: str

    @field_validator("peg")
    @classmethod
    def peg_required(cls, value: str | None) -> str:
        if not value:
            raise ValueError("Stablecoins must declare a peg currency")
        return value
