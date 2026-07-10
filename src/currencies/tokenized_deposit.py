"""Bank-issued tokenized deposit configuration."""

from typing import Literal

from src.currencies.currency import AssetClass, CurrencyConfig


class TokenizedDepositConfig(CurrencyConfig):
    asset_class: Literal[AssetClass.TOKENIZED_DEPOSIT] = AssetClass.TOKENIZED_DEPOSIT
    issuing_bank: str
    fdic_insured: bool
