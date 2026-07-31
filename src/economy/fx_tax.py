"""Cross-border FX conversion tax: a small (0.02%) friction applied whenever
a settlement currency's zone (USD/EUR) differs from the buyer's own
currency_zone. Gold-backed currencies (peg="XAU" etc.) are zone-neutral --
never taxed, regardless of the buyer's zone -- since they aren't a national
currency conversion. An agent with no currency_zone assigned (e.g. the
legacy count-based Environment.build path, which never sets it) never pays
this tax either: there is no "home zone" to compare against.

Follows Plan 2's trust.py/trust_params.yaml loader pattern exactly (see
src/economy/trust.py) so config-driven economic parameters stay consistent
across the codebase.
"""

from pathlib import Path

from pydantic import BaseModel

from src.currencies.currency import CurrencyConfig
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

FX_PARAMS_PATH = CONFIG_ROOT / "economy" / "fx_params.yaml"


class FxParams(BaseModel):
    fx_tax_rate: float


def load_fx_params(path: Path = FX_PARAMS_PATH) -> FxParams:
    return load_yaml_as(path, FxParams)


def currency_zone_of(currency: CurrencyConfig) -> str | None:
    if currency.peg == "USD":
        return "USD"
    if currency.peg == "EUR":
        return "EUR"
    return None


def compute_fx_tax(paid_value: float, currency: CurrencyConfig, buyer_zone: str | None, fx_tax_rate: float) -> float:
    currency_zone = currency_zone_of(currency)
    if currency_zone is None or buyer_zone is None or currency_zone == buyer_zone:
        return 0.0
    return paid_value * fx_tax_rate
