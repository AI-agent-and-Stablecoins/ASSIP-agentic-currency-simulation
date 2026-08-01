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

`compute_counterparty_zone_tax` (matrix runner cross-border-friction fix):
`compute_fx_tax` above only ever compares the SETTLEMENT CURRENCY's zone
against the buyer's zone -- it has no notion of which zone the SELLER is in.
That means the 6 cross-border matrix-runner cells (which force a
zone-mismatched buyer/seller pairing via `CrossZoneMarketplace`) add no
economic friction beyond what a same-currency-zone-mismatched buyer already
pays in a domestic cell: nothing in `compute_fx_tax` reacts to the
counterparty's zone at all. This function is a second, additive friction
that fires purely off the buyer/seller zone mismatch -- regardless of which
currency ends up settling the trade (even a zone-neutral gold-backed
currency) -- reusing the same `fx_tax_rate` from `configs/economy/fx_params.yaml`
rather than inventing a second rate.
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


def compute_counterparty_zone_tax(
    paid_value: float, buyer_zone: str | None, seller_zone: str | None, fx_tax_rate: float
) -> float:
    """Additional friction for a zone-mismatched buyer/seller pair, on top of
    (additive with) `compute_fx_tax`'s settlement-currency-zone check. Fires
    regardless of which currency actually settles the trade -- unlike
    `compute_fx_tax`, a zone-neutral gold-backed settlement currency does not
    exempt a cross-zone counterparty pairing from this tax, since the
    friction being modeled here is the cross-border COUNTERPARTY
    relationship itself, not the settlement currency's own zone. Zero when
    either side has no assigned zone (e.g. legacy count-based agents) or
    both sides share the same zone.
    """
    if buyer_zone is None or seller_zone is None or buyer_zone == seller_zone:
        return 0.0
    return paid_value * fx_tax_rate
