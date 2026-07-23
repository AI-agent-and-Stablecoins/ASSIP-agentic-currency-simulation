"""Feeds real-world stablecoin context into LLM prompts.

Two clearly separate sources: a static, git-versioned profile corpus (this
module's load_currency_profile, compiled from deep-research-report.md) and
an optional live price snapshot (added in a later task, via Polygon). The
static corpus must be presented to the LLM as background/historical
information, not current market state -- see the design doc's §6.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

PROFILES_DIR = CONFIG_ROOT / "currencies" / "profiles"


class TimelineEvent(BaseModel):
    date: str
    event: str


class CurrencyProfile(BaseModel):
    symbol: str
    executive_summary: str
    timeline: list[TimelineEvent] = Field(default_factory=list)
    reserves_and_transparency: str
    governance: str
    price_and_market_cap: str
    crra_cara_note: str
    use_cases: str
    regulatory_and_controversies: str
    source: str
    report_date: str


def load_currency_profile(symbol: str, profiles_dir: Path = PROFILES_DIR) -> CurrencyProfile | None:
    """Return the static profile for symbol, or None if no profile file exists.

    None (not an exception) on a missing file: a currency without a curated
    profile must degrade gracefully in the LLM context rather than crash the
    decision pipeline -- the same principle the live-price fetch (added
    later in this module) also follows.
    """
    path = profiles_dir / f"{symbol}.yaml"
    if not path.exists():
        return None
    return load_yaml_as(path, CurrencyProfile)
