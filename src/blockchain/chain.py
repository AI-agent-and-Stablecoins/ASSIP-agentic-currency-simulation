"""Blockchain properties: throughput, gas cost, and finality."""

from pathlib import Path

from pydantic import BaseModel, Field

from src.utils.config_loader import load_yaml_dir_as
from src.utils.constants import CONFIG_ROOT


class ChainConfig(BaseModel):
    name: str
    throughput: float = Field(gt=0.0)
    gas_fee: float = Field(ge=0.0)
    finality_seconds: float = Field(gt=0.0)


def load_chain_universe(config_dir: Path = CONFIG_ROOT / "blockchains") -> dict[str, ChainConfig]:
    """Load every blockchains/*.yaml file, keyed by chain name."""
    by_filename = load_yaml_dir_as(config_dir, ChainConfig)
    return {cfg.name: cfg for cfg in by_filename.values()}
