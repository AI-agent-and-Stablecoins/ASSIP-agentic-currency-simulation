"""Generic YAML -> Pydantic config loading shared by every config-driven module.

Centralizing this means individual modules (currencies, blockchain, governance,
agent profiles, scenarios, simulation) never hand-roll YAML parsing or
validation error handling.
"""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_yaml_as(path: Path, model: type[T]) -> T:
    """Load a single YAML file and validate it against a Pydantic model."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return model.model_validate(raw)


def load_yaml_dir_as(directory: Path, model: type[T]) -> dict[str, T]:
    """Load every *.yaml file in a directory, keyed by filename stem.

    Use this for homogeneous config directories (one Pydantic model per
    directory). Heterogeneous directories that need per-file dispatch
    (e.g. currencies, where each file's asset_class picks a subclass)
    should call load_yaml_as directly with their own dispatch logic.
    """
    result: dict[str, T] = {}
    for path in sorted(directory.glob("*.yaml")):
        result[path.stem] = load_yaml_as(path, model)
    return result
