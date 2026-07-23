"""OpenRouter-backed LLM routing.

Model roster (what's available) and routing policy (how it's used) are
deliberately separate concepts (configs/llm/models.yaml) so a model's
identity is never implicitly read as "better" than another's -- see the
design doc §4. This file grows across Tasks 8-11: roster loading and the
OpenRouter preflight check here; the actual chat-completion call, retry,
and fallback-chain logic are added in Tasks 9-10.
"""

from pathlib import Path

import httpx
from pydantic import BaseModel

from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

MODELS_CONFIG_PATH = CONFIG_ROOT / "llm" / "models.yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelEntry(BaseModel):
    id: str
    label: str


class ReliabilityChain(BaseModel):
    primary: str
    fallbacks: list[str]


class ModelComparisonPolicy(BaseModel):
    pinned_models: list[str]


class RoutingPolicies(BaseModel):
    default_reliability_chain: ReliabilityChain
    model_comparison: ModelComparisonPolicy


class ModelRosterConfig(BaseModel):
    models: list[ModelEntry]
    routing_policies: RoutingPolicies

    def resolve(self, label: str) -> str:
        for entry in self.models:
            if entry.label == label:
                return entry.id
        raise ValueError(f"No model with label {label!r} in the roster")


def load_model_roster(path: Path = MODELS_CONFIG_PATH) -> ModelRosterConfig:
    return load_yaml_as(path, ModelRosterConfig)


class ModelNotAvailableError(Exception):
    def __init__(self, label: str, model_id: str, detail: str):
        self.label = label
        self.model_id = model_id
        super().__init__(f"Model {label!r} ({model_id}) is not available on OpenRouter: {detail}")


def verify_model_roster(roster: ModelRosterConfig, client: httpx.Client) -> None:
    """Preflight check: fail loudly and specifically if a configured model ID
    doesn't resolve against OpenRouter, rather than discovering it mid-run as
    an unexplained call failure or a silent fallback substitution."""
    response = client.get("/models")
    response.raise_for_status()
    available_ids = {entry["id"] for entry in response.json()["data"]}

    for entry in roster.models:
        if entry.id not in available_ids:
            raise ModelNotAvailableError(entry.label, entry.id, "not present in OpenRouter's /models response")
