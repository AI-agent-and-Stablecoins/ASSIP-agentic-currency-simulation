"""OpenRouter-backed LLM routing.

Model roster (what's available) and routing policy (how it's used) are
deliberately separate concepts (configs/llm/models.yaml) so a model's
identity is never implicitly read as "better" than another's -- see the
design doc §4. This file grows across Tasks 8-11: roster loading and the
OpenRouter preflight check here; the actual chat-completion call, retry,
and fallback-chain logic are added in Tasks 9-10.
"""

import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
from pydantic import BaseModel

from src.llm.decision_schema import Decision
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


_TECHNICAL_RETRY_STATUS_CODES = {429, 500, 502, 503}


@dataclass
class RetryConfig:
    """Plain dataclass, not a pydantic model: sleep_fn is a callable injected
    by tests (to skip real backoff delays), not serializable config data."""

    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    sleep_fn: Callable[[float], None] = _time.sleep


class AuthenticationError(Exception):
    pass


class ModelCallFailedError(Exception):
    def __init__(self, model_id: str, reason: str):
        self.model_id = model_id
        self.reason = reason
        super().__init__(f"Model {model_id} failed: {reason}")


def build_openrouter_client(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=OPENROUTER_BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        transport=transport,
        timeout=30.0,
    )


def _post_chat_completion(client: httpx.Client, model_id: str, messages: list[dict]) -> httpx.Response:
    return client.post(
        "/chat/completions",
        json={"model": model_id, "messages": messages, "response_format": {"type": "json_object"}},
    )


def _parse_decision(response: httpx.Response) -> Decision:
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return Decision.model_validate_json(content)


def call_model(
    prompt: str,
    model_id: str,
    client: httpx.Client,
    retry_config: RetryConfig | None = None,
) -> Decision:
    """Call one specific model, handling the first two failure tiers:
    technical failures (retried with exponential backoff) and malformed
    output (one repair reprompt per attempt). Raises AuthenticationError
    immediately on 401/403 -- a bad key won't fix itself by trying again --
    or ModelCallFailedError once retries and repair are exhausted.

    Economic validity (tier 3) is deliberately not handled here: only the
    caller (src/llm/agent_reasoning.py) knows the wallet/currency/chain
    constraints a Decision must satisfy.
    """
    retry_config = retry_config or RetryConfig()
    messages = [{"role": "user", "content": prompt}]
    last_error = "unknown error"

    for attempt in range(retry_config.max_retries):
        try:
            response = _post_chat_completion(client, model_id, messages)
        except httpx.TimeoutException as exc:
            last_error = f"timeout: {exc}"
            retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
            continue

        if response.status_code in (401, 403):
            raise AuthenticationError(
                f"OpenRouter rejected the API key for model {model_id}: HTTP {response.status_code}"
            )

        if response.status_code in _TECHNICAL_RETRY_STATUS_CODES:
            last_error = f"HTTP {response.status_code}"
            retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
            continue

        if response.status_code != 200:
            raise ModelCallFailedError(model_id, f"unexpected HTTP {response.status_code}")

        try:
            return _parse_decision(response)
        except (KeyError, IndexError, ValueError) as exc:
            repair_messages = messages + [
                {"role": "assistant", "content": response.text},
                {
                    "role": "user",
                    "content": (
                        f"Your last response was not valid JSON matching the required schema: {exc}. "
                        "Respond again with valid JSON only."
                    ),
                },
            ]
            try:
                repair_response = _post_chat_completion(client, model_id, repair_messages)
                return _parse_decision(repair_response)
            except (KeyError, IndexError, ValueError) as repair_exc:
                last_error = f"malformed output, repair failed: {repair_exc}"
                retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
                continue

    raise ModelCallFailedError(model_id, last_error)
