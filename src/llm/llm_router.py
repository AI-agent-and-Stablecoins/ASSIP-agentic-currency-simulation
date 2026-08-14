"""OpenRouter-backed LLM routing.

Model roster (what's available) and routing policy (how it's used) are
deliberately separate concepts (configs/llm/models.yaml) so a model's
identity is never implicitly read as "better" than another's -- see the
design doc §4. This file grows across Tasks 8-11: roster loading and the
OpenRouter preflight check here; the actual chat-completion call, retry,
and fallback-chain logic are added in Tasks 9-10.
"""

import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
from pydantic import BaseModel

from src.llm.decision_schema import Decision
from src.llm.switch_elicitation import SwitchDecision
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

MODELS_CONFIG_PATH = CONFIG_ROOT / "llm" / "models.yaml"
MODEL_ROSTER_FULL_PATH = CONFIG_ROOT / "llm" / "model_roster_full.yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelEntry(BaseModel):
    id: str
    label: str


class ReliabilityChain(BaseModel):
    primary: str
    fallbacks: list[str]


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


_cumulative_usage = LLMUsage()
_cumulative_usage_lock = threading.Lock()


def get_cumulative_usage() -> LLMUsage:
    with _cumulative_usage_lock:
        return _cumulative_usage.model_copy()


def reset_cumulative_usage() -> None:
    global _cumulative_usage
    with _cumulative_usage_lock:
        _cumulative_usage = LLMUsage()


def _record_usage(response: httpx.Response) -> None:
    """Accumulates one response's usage into `_cumulative_usage` under
    `_cumulative_usage_lock` -- `call_model` runs concurrently across
    worker threads when `run_timestep(max_workers>1)` is in effect
    (Plan 6a), and the read-modify-write this does (`_cumulative_usage +
    _parse_usage(response)`) is not atomic without the lock: measured
    directly, 16 threads x 200 calls lost ~2.5% of total_tokens to a lost
    update before this lock was added. Records only SUCCESSFUL parses --
    a call that exhausts retries or fails repair records nothing, so this
    total is a lower bound on tokens actually billed, not an exact count."""
    global _cumulative_usage
    usage = _parse_usage(response)
    with _cumulative_usage_lock:
        _cumulative_usage = _cumulative_usage + usage


def _parse_usage(response: httpx.Response) -> LLMUsage:
    body = response.json()
    usage_block = body.get("usage") or {}
    return LLMUsage(
        prompt_tokens=usage_block.get("prompt_tokens", 0),
        completion_tokens=usage_block.get("completion_tokens", 0),
        total_tokens=usage_block.get("total_tokens", 0),
    )


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


class ModelCandidate(BaseModel):
    id: str
    label: str
    name: str


class ModelCandidateRoster(BaseModel):
    models: list[ModelCandidate]


def load_model_candidate_roster(path: Path = MODEL_ROSTER_FULL_PATH) -> ModelCandidateRoster:
    return load_yaml_as(path, ModelCandidateRoster)


def verify_model_candidates(candidate_ids: list[str], client: httpx.Client) -> tuple[list[str], list[str]]:
    """Preflight check for a large candidate pool: unlike verify_model_roster
    (which raises on the first missing model, correct for Phase 2's small
    fixed roster), this collects every result and returns
    (available, unavailable) so the caller can exclude failures and report
    them, rather than aborting entirely on one stale ID."""
    response = client.get("/models")
    response.raise_for_status()
    available_ids = {entry["id"] for entry in response.json()["data"]}

    available = [candidate_id for candidate_id in candidate_ids if candidate_id in available_ids]
    unavailable = [candidate_id for candidate_id in candidate_ids if candidate_id not in available_ids]
    return available, unavailable


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
            decision = _parse_decision(response)
            _record_usage(response)
            return decision
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
                repaired_decision = _parse_decision(repair_response)
                _record_usage(repair_response)
                return repaired_decision
            except (KeyError, IndexError, ValueError) as repair_exc:
                last_error = f"malformed output, repair failed: {repair_exc}"
                retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
                continue

    raise ModelCallFailedError(model_id, last_error)


def _parse_switch_decision(response: httpx.Response) -> SwitchDecision:
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return SwitchDecision.model_validate_json(content)


def call_model_for_switch(
    prompt: str,
    model_id: str,
    client: httpx.Client,
    retry_config: RetryConfig | None = None,
) -> SwitchDecision:
    """Same 3-tier failure handling as call_model (technical-failure retry,
    one repair reprompt), targeting SwitchDecision instead of Decision --
    a yes/no switch question has no economic-validity tier to check."""
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
            decision = _parse_switch_decision(response)
            _record_usage(response)
            return decision
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
                repaired_decision = _parse_switch_decision(repair_response)
                _record_usage(repair_response)
                return repaired_decision
            except (KeyError, IndexError, ValueError) as repair_exc:
                last_error = f"malformed output, repair failed: {repair_exc}"
                retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
                continue

    raise ModelCallFailedError(model_id, last_error)


class LLMCallResult(BaseModel):
    requested_model: str
    actual_model: str
    fallback_used: bool
    fallback_reason: str | None
    model_attempts: list[str]
    decision: Decision


class AllModelsFailedError(Exception):
    def __init__(self, model_ids: list[str], last_reason: str):
        self.model_ids = model_ids
        self.last_reason = last_reason
        super().__init__(f"All models in the chain failed: {model_ids}; last reason: {last_reason}")


def call_with_fallback_chain(
    prompt: str,
    model_ids: list[str],
    client: httpx.Client,
    retry_config: RetryConfig | None = None,
) -> LLMCallResult:
    """Try model_ids in order, stopping at the first success. model_ids[0] is
    the requested model; later entries are only tried once an earlier one
    exhausts its own retries/repair attempts inside call_model.

    Used by the default_reliability_chain routing policy. NOT used by the
    model_comparison policy, which calls call_model directly per pinned
    model with no substitution -- see Task 23's experiment_007, which must
    keep "model" a clean experimental factor rather than confounding it with
    reliability.
    """
    if not model_ids:
        raise ValueError("model_ids must contain at least one model")

    requested_model = model_ids[0]
    attempts: list[str] = []
    last_reason = "no models attempted"

    for index, model_id in enumerate(model_ids):
        attempts.append(model_id)
        try:
            decision = call_model(prompt, model_id, client, retry_config)
        except ModelCallFailedError as exc:
            last_reason = exc.reason
            continue

        return LLMCallResult(
            requested_model=requested_model,
            actual_model=model_id,
            fallback_used=index > 0,
            fallback_reason=last_reason if index > 0 else None,
            model_attempts=attempts,
            decision=decision,
        )

    raise AllModelsFailedError(model_ids, last_reason)
