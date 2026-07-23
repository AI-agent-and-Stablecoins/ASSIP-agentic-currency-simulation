import json

import httpx
import pytest

from src.llm.decision_schema import Decision, DecisionAction
from src.llm.llm_router import (
    AuthenticationError,
    ModelCallFailedError,
    ModelNotAvailableError,
    OPENROUTER_BASE_URL,
    RetryConfig,
    call_model,
    load_model_roster,
    verify_model_roster,
)


def _client_with_models(available_ids: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models"
        return httpx.Response(200, json={"data": [{"id": model_id} for model_id in available_ids]})

    return httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))


def test_loads_roster_from_config():
    roster = load_model_roster()
    labels = {entry.label for entry in roster.models}
    assert labels == {"claude-sonnet-5", "gpt-5.6-luna", "deepseek-v4-pro", "gemini-3.5-flash-lite", "perplexity-sonar"}
    assert roster.routing_policies.default_reliability_chain.primary == "claude-sonnet-5"
    assert roster.routing_policies.model_comparison.pinned_models == [
        "claude-sonnet-5",
        "gpt-5.6-luna",
        "deepseek-v4-pro",
        "gemini-3.5-flash-lite",
        "perplexity-sonar",
    ]


def test_resolve_looks_up_id_by_label():
    roster = load_model_roster()
    assert roster.resolve("claude-sonnet-5") == "anthropic/claude-sonnet-5"


def test_resolve_raises_for_unknown_label():
    roster = load_model_roster()
    with pytest.raises(ValueError):
        roster.resolve("not-a-real-model")


def test_verify_model_roster_passes_when_all_ids_available():
    roster = load_model_roster()
    all_ids = [entry.id for entry in roster.models]
    client = _client_with_models(all_ids)

    verify_model_roster(roster, client)  # must not raise


def test_verify_model_roster_fails_loudly_on_missing_model():
    roster = load_model_roster()
    ids_missing_one = [entry.id for entry in roster.models if entry.label != "gpt-5.6-luna"]
    client = _client_with_models(ids_missing_one)

    with pytest.raises(ModelNotAvailableError) as exc_info:
        verify_model_roster(roster, client)

    assert exc_info.value.label == "gpt-5.6-luna"
    assert exc_info.value.model_id == "openai/gpt-5.6-luna"


def _decision_json(action: str = "OFFER") -> str:
    return json.dumps(
        {
            "action": action,
            "proposed_currency": "USDC",
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 100.0,
            "reasoning": "test",
        }
    )


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_call_model_succeeds_on_first_try():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert isinstance(decision, Decision)
    assert decision.proposed_currency == "USDC"


def test_call_model_retries_on_429_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert decision.action == DecisionAction.OFFER
    assert calls["count"] == 2


def test_call_model_retries_on_timeout_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.TimeoutException("timed out")
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert decision.action == DecisionAction.OFFER


def test_call_model_gives_up_after_persistent_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ModelCallFailedError):
        call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(max_retries=2, sleep_fn=lambda s: None))


def test_call_model_aborts_immediately_on_auth_failure_without_retrying():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "invalid api key"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationError):
        call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(max_retries=3, sleep_fn=lambda s: None))

    assert calls["count"] == 1


def test_call_model_repairs_malformed_json_on_first_attempt():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, json=_chat_response("not valid json"))
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert decision.action == DecisionAction.OFFER
    assert calls["count"] == 2


def test_call_model_gives_up_when_repair_also_fails_repeatedly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response("still not valid json"))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ModelCallFailedError):
        call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(max_retries=2, sleep_fn=lambda s: None))
