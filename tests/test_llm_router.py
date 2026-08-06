import json

import httpx
import pytest

from src.llm.decision_schema import Decision, DecisionAction
from src.llm.llm_router import (
    AllModelsFailedError,
    AuthenticationError,
    ModelCallFailedError,
    ModelNotAvailableError,
    OPENROUTER_BASE_URL,
    RetryConfig,
    call_model,
    call_with_fallback_chain,
    get_cumulative_usage,
    load_model_roster,
    reset_cumulative_usage,
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


def _mock_client_with_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _chat_response(_decision_json())
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        return httpx.Response(200, json=body)

    return httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))


def test_call_model_captures_token_usage_and_accumulates_across_calls():
    reset_cumulative_usage()
    client = _mock_client_with_usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    call_model("some prompt", "vendor/fake-model", client)

    usage = get_cumulative_usage()
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150

    call_model("some prompt", "vendor/fake-model", client)
    usage_after_second_call = get_cumulative_usage()
    assert usage_after_second_call.total_tokens == 300


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


def test_fallback_chain_uses_primary_when_it_succeeds():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    result = call_with_fallback_chain("prompt", ["model-a", "model-b"], client, RetryConfig(sleep_fn=lambda s: None))

    assert result.requested_model == "model-a"
    assert result.actual_model == "model-a"
    assert result.fallback_used is False
    assert result.fallback_reason is None
    assert result.model_attempts == ["model-a"]


def test_fallback_chain_falls_through_when_primary_exhausts_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "model-a":
            return httpx.Response(500, json={"error": "down"})
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    result = call_with_fallback_chain(
        "prompt", ["model-a", "model-b"], client, RetryConfig(max_retries=1, sleep_fn=lambda s: None)
    )

    assert result.requested_model == "model-a"
    assert result.actual_model == "model-b"
    assert result.fallback_used is True
    assert result.fallback_reason == "HTTP 500"
    assert result.model_attempts == ["model-a", "model-b"]


def test_fallback_chain_raises_when_every_model_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(AllModelsFailedError):
        call_with_fallback_chain(
            "prompt", ["model-a", "model-b"], client, RetryConfig(max_retries=1, sleep_fn=lambda s: None)
        )


def test_fallback_chain_propagates_auth_error_without_trying_other_models():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationError):
        call_with_fallback_chain(
            "prompt", ["model-a", "model-b"], client, RetryConfig(max_retries=3, sleep_fn=lambda s: None)
        )

    assert calls["count"] == 1


def test_fallback_chain_rejects_empty_model_list():
    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with pytest.raises(ValueError):
        call_with_fallback_chain("prompt", [], client)


def test_loads_full_model_candidate_roster():
    from src.llm.llm_router import load_model_candidate_roster

    roster = load_model_candidate_roster()
    assert len(roster.models) == 99
    ids = [entry.id for entry in roster.models]
    assert len(set(ids)) == 99  # all unique
    labels = [entry.label for entry in roster.models]
    assert len(set(labels)) == 99  # all unique
    assert any(entry.name == "GPT-5" for entry in roster.models)
    assert any(entry.name == "WizardLM" for entry in roster.models)


def test_verify_model_candidates_returns_all_available():
    from src.llm.llm_router import verify_model_candidates, load_model_candidate_roster

    roster = load_model_candidate_roster()
    all_ids = [entry.id for entry in roster.models]
    client = _client_with_models(all_ids)

    available, unavailable = verify_model_candidates(all_ids, client)

    assert set(available) == set(all_ids)
    assert unavailable == []


def test_verify_model_candidates_collects_all_failures_without_raising():
    from src.llm.llm_router import verify_model_candidates, load_model_candidate_roster

    roster = load_model_candidate_roster()
    all_ids = [entry.id for entry in roster.models]
    # Simulate 3 stale/deprecated IDs missing from OpenRouter's live roster.
    missing = {all_ids[0], all_ids[1], all_ids[2]}
    present_ids = [i for i in all_ids if i not in missing]
    client = _client_with_models(present_ids)

    available, unavailable = verify_model_candidates(all_ids, client)

    assert set(unavailable) == missing
    assert set(available) == set(present_ids)
    assert len(available) + len(unavailable) == len(all_ids)


def test_verify_model_candidates_preserves_input_order_in_available():
    from src.llm.llm_router import verify_model_candidates

    client = _client_with_models(["b", "a", "c"])

    available, unavailable = verify_model_candidates(["a", "b", "c"], client)

    assert available == ["a", "b", "c"]
    assert unavailable == []
