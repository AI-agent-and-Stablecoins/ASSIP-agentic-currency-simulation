import httpx
import pytest

from src.llm.llm_router import ModelNotAvailableError, load_model_roster, verify_model_roster


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
