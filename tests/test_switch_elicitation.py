from src.llm.agent_reasoning import AgentUtilityContext
from src.llm.llm_router import call_model_for_switch
from src.llm.switch_elicitation import SwitchDecision, render_switch_prompt
from tests.llm_test_helpers import mock_openrouter_client


def _context():
    return AgentUtilityContext(
        agent_id="consumer-seed0-000",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=2.0,
        eis=None,
        multi_attribute_weights=None,
        wallet_balances={"USDT": 100.0, "TDUSD": 50.0},
        currency_zone="USD",
        assigned_model="vendor/model",
        cara_coefficient=None,
    )


def test_render_switch_prompt_includes_both_coins_and_the_values():
    prompt = render_switch_prompt(
        _context(),
        fixed_symbol="USDT",
        fixed_field="liquidity_score",
        fixed_value=0.98,
        varied_symbol="TDUSD",
        varied_field="liquidity_score",
        varied_value=0.50,
    )

    assert "USDT" in prompt
    assert "TDUSD" in prompt
    assert "0.98" in prompt
    assert "0.5" in prompt
    assert "will_switch" in prompt


def test_call_model_for_switch_parses_a_valid_response():
    client = mock_openrouter_client({"vendor/model": {"will_switch": True, "reasoning": "better liquidity"}})

    result = call_model_for_switch("some prompt", "vendor/model", client)

    assert isinstance(result, SwitchDecision)
    assert result.will_switch is True
    assert result.reasoning == "better liquidity"


def test_call_model_for_switch_raises_after_exhausting_retries_on_repeated_bad_json():
    import httpx
    import pytest

    from src.llm.llm_router import ModelCallFailedError, RetryConfig, OPENROUTER_BASE_URL

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    retry_config = RetryConfig(max_retries=2, backoff_base_seconds=0.0, sleep_fn=lambda _: None)

    with pytest.raises(ModelCallFailedError):
        call_model_for_switch("some prompt", "vendor/model", client, retry_config=retry_config)
