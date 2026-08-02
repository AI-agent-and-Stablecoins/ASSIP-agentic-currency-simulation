"""Proves mock_openrouter_client/mock_polygon_client work end-to-end through
the real call_model/fetch_live_price functions -- not just that they build a
plausible-looking httpx.Client. See tests/llm_test_helpers.py."""

from src.llm.decision_schema import Decision, DecisionAction
from src.llm.llm_router import RetryConfig, call_model
from src.llm.market_intelligence import fetch_live_price
from tests.llm_test_helpers import mock_openrouter_client, mock_polygon_client


def _decision_dict(action: str = "OFFER") -> dict:
    return {
        "action": action,
        "proposed_currency": "USDC",
        "proposed_chain": "ethereum",
        "amount": 1.0,
        "price": 100.0,
        "reasoning": "test",
    }


def test_mock_openrouter_client_serves_model_specific_response_through_call_model():
    client = mock_openrouter_client({"anthropic/claude-sonnet-5": _decision_dict()})

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert isinstance(decision, Decision)
    assert decision.action == DecisionAction.OFFER
    assert decision.proposed_currency == "USDC"
    assert decision.proposed_chain == "ethereum"


def test_mock_openrouter_client_distinguishes_between_models():
    client = mock_openrouter_client(
        {
            "model-a": _decision_dict(action="ACCEPT"),
            "model-b": _decision_dict(action="REJECT"),
        }
    )

    decision_a = call_model("prompt", "model-a", client, RetryConfig(sleep_fn=lambda s: None))
    decision_b = call_model("prompt", "model-b", client, RetryConfig(sleep_fn=lambda s: None))

    assert decision_a.action == DecisionAction.ACCEPT
    assert decision_b.action == DecisionAction.REJECT


def test_mock_polygon_client_serves_ticker_specific_price_through_fetch_live_price():
    client = mock_polygon_client({"X:USDCUSD": 1.0002, "X:USDTUSD": 0.9998})

    snapshot_usdc = fetch_live_price("X:USDCUSD", client)
    snapshot_usdt = fetch_live_price("X:USDTUSD", client)

    assert snapshot_usdc.price == 1.0002
    assert snapshot_usdc.unavailable_reason is None
    assert snapshot_usdc.ticker == "X:USDCUSD"
    assert snapshot_usdt.price == 0.9998


def test_mock_polygon_client_returns_no_data_snapshot_for_unmocked_ticker():
    client = mock_polygon_client({"X:USDCUSD": 1.0002})

    snapshot = fetch_live_price("X:NOTATICKER", client)

    assert snapshot.price is None
    assert snapshot.unavailable_reason is not None
