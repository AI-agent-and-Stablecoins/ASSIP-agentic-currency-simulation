"""Shared httpx.MockTransport-backed client factories for LLM/Polygon tests.

Every hand-rolled MockTransport in tests/test_llm_router.py and
tests/test_market_intelligence.py builds the same two request/response
shapes: an OpenRouter chat-completion call (src/llm/llm_router.py's
_post_chat_completion/call_model) and a Polygon previous-close aggregate
call (src/llm/market_intelligence.py's fetch_live_price). Centralizing them
here means the many new LLM/Polygon-calling tests this plan adds don't each
re-derive those shapes from scratch.
"""

import json
import re

import httpx

from src.llm.llm_router import OPENROUTER_BASE_URL
from src.llm.market_intelligence import POLYGON_BASE_URL


def mock_openrouter_client(model_responses: dict[str, dict]) -> httpx.Client:
    """Build an httpx.Client whose transport fakes OpenRouter's
    /chat/completions endpoint for use with llm_router.call_model (and
    call_with_fallback_chain).

    `model_responses` maps a model_id to the Decision-shaped dict that
    model should "answer" with (the same fields test_llm_router.py's
    _decision_json() produces, as a dict rather than a JSON string). The
    handler reads the model_id the real call_model sent in its request
    body (see _post_chat_completion: {"model": ..., "messages": ...,
    "response_format": ...}) and echoes back the matching response wrapped
    in the OpenRouter chat-completion envelope
    ({"choices": [{"message": {"content": <json string>}}]}) that
    _parse_decision expects.

    A model_id with no entry in model_responses gets a 404, which
    call_model treats as ModelCallFailedError (an unmocked model is
    a test-authoring bug, not a technical failure worth retrying).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        body = json.loads(request.content)
        model_id = body["model"]
        if model_id not in model_responses:
            return httpx.Response(404, json={"error": f"no mocked response for model {model_id!r}"})
        content = json.dumps(model_responses[model_id])
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))


def mock_switch_threshold_client(
    field: str, threshold: float, higher_is_better: bool, model_id: str = "vendor/model"
) -> httpx.Client:
    """Build an httpx.Client whose transport fakes OpenRouter's chat-
    completion endpoint for use with llm_router.call_model_for_switch, with
    a THRESHOLD-DEPENDENT SwitchDecision -- unlike mock_openrouter_client,
    which can only ever return one constant response and so degenerates a
    binary search to one bound regardless of whether its direction logic is
    correct, silently masking bugs in it.

    Reads `field`'s value straight out of the rendered prompt's
    "Coin B (...): ...{field}=<value>" line (src.llm.switch_elicitation.
    render_switch_prompt always renders the varied field there) and answers
    will_switch=True on whichever side of `threshold` is more attractive:
    values >= threshold for a higher-is-better field (liquidity_score,
    governance_score), values <= threshold for a lower-is-better one
    (peg_error, gas_fee).
    """

    pattern = re.compile(rf"Coin B \([^)]*\):[^\n]*{re.escape(field)}=(-?[\d.]+)")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        body = json.loads(request.content)
        if body["model"] != model_id:
            return httpx.Response(404, json={"error": f"no mocked response for model {body['model']!r}"})
        prompt = body["messages"][0]["content"]
        match = pattern.search(prompt)
        assert match, f"could not find {field!r} in the Coin B block of prompt:\n{prompt}"
        varied_value = float(match.group(1))
        will_switch = varied_value >= threshold if higher_is_better else varied_value <= threshold
        content = json.dumps({"will_switch": will_switch, "reasoning": "test"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))


def mock_polygon_client(ticker_prices: dict[str, float]) -> httpx.Client:
    """Build an httpx.Client whose transport fakes Polygon's previous-close
    aggregate endpoint for use with market_intelligence.fetch_live_price.

    `ticker_prices` maps a ticker (e.g. "X:USDCUSD") to the price
    fetch_live_price should observe as `results[0]["c"]`. The handler
    parses the ticker out of the request path
    (/v2/aggs/ticker/{ticker}/prev, per fetch_live_price) and returns the
    matching {"results": [{"c": price}]} body.

    A ticker with no entry in ticker_prices gets {"results": []} -- the
    same "no data for this ticker" shape Polygon itself returns, which
    fetch_live_price already degrades gracefully on.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        prefix, suffix = "/v2/aggs/ticker/", "/prev"
        assert path.startswith(prefix) and path.endswith(suffix)
        ticker = path[len(prefix) : -len(suffix)]
        if ticker not in ticker_prices:
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"results": [{"c": ticker_prices[ticker]}]})

    return httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))
