import httpx

from src.llm.market_intelligence import POLYGON_BASE_URL, CurrencyProfile, fetch_live_price, load_currency_profile

_ALL_SYMBOLS = ["DAI", "EURC", "EURT", "FDUSD", "PAXG", "Tokenized_Deposits", "USDC", "USDT", "XAUT"]


def test_loads_every_currency_profile_file():
    for symbol in _ALL_SYMBOLS:
        profile = load_currency_profile(symbol)
        assert profile is not None
        assert isinstance(profile, CurrencyProfile)
        assert profile.symbol == symbol
        assert profile.executive_summary
        assert profile.source == "deep-research-report.md"


def test_missing_profile_returns_none_not_an_exception():
    assert load_currency_profile("NOTACOIN") is None


def test_usdc_profile_has_expected_timeline_entries():
    profile = load_currency_profile("USDC")
    assert profile is not None
    assert len(profile.timeline) >= 3
    assert any("Circle" in event.event or "Coinbase" in event.event for event in profile.timeline)


def test_fetch_live_price_returns_price_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"c": 1.0002}]})

    client = httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))

    snapshot = fetch_live_price("X:USDCUSD", client)

    assert snapshot.price == 1.0002
    assert snapshot.unavailable_reason is None
    assert snapshot.ticker == "X:USDCUSD"


def test_fetch_live_price_degrades_gracefully_when_polygon_has_no_data():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))

    snapshot = fetch_live_price("X:NOTATICKER", client)

    assert snapshot.price is None
    assert snapshot.unavailable_reason is not None


def test_fetch_live_price_degrades_gracefully_on_http_error_rather_than_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    client = httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))

    snapshot = fetch_live_price("X:USDCUSD", client)

    assert snapshot.price is None
    assert snapshot.unavailable_reason is not None


def test_fetch_live_price_degrades_gracefully_on_malformed_json_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json{{{")

    client = httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))

    snapshot = fetch_live_price("X:USDCUSD", client)

    assert snapshot.price is None
    assert snapshot.unavailable_reason is not None


def test_fetch_live_price_degrades_gracefully_on_unexpected_response_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"no_close_field": True}]})

    client = httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))

    snapshot = fetch_live_price("X:USDCUSD", client)

    assert snapshot.price is None
    assert snapshot.unavailable_reason is not None
