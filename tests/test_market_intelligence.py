from src.llm.market_intelligence import CurrencyProfile, load_currency_profile

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
