from src.currencies.currency import AssetClass, load_currency_universe
from src.currencies.stablecoin import StablecoinConfig


def test_load_currency_universe_defaults_bid_ask_spread_to_none():
    currencies = load_currency_universe()
    assert currencies
    for symbol, config in currencies.items():
        assert config.bid_ask_spread is None, f"{symbol}: expected bid_ask_spread to default to None"


def test_stablecoin_config_bid_ask_spread_round_trips():
    config = StablecoinConfig(
        symbol="SYN_TEST",
        asset_class=AssetClass.STABLECOIN,
        peg="USD",
        governance_score=0.5,
        liquidity_score=0.5,
        peg_error=0.001,
        issuer_risk=0.10,
        genius_compliant=True,
        bid_ask_spread=0.0001,
        redemption_mechanism="Test redemption mechanism",
    )

    assert config.bid_ask_spread == 0.0001
