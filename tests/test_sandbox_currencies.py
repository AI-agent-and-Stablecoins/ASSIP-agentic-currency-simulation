from src.currencies.currency import AssetClass
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS

EXPECTED_SANDBOX_NAMES = {
    "liquidity_vs_governance",
    "governance_vs_stability",
    "liquidity_vs_stability",
    "asset_backing_vs_liquidity",
    "asset_backing_vs_stability",
    "asset_backing_vs_governance",
}


def test_all_six_sandboxes_present():
    assert set(SANDBOX_CURRENCY_PAIRS.keys()) == EXPECTED_SANDBOX_NAMES


def test_every_pair_has_two_distinct_symbols():
    for name, (option_a, option_b) in SANDBOX_CURRENCY_PAIRS.items():
        assert option_a.symbol != option_b.symbol, f"{name}: symbols must differ"


def test_liquidity_vs_governance_isolates_exactly_those_two_dimensions():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_governance"]
    assert option_a.liquidity_score == 0.99
    assert option_b.liquidity_score == 0.90
    assert option_a.governance_score == 0.55
    assert option_b.governance_score == 0.95
    assert option_a.genius_compliant is False
    assert option_b.genius_compliant is True
    assert option_a.liquidity_score != option_b.liquidity_score
    assert option_a.governance_score != option_b.governance_score
    assert option_a.peg_error == option_b.peg_error  # held constant
    assert option_a.issuer_risk == option_b.issuer_risk  # held constant
    assert option_a.asset_class == option_b.asset_class  # held constant
    assert option_a.peg == option_b.peg == "USD"


def test_governance_vs_stability_isolates_exactly_those_two_dimensions():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["governance_vs_stability"]
    assert option_a.governance_score == 0.95
    assert option_b.governance_score == 0.55
    assert option_a.genius_compliant is True
    assert option_b.genius_compliant is False
    assert option_a.peg_error == 0.02
    assert option_b.peg_error == 0.0001
    assert option_a.governance_score != option_b.governance_score
    assert option_a.peg_error != option_b.peg_error
    assert option_a.liquidity_score == option_b.liquidity_score  # held constant
    assert option_a.issuer_risk == option_b.issuer_risk  # held constant
    assert option_a.asset_class == option_b.asset_class  # held constant
    assert option_a.peg == option_b.peg == "USD"


def test_liquidity_vs_stability_isolates_exactly_those_two_dimensions():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_stability"]
    assert option_a.liquidity_score == 0.99
    assert option_b.liquidity_score == 0.75
    assert option_a.peg_error == 0.04
    assert option_b.peg_error == 0.0001
    assert option_a.liquidity_score != option_b.liquidity_score
    assert option_a.peg_error != option_b.peg_error
    assert option_a.governance_score == option_b.governance_score  # held constant
    assert option_a.issuer_risk == option_b.issuer_risk  # held constant
    assert option_a.genius_compliant == option_b.genius_compliant  # held constant
    assert option_a.asset_class == option_b.asset_class  # held constant
    assert option_a.peg == option_b.peg == "USD"


def test_asset_backing_vs_liquidity_isolates_exactly_those_two_dimensions():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_liquidity"]
    assert option_a.asset_class == AssetClass.GOLD_BACKED
    assert option_b.asset_class == AssetClass.STABLECOIN
    assert option_a.liquidity_score == 0.70
    assert option_b.liquidity_score == 0.99
    assert option_a.asset_class != option_b.asset_class
    assert option_a.liquidity_score != option_b.liquidity_score
    assert option_a.governance_score == option_b.governance_score  # held constant
    assert option_a.peg_error == option_b.peg_error  # held constant
    assert option_a.issuer_risk == option_b.issuer_risk  # held constant
    assert option_a.genius_compliant == option_b.genius_compliant  # held constant


def test_asset_backing_vs_stability_isolates_exactly_those_two_dimensions():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_stability"]
    assert option_a.asset_class == AssetClass.GOLD_BACKED
    assert option_b.asset_class == AssetClass.TOKENIZED_DEPOSIT
    assert option_a.peg_error == 0.015
    assert option_b.peg_error == 0.0001
    assert option_a.asset_class != option_b.asset_class
    assert option_a.peg_error != option_b.peg_error
    assert option_a.governance_score == option_b.governance_score  # held constant
    assert option_a.liquidity_score == option_b.liquidity_score  # held constant
    assert option_a.issuer_risk == option_b.issuer_risk  # held constant
    assert option_a.genius_compliant == option_b.genius_compliant  # held constant


def test_asset_backing_vs_governance_isolates_exactly_those_dimensions():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_governance"]
    assert option_a.asset_class == AssetClass.TOKENIZED_DEPOSIT
    assert option_b.asset_class == AssetClass.STABLECOIN
    assert option_a.governance_score == 0.75
    assert option_b.governance_score == 0.70
    assert option_a.issuer_risk == 0.25
    assert option_b.issuer_risk == 0.20
    assert option_a.asset_class != option_b.asset_class
    assert option_a.governance_score != option_b.governance_score
    assert option_a.issuer_risk != option_b.issuer_risk
    assert option_a.liquidity_score == option_b.liquidity_score  # held constant
    assert option_a.peg_error == option_b.peg_error  # held constant
    assert option_a.genius_compliant == option_b.genius_compliant  # held constant


def test_all_configs_are_valid_pydantic_models_with_correct_subclass():
    from src.currencies.gold_token import GoldBackedConfig
    from src.currencies.stablecoin import StablecoinConfig
    from src.currencies.tokenized_deposit import TokenizedDepositConfig

    subclass_by_asset_class = {
        AssetClass.STABLECOIN: StablecoinConfig,
        AssetClass.GOLD_BACKED: GoldBackedConfig,
        AssetClass.TOKENIZED_DEPOSIT: TokenizedDepositConfig,
    }
    for option_a, option_b in SANDBOX_CURRENCY_PAIRS.values():
        for option in (option_a, option_b):
            assert isinstance(option, subclass_by_asset_class[option.asset_class])
