from src.blockchain.chain import load_chain_universe
from src.currencies.synthetic_hypothesis_currencies import (
    NEUTRAL_FIXED_VALUES,
    SYNTHETIC_CHAINS,
    SYNTHETIC_DIMENSION_PAIRS,
    build_synthetic_hypothesis_currencies,
)


def test_synthetic_chains_has_exactly_3_entries_with_the_right_gas_fees():
    assert set(SYNTHETIC_CHAINS.keys()) == {"synthetic_gas_low", "synthetic_gas_mid", "synthetic_gas_high"}
    gas_fees = {name: chain.gas_fee for name, chain in SYNTHETIC_CHAINS.items()}
    assert gas_fees == {
        "synthetic_gas_low": 0.01,
        "synthetic_gas_mid": 0.05,
        "synthetic_gas_high": 0.10,
    }


def test_synthetic_chains_never_leak_into_the_real_chain_universe():
    real_chains = load_chain_universe()
    assert set(real_chains.keys()) == {"ethereum", "arbitrum", "base", "solana"}
    assert not set(real_chains.keys()) & set(SYNTHETIC_CHAINS.keys())


def test_h1_has_exactly_3_currencies_one_per_medium_level_pinned_to_gas_mid():
    currencies, chain_pins = build_synthetic_hypothesis_currencies("H1")

    assert len(currencies) == 3
    pegs = {config.peg for config in currencies.values()}
    assert pegs == {"USD", "EUR", "XAU"}

    assert set(chain_pins.keys()) == set(currencies.keys())
    assert set(chain_pins.values()) == {"synthetic_gas_mid"}


def test_h3_governance_x_liquidity_has_6_currencies_with_untested_dims_held_constant():
    currencies, chain_pins = build_synthetic_hypothesis_currencies("H3")

    assert len(currencies) == 6

    pegs = {config.peg for config in currencies.values()}
    peg_errors = {config.peg_error for config in currencies.values()}
    issuer_risks = {config.issuer_risk for config in currencies.values()}
    assert pegs == {NEUTRAL_FIXED_VALUES["medium"]}
    assert peg_errors == {NEUTRAL_FIXED_VALUES["volatility"]}
    assert len(issuer_risks) == 1

    combos = {(config.governance_score, config.bid_ask_spread) for config in currencies.values()}
    assert len(combos) == 6
    governance_values = {c[0] for c in combos}
    liquidity_values = {c[1] for c in combos}
    assert governance_values == {0.0, 1.0}
    assert liquidity_values == {0.0001, 0.0005, 0.0010}

    assert set(chain_pins.keys()) == set(currencies.keys())
    assert set(chain_pins.values()) == {"synthetic_gas_mid"}


def test_h6_medium_x_liquidity_has_9_currencies():
    currencies, chain_pins = build_synthetic_hypothesis_currencies("H6")

    assert len(currencies) == 9
    assert set(chain_pins.keys()) == set(currencies.keys())


def test_symbols_are_unique_within_every_hypothesis_grid():
    for hypothesis in SYNTHETIC_DIMENSION_PAIRS:
        currencies, chain_pins = build_synthetic_hypothesis_currencies(hypothesis)
        assert len(currencies) == len(set(currencies.keys()))
        for symbol, config in currencies.items():
            assert config.symbol == symbol


def test_every_chain_pin_resolves_to_a_real_synthetic_chain_with_the_correct_gas_fee():
    for hypothesis, tested_dimensions in SYNTHETIC_DIMENSION_PAIRS.items():
        currencies, chain_pins = build_synthetic_hypothesis_currencies(hypothesis)

        for symbol in currencies:
            chain_name = chain_pins[symbol]
            assert chain_name in SYNTHETIC_CHAINS

        if "gas_fee" not in tested_dimensions:
            # gas fee isn't tested by this hypothesis -- every coin must be
            # pinned to the neutral mid-gas chain.
            assert set(chain_pins.values()) == {"synthetic_gas_mid"}
        else:
            # gas fee is tested -- all 3 synthetic chains must appear.
            assert set(chain_pins.values()) == set(SYNTHETIC_CHAINS.keys())


def test_every_hypothesis_grid_size_matches_the_spec_table():
    expected_sizes = {
        "H1": 3,
        "H2": 6,
        "H3": 6,
        "H4": 6,
        "H5": 6,
        "H6": 9,
        "H7": 9,
        "H8": 9,
        "H9": 9,
        "H10": 9,
        "H11": 9,
    }
    for hypothesis, expected_size in expected_sizes.items():
        currencies, _ = build_synthetic_hypothesis_currencies(hypothesis)
        assert len(currencies) == expected_size
