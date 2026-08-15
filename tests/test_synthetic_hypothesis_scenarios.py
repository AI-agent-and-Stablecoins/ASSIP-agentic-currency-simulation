from src.currencies.synthetic_hypothesis_currencies import build_synthetic_hypothesis_currencies
from src.economy.synthetic_hypothesis_scenarios import (
    SyntheticHypothesisCellSpec,
    build_synthetic_hypothesis_cell_specs,
)

_ALL_HYPOTHESES = {f"H{i}" for i in range(1, 12)}


def test_returns_exactly_11_specs():
    specs = build_synthetic_hypothesis_cell_specs()
    assert len(specs) == 11


def test_every_spec_hypothesis_is_h1_through_h11_all_distinct():
    specs = build_synthetic_hypothesis_cell_specs()
    hypotheses = [spec.hypothesis for spec in specs]

    assert set(hypotheses) == _ALL_HYPOTHESES
    assert len(hypotheses) == len(set(hypotheses))


def test_every_spec_key_is_unique_and_follows_the_synthetic_suffix_pattern():
    specs = build_synthetic_hypothesis_cell_specs()
    keys = [spec.key for spec in specs]

    assert len(keys) == len(set(keys))
    for spec in specs:
        assert spec.key == f"{spec.hypothesis}_synthetic"
        assert spec.key.endswith("_synthetic")

    assert "H1_synthetic" in keys
    assert "H3_synthetic" in keys


def test_h3_spec_currencies_and_chain_pins_match_the_builder_directly():
    specs = build_synthetic_hypothesis_cell_specs()
    h3_spec = next(spec for spec in specs if spec.hypothesis == "H3")

    expected_currencies, expected_chain_pins = build_synthetic_hypothesis_currencies("H3")

    assert h3_spec.currencies.keys() == expected_currencies.keys()
    for symbol, expected_config in expected_currencies.items():
        assert h3_spec.currencies[symbol] == expected_config
    assert h3_spec.chain_pins == expected_chain_pins


def test_spec_is_frozen_dataclass_instance():
    specs = build_synthetic_hypothesis_cell_specs()
    assert isinstance(specs[0], SyntheticHypothesisCellSpec)
