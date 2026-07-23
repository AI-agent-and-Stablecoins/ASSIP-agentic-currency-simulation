# tests/test_utility_epstein_zin.py
import math

import pytest

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.epstein_zin import EpsteinZinProxyUtility


def _option(**overrides) -> CurrencyChainOption:
    defaults = dict(
        currency_symbol="USDC",
        chain_name="ethereum",
        governance_score=0.95,
        liquidity_score=0.97,
        peg_error=0.0003,
        gas_fee=2.5,
        finality_seconds=12.0,
        genius_compliant=True,
    )
    defaults.update(overrides)
    return CurrencyChainOption(**defaults)


def test_log_utility_at_unit_risk_aversion():
    utility = EpsteinZinProxyUtility(risk_aversion=1.0, eis=1.0)
    option = _option()

    safety = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
    effective_wealth = 100.0 * safety - option.gas_fee * (1.0 / 1.0)

    assert utility.evaluate(option, wealth=100.0) == pytest.approx(math.log(effective_wealth))


def test_zero_or_negative_eis_is_rejected():
    with pytest.raises(ValueError):
        EpsteinZinProxyUtility(risk_aversion=2.0, eis=0.0)
    with pytest.raises(ValueError):
        EpsteinZinProxyUtility(risk_aversion=2.0, eis=-1.0)


def test_risk_aversion_shapes_curvature_independently_of_eis():
    """Varying gamma (risk_aversion) with psi (eis) fixed must (a) always
    rank the safer option above the riskier one -- CRRA is a strictly
    increasing transform of a single effective-wealth scalar, so gamma can
    never flip an ordering, only rescale the numbers -- and (b) still
    measurably change the utility values, proving gamma is not a no-op.
    This is the design's independent-testability requirement for gamma."""
    low_gamma = EpsteinZinProxyUtility(risk_aversion=0.5, eis=1.0)
    high_gamma = EpsteinZinProxyUtility(risk_aversion=5.0, eis=1.0)

    safer = _option(governance_score=0.99, liquidity_score=0.99, peg_error=0.0001)
    riskier = _option(governance_score=0.60, liquidity_score=0.60, peg_error=0.02)

    low_gap = low_gamma.evaluate(safer, wealth=100.0) - low_gamma.evaluate(riskier, wealth=100.0)
    high_gap = high_gamma.evaluate(safer, wealth=100.0) - high_gamma.evaluate(riskier, wealth=100.0)

    assert low_gap > 0
    assert high_gap > 0
    assert low_gap != pytest.approx(high_gap)


def test_eis_shapes_fee_sensitivity_independently_of_risk_aversion():
    """Varying psi (eis) with gamma (risk_aversion) fixed must change the gap
    between a cheap and an expensive option, while a fixed gamma keeps the
    safety-driven ordering direction unchanged -- the design's independent-
    testability requirement for psi."""
    reluctant_to_substitute = EpsteinZinProxyUtility(risk_aversion=2.0, eis=0.5)
    happy_to_substitute = EpsteinZinProxyUtility(risk_aversion=2.0, eis=5.0)

    cheap = _option(gas_fee=0.5)
    expensive = _option(gas_fee=20.0)

    low_eis_gap = reluctant_to_substitute.evaluate(cheap, wealth=100.0) - reluctant_to_substitute.evaluate(
        expensive, wealth=100.0
    )
    high_eis_gap = happy_to_substitute.evaluate(cheap, wealth=100.0) - happy_to_substitute.evaluate(
        expensive, wealth=100.0
    )

    # Low EIS (reluctant to substitute) must penalize the fee gap more
    # harshly than high EIS (happy to substitute).
    assert low_eis_gap > high_eis_gap > 0
