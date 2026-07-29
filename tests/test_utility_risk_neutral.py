from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.risk_neutral import RiskNeutralUtility


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


def test_risk_neutral_is_linear_in_wealth():
    utility = RiskNeutralUtility()
    option = _option()

    u_100 = utility.evaluate(option, wealth=100.0)
    u_200 = utility.evaluate(option, wealth=200.0)

    # Linear (no curvature): doubling wealth must double the wealth-driven
    # component exactly, since the safety multiplier and gas fee are constant.
    safety = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
    assert u_200 - u_100 == (200.0 - 100.0) * safety


def test_risk_neutral_subtracts_gas_fee_directly():
    utility = RiskNeutralUtility()
    cheap = _option(gas_fee=0.5)
    expensive = _option(gas_fee=5.0)

    assert utility.evaluate(cheap, wealth=100.0) > utility.evaluate(expensive, wealth=100.0)
    # The gap must equal exactly the gas fee difference (no hidden curvature
    # or extra penalty beyond the raw fee).
    assert utility.evaluate(cheap, wealth=100.0) - utility.evaluate(expensive, wealth=100.0) == 4.5


def test_risk_neutral_ranks_by_safety_adjusted_payoff_not_safety_alone():
    utility = RiskNeutralUtility()
    safer_but_costly = _option(governance_score=0.99, gas_fee=50.0)
    riskier_but_cheap = _option(governance_score=0.50, gas_fee=0.1)

    # Risk-neutral cares only about net payoff, so a large enough fee can beat
    # a governance edge -- unlike CRRA/CARA, which apply curvature that can
    # reverse this ranking.
    assert utility.evaluate(riskier_but_cheap, wealth=100.0) > utility.evaluate(safer_but_costly, wealth=100.0)
