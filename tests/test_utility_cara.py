import math

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.cara import CARAUtility


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


def test_evaluate_does_not_overflow_for_negative_risk_aversion_at_realistic_wealth():
    utility = CARAUtility(risk_aversion=-1.0)
    option = _option()

    result = utility.evaluate(option, wealth=1300.0)

    assert math.isfinite(result)


def test_evaluate_preserves_ranking_under_clamping_for_extreme_values():
    utility = CARAUtility(risk_aversion=-1.0)
    higher_effective_wealth = _option(governance_score=0.99)
    lower_effective_wealth = _option(governance_score=0.50)

    u_higher = utility.evaluate(higher_effective_wealth, wealth=1300.0)
    u_lower = utility.evaluate(lower_effective_wealth, wealth=1300.0)

    # Once both saturate at the clamp boundary they may legitimately tie, so
    # this must be >=, not a strict >.
    assert u_higher >= u_lower
