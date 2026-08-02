from src.blockchain.routing_engine import CurrencyChainOption
from src.simulation.timestep import _spread_and_gas_optimal


def test_spread_and_gas_optimal_picks_highest_liquidity_and_lowest_gas():
    candidates = [
        CurrencyChainOption(
            currency_symbol="USDC", chain_name="ethereum", governance_score=0.9,
            liquidity_score=0.5, peg_error=0.0, gas_fee=5.0, finality_seconds=12.0,
            genius_compliant=True,
        ),
        CurrencyChainOption(
            currency_symbol="USDC", chain_name="solana", governance_score=0.9,
            liquidity_score=0.5, peg_error=0.0, gas_fee=0.01, finality_seconds=1.0,
            genius_compliant=True,
        ),
        CurrencyChainOption(
            currency_symbol="USDT", chain_name="ethereum", governance_score=0.6,
            liquidity_score=0.95, peg_error=0.01, gas_fee=5.0, finality_seconds=12.0,
            genius_compliant=False,
        ),
    ]

    spread_currency, spread_chain, gas_currency, gas_chain = _spread_and_gas_optimal(candidates)

    assert (spread_currency, spread_chain) == ("USDT", "ethereum")  # highest liquidity_score (0.95)
    assert (gas_currency, gas_chain) == ("USDC", "solana")  # lowest gas_fee (0.01)


def test_spread_and_gas_optimal_can_be_the_same_candidate():
    candidates = [
        CurrencyChainOption(
            currency_symbol="USDC", chain_name="solana", governance_score=0.9,
            liquidity_score=0.99, peg_error=0.0, gas_fee=0.01, finality_seconds=1.0,
            genius_compliant=True,
        ),
        CurrencyChainOption(
            currency_symbol="USDT", chain_name="ethereum", governance_score=0.6,
            liquidity_score=0.5, peg_error=0.01, gas_fee=5.0, finality_seconds=12.0,
            genius_compliant=False,
        ),
    ]

    spread_currency, spread_chain, gas_currency, gas_chain = _spread_and_gas_optimal(candidates)

    assert (spread_currency, spread_chain) == (gas_currency, gas_chain) == ("USDC", "solana")
