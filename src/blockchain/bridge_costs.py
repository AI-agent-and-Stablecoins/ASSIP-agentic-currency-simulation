"""Cost of moving a currency between chains -- relevant to cross-border transactions."""

from src.utils.helpers import clamp


def bridge_cost(currency_liquidity_score: float, from_chain_gas_fee: float, to_chain_gas_fee: float) -> float:
    """Approximate USD cost of bridging: both chains' gas fees, inflated by an
    illiquidity penalty (thinner liquidity means wider bridging slippage)."""
    illiquidity_penalty = 1.0 + (1.0 - clamp(currency_liquidity_score, 0.0, 1.0))
    return (from_chain_gas_fee + to_chain_gas_fee) * illiquidity_penalty
