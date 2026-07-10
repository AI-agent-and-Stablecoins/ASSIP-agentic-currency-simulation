"""Transaction cost lookups, used directly in utility calculations."""

from src.blockchain.chain import ChainConfig


def get_gas_fee(chain: ChainConfig, tx_complexity: float = 1.0) -> float:
    """Return the USD cost of a transaction on this chain."""
    return chain.gas_fee * tx_complexity
