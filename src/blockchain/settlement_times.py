"""Confirmation/finality time lookups."""

from src.blockchain.chain import ChainConfig


def get_finality(chain: ChainConfig) -> float:
    """Return seconds until a transaction on this chain is considered final."""
    return chain.finality_seconds
