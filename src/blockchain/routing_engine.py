"""Generates candidate (currency, chain) pairs for an agent to evaluate.

Takes plain currency-symbol -> balance mappings rather than a Wallet object,
so this module has no dependency on src/agents -- blockchain sits below
agents in the module dependency order.
"""

from pydantic import BaseModel

from src.blockchain.chain import ChainConfig
from src.blockchain.gas_fees import get_gas_fee
from src.blockchain.liquidity_pools import LiquidityPoolRegistry
from src.currencies.currency import CurrencyConfig


class CurrencyChainOption(BaseModel):
    currency_symbol: str
    chain_name: str
    governance_score: float
    liquidity_score: float
    peg_error: float
    gas_fee: float
    finality_seconds: float
    genius_compliant: bool


def generate_candidates(
    available_balances: dict[str, float],
    currencies: dict[str, CurrencyConfig],
    chains: dict[str, ChainConfig],
    liquidity_pools: LiquidityPoolRegistry | None = None,
) -> list[CurrencyChainOption]:
    """One candidate per (currency the agent holds a positive balance of) x (chain)."""
    liquidity_pools = liquidity_pools or LiquidityPoolRegistry()
    options: list[CurrencyChainOption] = []
    for symbol, balance in available_balances.items():
        if balance <= 0 or symbol not in currencies:
            continue
        currency = currencies[symbol]
        for chain in chains.values():
            options.append(
                CurrencyChainOption(
                    currency_symbol=symbol,
                    chain_name=chain.name,
                    governance_score=currency.governance_score,
                    liquidity_score=liquidity_pools.get_liquidity(currency, chain.name),
                    peg_error=currency.peg_error,
                    gas_fee=get_gas_fee(chain),
                    finality_seconds=chain.finality_seconds,
                    genius_compliant=currency.genius_compliant,
                )
            )
    return options
