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
from src.economy.trust import TrustLedger


class CurrencyChainOption(BaseModel):
    currency_symbol: str
    chain_name: str
    governance_score: float
    liquidity_score: float
    peg_error: float
    gas_fee: float
    finality_seconds: float
    genius_compliant: bool
    bid_ask_spread: float | None = None


def generate_candidates(
    available_balances: dict[str, float],
    currencies: dict[str, CurrencyConfig],
    chains: dict[str, ChainConfig],
    liquidity_pools: LiquidityPoolRegistry | None = None,
    trust_ledger: TrustLedger | None = None,
    currency_chain_pins: dict[str, str] | None = None,
) -> list[CurrencyChainOption]:
    """One candidate per (currency the agent holds a positive balance of) x (chain),
    unless currency_chain_pins restricts a currency to exactly one chain."""
    liquidity_pools = liquidity_pools or LiquidityPoolRegistry()
    currency_chain_pins = currency_chain_pins or {}
    options: list[CurrencyChainOption] = []
    for symbol, balance in available_balances.items():
        if balance <= 0 or symbol not in currencies:
            continue
        currency = currencies[symbol]
        if trust_ledger is not None:
            peg_error = trust_ledger.effective_peg_error(symbol, currency.peg_error)
        else:
            peg_error = currency.peg_error
        pinned_chain = currency_chain_pins.get(symbol)
        candidate_chains = [chains[pinned_chain]] if pinned_chain is not None else list(chains.values())
        for chain in candidate_chains:
            pool_liquidity_score = liquidity_pools.get_liquidity(currency, chain.name)
            if trust_ledger is not None:
                pool_liquidity_score = trust_ledger.effective_liquidity_score(symbol, pool_liquidity_score)
            options.append(
                CurrencyChainOption(
                    currency_symbol=symbol,
                    chain_name=chain.name,
                    governance_score=currency.governance_score,
                    liquidity_score=pool_liquidity_score,
                    peg_error=peg_error,
                    gas_fee=get_gas_fee(chain),
                    finality_seconds=chain.finality_seconds,
                    genius_compliant=currency.genius_compliant,
                    bid_ask_spread=currency.bid_ask_spread,
                )
            )
    return options
