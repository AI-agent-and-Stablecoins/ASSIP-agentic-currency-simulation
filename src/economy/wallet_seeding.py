"""Seeds an agent population's wallets into a currency-restricted universe
(a factor-isolation sandbox or a hypothesis-sim's 2-6 real symbols), split
evenly by USD value. Every agent's wallet balance is REPLACED (not merged)
with a share of that agent's pre-existing total USD value, split evenly
across the restricted universe's symbols -- buyers need this to get any
candidates at all from generate_candidates, and every agent needs their
leftover unrestricted-universe balances removed so real_purchasing_power's
ExchangeRateTable lookup (built from only the restricted symbols) never
KeyErrors on a symbol it doesn't know.

Splitting BY USD VALUE (not raw unit count) matters whenever the restricted
universe mixes assets of very different per-unit value (e.g. a gold-backed
token at ~2400 USD/unit against a stablecoin at ~1 USD/unit) -- splitting by
raw unit count would badly skew such a pair's actual USD-value balance.

Shared by src/simulation/matrix_runner.py's 6 factor-isolation sandboxes and
src/economy/hypothesis_scenarios.py's 11 hypothesis-sims -- both need this
same step before generate_candidates can offer any candidates in their
respective restricted universe, and neither should have to import
matrix_runner.py's much heavier httpx/sqlalchemy/database dependencies just
to seed a wallet.
"""

from src.agents.base_agent import BaseAgent
from src.currencies.currency import CurrencyConfig
from src.currencies.exchange_rates import ExchangeRateTable


def seed_restricted_wallets(
    agents: dict[str, BaseAgent],
    restricted_currencies: dict[str, CurrencyConfig],
    real_currencies: dict[str, CurrencyConfig],
    peg_reference_rates: dict[str, float],
) -> None:
    """`real_currencies` and `peg_reference_rates` let this function build an
    ExchangeRateTable for the agent's ORIGINAL (unrestricted-universe)
    balances -- by the time this is called, the caller's Environment.currencies
    has already been replaced with `restricted_currencies`, so the original
    universe must be supplied separately. `peg_reference_rates` (e.g.
    `{"USD": 1.0, "EUR": 1.08, "XAU": 2400.0}`) is shared between both tables:
    every real and restricted-universe currency alike pegs to one of
    USD/EUR/XAU, so the same reference dict prices both sides consistently.
    """
    real_rates = ExchangeRateTable(real_currencies, peg_reference_rates)
    restricted_rates = ExchangeRateTable(restricted_currencies, peg_reference_rates)
    symbols = list(restricted_currencies.keys())
    for agent in agents.values():
        total_usd_value = agent.wallet.total_value_usd(real_rates)
        if total_usd_value <= 0:
            total_usd_value = 1000.0  # safe floor, matching consumer.yaml's USDC scale
        share_usd = total_usd_value / len(symbols)
        agent.wallet.balances = {
            symbol: restricted_rates.convert(share_usd, "USD", symbol) for symbol in symbols
        }
