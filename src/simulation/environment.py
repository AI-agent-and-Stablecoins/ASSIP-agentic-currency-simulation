"""Creates the world: the composition root every other simulation module reads/writes through.

No other module keeps its own copy of agents/currencies/chains -- everything
routes through one Environment instance, per the "no global state" standard.
"""

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.agents.base_agent import BaseAgent
from src.blockchain.chain import ChainConfig, load_chain_universe
from src.blockchain.liquidity_pools import LiquidityPoolRegistry
from src.currencies.currency import CurrencyConfig, load_currency_universe
from src.currencies.exchange_rates import ExchangeRateTable
from src.economy.shocks import ScenarioConfig, load_scenario
from src.market.goods import Good
from src.market.marketplace import Marketplace
from src.simulation.event_queue import EventQueue
from src.transactions.ledger import Ledger

DEFAULT_GOODS: list[Good] = [
    Good(name="cloud_compute", category="cloud_compute", base_price_usd=100.0),
    Good(name="electricity", category="electricity", base_price_usd=50.0),
    Good(name="data", category="data", base_price_usd=20.0),
    Good(name="ai_services", category="ai_services", base_price_usd=200.0),
]


class Environment:
    def __init__(
        self,
        currencies: dict[str, CurrencyConfig],
        chains: dict[str, ChainConfig],
        scenario: ScenarioConfig,
        agents: list[BaseAgent],
        goods: list[Good] | None = None,
    ):
        self.currencies = currencies
        self.chains = chains
        self.scenario = scenario
        self.macro_state = scenario.initial_state.model_copy(deep=True)
        self.agents: dict[str, BaseAgent] = {agent.agent_id: agent for agent in agents}
        self.goods = goods if goods is not None else list(DEFAULT_GOODS)
        self.liquidity_pools = LiquidityPoolRegistry()
        self.marketplace = Marketplace()
        self.ledger = Ledger()
        self.event_queue = EventQueue(scenario.shocks)
        self.exchange_rates = ExchangeRateTable(currencies, self.macro_state.peg_reference_rates)

    def refresh_exchange_rates(self) -> None:
        """Call after macro_state changes (e.g. a shock) to rebuild derived rate lookups."""
        self.exchange_rates = ExchangeRateTable(self.currencies, self.macro_state.peg_reference_rates)

    @classmethod
    def build(cls, scenario_name: str, agent_mix: dict[str, int]) -> "Environment":
        currencies = load_currency_universe()
        chains = load_chain_universe()
        scenario = load_scenario(scenario_name)

        profiles = load_agent_profiles()
        agents: list[BaseAgent] = []
        for profile_name, count in agent_mix.items():
            profile = profiles[profile_name]
            agents.extend(build_agent(profile) for _ in range(count))

        return cls(currencies=currencies, chains=chains, scenario=scenario, agents=agents)
