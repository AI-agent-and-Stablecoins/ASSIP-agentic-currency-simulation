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
from src.economy.event_log import EventLog
from src.economy.shocks import ScenarioConfig, load_scenario
from src.economy.trust import TrustLedger, load_trust_params
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
        self.trust_ledger = TrustLedger(currencies, load_trust_params())
        self.event_log = EventLog()
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
        self.price_index: float = 1.0
        # Task 11 (Phase 3 Plan 4): day-over-day real purchasing power per
        # agent, keyed by agent_id -- the state persist_full_timestep needs
        # to drive Task 7's adapt_cara_coefficient (which requires a
        # w_real_before/w_real_after pair). Empty until an agent's first
        # persist_full_timestep call, which seeds it without adapting
        # (there is no genuine "before" to compare against on an agent's
        # first day).
        self.previous_real_purchasing_power: dict[str, float] = {}
        self.currency_chain_pins: dict[str, str] = {}

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

    @classmethod
    def build_from_population(
        cls,
        scenario_name: str,
        agents: list[BaseAgent],
        currencies: dict[str, CurrencyConfig] | None = None,
        goods: list[Good] | None = None,
        scenario: ScenarioConfig | None = None,
    ) -> "Environment":
        """Build an Environment from an already-constructed agent population.

        Alongside `Environment.build` (unchanged), for callers that build the
        agent population themselves (e.g. `generate_agent_population`) instead
        of an `agent_mix` count dict. `currencies=None` uses the full real
        9-currency universe; a caller-supplied dict (e.g. one of
        `SANDBOX_CURRENCY_PAIRS`) is used as-is -- the hook the 6
        factor-isolation sandboxes use.

        `scenario`, if given, is used AS-IS instead of loading `scenario_name`
        from YAML -- the hook the matrix runner's 12 sandbox cells use to pass
        a `build_sandbox_scenario`-constructed `ScenarioConfig` (whose shocks
        actually target that sandbox's own synthetic currency symbols, unlike
        `master_simulation.yaml`'s real-universe-only currency-targeted
        shocks) while `scenario_name` still identifies which base scenario it
        was derived from for logging/provenance purposes. `scenario_name` is
        still loaded from YAML when `scenario` is omitted, matching this
        method's behavior before this parameter existed.
        """
        resolved_currencies = currencies if currencies is not None else load_currency_universe()
        chains = load_chain_universe()
        resolved_scenario = scenario if scenario is not None else load_scenario(scenario_name)
        return cls(
            currencies=resolved_currencies, chains=chains, scenario=resolved_scenario, agents=agents, goods=goods
        )
