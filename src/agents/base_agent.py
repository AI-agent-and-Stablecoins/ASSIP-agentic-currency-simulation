"""Parent class every agent inherits from."""

from pydantic import BaseModel, ConfigDict, Field

from src.agents.memory import AgentMemory
from src.agents.preferences import AgentPreferences
from src.agents.wallet import Wallet
from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction, choose_best
from src.utility.multi_attribute import MultiAttributeWeights


class BaseAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    agent_class: str
    profile_name: str
    risk_profile: str
    wallet: Wallet
    utility_fn: UtilityFunction
    utility_type: str
    risk_aversion: float | None = None
    eis: float | None = None
    multi_attribute_weights: MultiAttributeWeights | None = None
    memory: AgentMemory = Field(default_factory=AgentMemory)
    preferences: AgentPreferences = Field(default_factory=AgentPreferences)

    def choose_currency_and_chain(self, candidates: list[CurrencyChainOption]) -> CurrencyChainOption:
        wealth = sum(self.wallet.balances.values())
        return choose_best(candidates, self.utility_fn, wealth=wealth)

    def update_memory(self, symbol: str, success: bool) -> None:
        self.memory.record(symbol, success)
        self.preferences.update(symbol, 1.0 if success else 0.0)

    def build_llm_context(self) -> "AgentUtilityContext":
        from src.llm.agent_reasoning import AgentUtilityContext

        return AgentUtilityContext(
            agent_id=self.agent_id,
            agent_class=self.agent_class,
            risk_profile=self.risk_profile,
            utility_type=self.utility_type,
            risk_aversion=self.risk_aversion,
            eis=self.eis,
            multi_attribute_weights=self.multi_attribute_weights,
            wallet_balances=dict(self.wallet.balances),
        )
