"""Builds concrete agent instances from configs/agent_profiles/*.yaml.

Not part of the original module list, but needed because profile files are
personality parameterizations (consumer, merchant, bank, institution,
investor) that don't map 1:1 onto the five agent classes -- an explicit
agent_class field in each profile picks the class to instantiate.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.agents.bank_agent import BankAgent
from src.agents.base_agent import BaseAgent
from src.agents.buyer_agent import BuyerAgent
from src.agents.investor_agent import InvestorAgent
from src.agents.regulator_agent import RegulatorAgent
from src.agents.seller_agent import SellerAgent
from src.agents.wallet import Wallet
from src.utility.multi_attribute import MultiAttributeWeights
from src.utility.utility_factory import build_utility_function
from src.utils.config_loader import load_yaml_dir_as
from src.utils.constants import CONFIG_ROOT
from src.utils.helpers import generate_id

AgentClass = Literal["buyer", "seller", "bank", "investor", "regulator"]

_AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "buyer": BuyerAgent,
    "seller": SellerAgent,
    "bank": BankAgent,
    "investor": InvestorAgent,
    "regulator": RegulatorAgent,
}


class AgentProfileConfig(BaseModel):
    name: str
    agent_class: AgentClass
    risk_tolerance: Literal["low", "medium", "high"]
    utility_type: Literal["crra", "cara", "multi_attribute"]
    risk_aversion: float | None = None
    weights: MultiAttributeWeights | None = None
    initial_wallet: dict[str, float] = {}


def load_agent_profiles(config_dir: Path = CONFIG_ROOT / "agent_profiles") -> dict[str, AgentProfileConfig]:
    return load_yaml_dir_as(config_dir, AgentProfileConfig)


def build_agent(profile: AgentProfileConfig) -> BaseAgent:
    agent_cls = _AGENT_CLASSES[profile.agent_class]
    utility_fn = build_utility_function(profile.utility_type, profile.risk_aversion, profile.weights)
    wallet = Wallet(balances=dict(profile.initial_wallet))
    return agent_cls(
        agent_id=generate_id(profile.agent_class),
        agent_class=profile.agent_class,
        profile_name=profile.name,
        risk_profile=profile.risk_tolerance,
        wallet=wallet,
        utility_fn=utility_fn,
    )
