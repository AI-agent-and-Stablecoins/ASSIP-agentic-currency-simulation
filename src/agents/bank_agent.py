"""Bank agent: holds a reserve composition in addition to standard agent state."""

from src.agents.base_agent import BaseAgent
from src.governance.reserve_models import ReserveComposition


class BankAgent(BaseAgent):
    reserves: ReserveComposition | None = None
