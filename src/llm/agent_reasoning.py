"""Assembles the context an LLM needs to make an economically meaningful
decision, and (in Task 15) drives the actual LLM call.

AgentUtilityContext is the agent-side slice only (identity, risk profile,
utility parameters, wallet) -- everything an agent knows about itself.
Environment-level context (currency governance, market intelligence, macro
state, opponent offers) is assembled separately in Task 13's
build_decision_context, which takes plain values rather than Environment/
BaseAgent objects, matching this codebase's existing layering convention
(e.g. src.blockchain.routing_engine.generate_candidates takes plain
balances, not a Wallet).
"""

from pydantic import BaseModel

from src.utility.multi_attribute import MultiAttributeWeights


class AgentUtilityContext(BaseModel):
    agent_id: str
    agent_class: str
    risk_profile: str
    utility_type: str
    risk_aversion: float | None = None
    eis: float | None = None
    multi_attribute_weights: MultiAttributeWeights | None = None
    wallet_balances: dict[str, float] = {}
