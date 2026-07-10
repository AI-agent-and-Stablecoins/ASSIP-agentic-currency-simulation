"""Decides which LLM model handles which agent's requests.

Not implemented in Phase 1 -- rule-based agents make no LLM calls (see
src/utility/ for the decision logic they actually use). This is the Phase 2
extension point.
"""


def route_model(agent_class: str) -> str:
    raise NotImplementedError("LLM routing is a Phase 2 feature; Phase 1 agents are rule-based")
