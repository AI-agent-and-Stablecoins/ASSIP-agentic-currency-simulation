"""Calls an LLM to generate agent decisions.

Not implemented in Phase 1 -- see src/utility/ for the rule-based decision
logic Phase 1 agents actually use. This is the Phase 2 extension point where
LLM-generated reasoning replaces/augments utility-function math. Per the
project's coding standards, this module must never directly alter balances
even once implemented -- it only produces reasoning/decisions that flow
through src/transactions/settlement.py like any other agent decision.
"""


def generate_reasoning(prompt: str) -> str:
    raise NotImplementedError("LLM reasoning is a Phase 2 feature; Phase 1 agents are rule-based")
