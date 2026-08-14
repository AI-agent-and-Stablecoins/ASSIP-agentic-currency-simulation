"""Schema and prompt-rendering for the end-of-run switch-elicitation
question (docs/superpowers/specs/2026-08-14-equivalence-framework-design.md
§4): a simpler, parallel path alongside src/llm/decision_schema.py's
negotiation Decision -- a yes/no switch question has no candidates, live
prices, or conversation history to describe, and no wallet/currency/chain
constraint to validate the way a negotiation Decision does.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.llm.agent_reasoning import AgentUtilityContext

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

SWITCH_PROMPT_PATH = PROMPTS_DIR / "switch_question_prompt.txt"


class SwitchDecision(BaseModel):
    will_switch: bool
    reasoning: str


def _format_utility_context(agent: "AgentUtilityContext") -> str:
    parts = [f"Risk profile: {agent.risk_profile}", f"Utility type: {agent.utility_type}"]
    if agent.risk_aversion is not None:
        parts.append(f"Risk aversion (CRRA/CARA-style gamma): {agent.risk_aversion}")
    return "\n".join(parts)


def render_switch_prompt(
    agent_context: "AgentUtilityContext",
    fixed_symbol: str,
    fixed_field: str,
    fixed_value: float,
    varied_symbol: str,
    varied_field: str,
    varied_value: float,
) -> str:
    template = SWITCH_PROMPT_PATH.read_text(encoding="utf-8")
    comparison_block = (
        f"Coin A ({fixed_symbol}): {fixed_field} = {fixed_value}\n"
        f"Coin B ({varied_symbol}): {varied_field} = {varied_value}\n"
        f"Would you switch your holdings from {fixed_symbol} to {varied_symbol} given this?"
    )
    schema_block = '{"will_switch": true or false, "reasoning": "one sentence"}'
    return template.format(
        utility_context_block=_format_utility_context(agent_context),
        comparison_block=comparison_block,
        schema_block=schema_block,
    )
