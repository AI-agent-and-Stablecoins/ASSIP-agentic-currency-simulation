"""Named agent risk personalities, referenced by configs/agent_profiles/*.yaml."""

from enum import Enum


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    INSTITUTIONAL = "institutional"
    CROSS_BORDER_TRADER = "cross_border_trader"
