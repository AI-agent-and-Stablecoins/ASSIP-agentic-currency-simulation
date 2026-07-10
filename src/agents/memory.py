"""Rolling per-currency transaction success/failure counts."""

from pydantic import BaseModel, Field


class AgentMemory(BaseModel):
    outcomes: dict[str, dict[str, int]] = Field(default_factory=dict)

    def record(self, symbol: str, success: bool) -> None:
        bucket = self.outcomes.setdefault(symbol, {"success": 0, "fail": 0})
        bucket["success" if success else "fail"] += 1

    def success_rate(self, symbol: str) -> float:
        bucket = self.outcomes.get(symbol)
        if not bucket:
            return 1.0
        total = bucket["success"] + bucket["fail"]
        return bucket["success"] / total if total else 1.0
