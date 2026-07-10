"""Per-currency preference scores, updated after each transaction outcome."""

from pydantic import BaseModel, Field


class AgentPreferences(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)
    learning_rate: float = 0.1

    def score(self, symbol: str) -> float:
        return self.weights.get(symbol, 0.5)

    def update(self, symbol: str, outcome_value: float) -> None:
        """Exponential moving average toward outcome_value (expected in [0, 1])."""
        current = self.score(symbol)
        self.weights[symbol] = current + self.learning_rate * (outcome_value - current)
