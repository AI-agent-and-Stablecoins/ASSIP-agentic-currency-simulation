"""Reserve backing composition for a currency issuer."""

from pydantic import BaseModel, model_validator


class ReserveComposition(BaseModel):
    treasuries: float = 0.0
    cash: float = 0.0
    commercial_paper: float = 0.0
    gold: float = 0.0
    bank_deposits: float = 0.0

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "ReserveComposition":
        total = self.treasuries + self.cash + self.commercial_paper + self.gold + self.bank_deposits
        if total > 0 and abs(total - 1.0) > 1e-6:
            raise ValueError(f"Reserve composition fractions must sum to 1.0, got {total}")
        return self
