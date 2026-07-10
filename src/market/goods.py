"""Products traded in the simulation (cloud compute, electricity, data, AI services)."""

from pydantic import BaseModel, Field


class Good(BaseModel):
    name: str
    category: str
    base_price_usd: float = Field(gt=0.0)
