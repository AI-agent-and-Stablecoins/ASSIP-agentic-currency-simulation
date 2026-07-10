"""Represents a single transaction between a buyer and a seller.

Field set matches the persistence requirement that every completed
transaction record buyer, seller, timestamp, blockchain, currency, gas fee,
expected value, paid value, and settlement status.
"""

from enum import Enum

from pydantic import BaseModel, Field

from src.utils.helpers import generate_id


class TransactionStatus(str, Enum):
    PENDING = "pending"
    SETTLED = "settled"
    FAILED = "failed"


class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: generate_id("tx"))
    buyer_id: str
    seller_id: str
    good_name: str
    currency_symbol: str
    chain_name: str
    gas_fee: float
    expected_value: float
    paid_value: float
    timestep: int
    status: TransactionStatus = TransactionStatus.PENDING
