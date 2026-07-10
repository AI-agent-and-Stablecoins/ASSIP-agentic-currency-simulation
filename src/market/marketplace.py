"""The central exchange: matches buyers and sellers, posts available goods.

Listings reference sellers by agent_id rather than holding agent objects
directly, keeping this module decoupled from src/agents.
"""

from pydantic import BaseModel

from src.market.goods import Good


class Listing(BaseModel):
    seller_id: str
    good: Good
    true_price: float


class Marketplace:
    def __init__(self):
        self._listings: list[Listing] = []

    def post_listing(self, seller_id: str, good: Good, price: float) -> Listing:
        listing = Listing(seller_id=seller_id, good=good, true_price=price)
        self._listings.append(listing)
        return listing

    def list_goods(self) -> list[Listing]:
        return list(self._listings)

    def find_counterparties(self, good_name: str, exclude_agent_id: str | None = None) -> list[Listing]:
        return [
            listing
            for listing in self._listings
            if listing.good.name == good_name and listing.seller_id != exclude_agent_id
        ]

    def clear_listings(self) -> None:
        self._listings.clear()
