"""Generates the next offer, conceding a fraction of the way toward a target price."""

from src.negotiation.offer import Offer


class CounterOffer(Offer):
    pass


def generate_counter(prev_offer: Offer, target_price: float, concession_rate: float) -> CounterOffer:
    new_price = prev_offer.price + concession_rate * (target_price - prev_offer.price)
    return CounterOffer(
        price=new_price,
        currency_symbol=prev_offer.currency_symbol,
        chain_name=prev_offer.chain_name,
        round=prev_offer.round + 1,
    )
