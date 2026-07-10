"""Runs rule-based (no-LLM) negotiation conversations.

Buyer opens low, seller opens high; both concede a fraction of the gap
toward each other every round. Termination is guaranteed by a hard
max_rounds cap combined with an early-accept check once the gap closes to
within agreement_tolerance -- conversations can never loop indefinitely.
"""

from src.negotiation.conversation_history import ConversationLog
from src.negotiation.counter_offer import generate_counter
from src.negotiation.offer import Offer


def negotiate(
    buyer_opening_price: float,
    seller_opening_price: float,
    currency_symbol: str,
    chain_name: str,
    true_price: float,
    supported_currencies: set[str],
    max_rounds: int = 10,
    agreement_tolerance: float = 0.01,
    concession_rate: float = 0.3,
) -> tuple[float | None, ConversationLog]:
    log = ConversationLog()
    opening = Offer(price=buyer_opening_price, currency_symbol=currency_symbol, chain_name=chain_name, round=0)

    if not opening.is_valid(supported_currencies) or seller_opening_price <= 0:
        log.finalize("rejected")
        return None, log

    log.add(opening)
    buyer_offer = opening
    seller_price = seller_opening_price

    for round_number in range(1, max_rounds + 1):
        if seller_price - buyer_offer.price <= agreement_tolerance * true_price:
            agreed_price = (buyer_offer.price + seller_price) / 2
            log.finalize("accepted")
            return agreed_price, log

        buyer_offer = generate_counter(buyer_offer, seller_price, concession_rate)
        log.add(buyer_offer)

        seller_offer = generate_counter(
            Offer(price=seller_price, currency_symbol=currency_symbol, chain_name=chain_name, round=round_number),
            buyer_offer.price,
            concession_rate,
        )
        seller_price = seller_offer.price
        log.add(seller_offer)

    log.finalize("timeout")
    return None, log
