from src.llm.decision_schema import DecisionAction
from src.negotiation.llm_offer import LLMOffer


def _offer(**overrides) -> LLMOffer:
    defaults = dict(
        negotiation_id="neg-1",
        agent_id="buyer-1",
        action=DecisionAction.OFFER,
        price=100.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        reasoning="test",
        round=0,
    )
    defaults.update(overrides)
    return LLMOffer(**defaults)


def test_llm_offer_generates_unique_id_and_has_no_previous_by_default():
    offer = _offer()

    assert offer.offer_id.startswith("offer-")
    assert offer.previous_offer_id is None
    assert offer.timestamp is not None


def test_counter_offer_references_previous_offer_id():
    first = _offer(price=90.0)
    counter = _offer(
        previous_offer_id=first.offer_id, agent_id="seller-1", action=DecisionAction.COUNTER_OFFER, price=95.0, round=1
    )

    assert counter.previous_offer_id == first.offer_id
    assert counter.offer_id != first.offer_id


def test_two_offers_never_share_an_id():
    assert _offer().offer_id != _offer().offer_id
