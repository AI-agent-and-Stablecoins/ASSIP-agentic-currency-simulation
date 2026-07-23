# tests/test_llm_negotiation_engine.py
from src.llm.decision_adapter import NegotiationAction
from src.llm.decision_schema import DecisionAction
from src.negotiation.llm_negotiation_engine import NegotiationSession, NegotiationStatus, run_llm_negotiation


def _action(action: DecisionAction, price: float = 100.0) -> NegotiationAction:
    return NegotiationAction(
        action=action, price=price, amount=1.0, currency_symbol="USDC", chain_name="ethereum", reasoning="test"
    )


def test_negotiation_accepts_when_seller_accepts_buyers_offer():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.OFFER, price=95.0)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.ACCEPT, price=session.current_offer.price)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)

    assert session.status == NegotiationStatus.ACCEPTED
    assert session.completed_at is not None
    assert len(session.conversation_history) == 2
    assert session.conversation_history[0].agent_id == "buyer-1"
    assert session.conversation_history[1].agent_id == "seller-1"


def test_negotiation_terminates_on_reject():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.OFFER)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.REJECT)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)

    assert session.status == NegotiationStatus.REJECTED


def test_negotiation_terminates_on_walk_away():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.WALK_AWAY)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.OFFER)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)

    assert session.status == NegotiationStatus.WALKED_AWAY
    assert len(session.conversation_history) == 1


def test_negotiation_hits_max_rounds_cap_and_never_loops_forever():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.COUNTER_OFFER)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.COUNTER_OFFER)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=4)

    assert session.status == NegotiationStatus.MAX_ROUNDS_REACHED
    assert len(session.conversation_history) == 4


def test_offers_form_a_previous_offer_id_chain():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.OFFER)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.ACCEPT, price=session.current_offer.price)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)

    assert session.conversation_history[0].previous_offer_id is None
    assert session.conversation_history[1].previous_offer_id == session.conversation_history[0].offer_id
