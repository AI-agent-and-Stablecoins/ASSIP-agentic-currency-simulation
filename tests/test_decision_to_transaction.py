# tests/test_decision_to_transaction.py
from src.blockchain.routing_engine import CurrencyChainOption
from src.llm.decision_adapter import NegotiationAction
from src.llm.decision_schema import DecisionAction
from src.llm.decision_to_transaction import build_transaction_from_negotiation
from src.negotiation.llm_negotiation_engine import run_llm_negotiation
from src.transactions.transaction import TransactionStatus


def _action(action: DecisionAction, price: float = 100.0, currency: str = "USDC", chain: str = "ethereum") -> NegotiationAction:
    return NegotiationAction(
        action=action, price=price, amount=1.0, currency_symbol=currency, chain_name=chain, reasoning="test"
    )


def _candidate(currency: str = "USDC", chain: str = "ethereum", gas_fee: float = 0.42) -> CurrencyChainOption:
    return CurrencyChainOption(
        currency_symbol=currency,
        chain_name=chain,
        governance_score=0.9,
        liquidity_score=0.8,
        peg_error=0.0,
        gas_fee=gas_fee,
        finality_seconds=12.0,
        genius_compliant=True,
    )


def test_accepted_negotiation_with_matching_candidate_builds_transaction():
    def buyer_decide(session):
        return _action(DecisionAction.OFFER, price=95.0)

    def seller_decide(session):
        return _action(DecisionAction.ACCEPT, price=session.current_offer.price)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)
    candidates = [_candidate(gas_fee=0.42), _candidate(currency="USDT", chain="polygon", gas_fee=0.01)]

    tx = build_transaction_from_negotiation(
        session, candidates, buyer_id="buyer-1", seller_id="seller-1", good_name="cloud_compute", day=3
    )

    assert tx is not None
    assert tx.buyer_id == "buyer-1"
    assert tx.seller_id == "seller-1"
    assert tx.good_name == "cloud_compute"
    assert tx.currency_symbol == "USDC"
    assert tx.chain_name == "ethereum"
    assert tx.gas_fee == 0.42
    assert tx.paid_value == 95.0
    assert tx.timestep == 3
    assert tx.status == TransactionStatus.PENDING


def test_accepted_negotiation_with_no_matching_candidate_returns_none():
    """Simulates a hallucinated currency/chain that slipped past adapt_decision's
    looser (full-universe) check but doesn't appear in the exact candidate list
    that was offered to the agent this round."""

    def buyer_decide(session):
        return _action(DecisionAction.OFFER, price=95.0, currency="USDC", chain="ethereum")

    def seller_decide(session):
        return _action(DecisionAction.ACCEPT, price=session.current_offer.price, currency="DGX", chain="solana")

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)
    candidates = [_candidate(currency="USDC", chain="ethereum")]

    tx = build_transaction_from_negotiation(
        session, candidates, buyer_id="buyer-1", seller_id="seller-1", good_name="cloud_compute", day=3
    )

    assert tx is None


def test_rejected_negotiation_returns_none():
    def buyer_decide(session):
        return _action(DecisionAction.OFFER)

    def seller_decide(session):
        return _action(DecisionAction.REJECT)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)
    candidates = [_candidate()]

    tx = build_transaction_from_negotiation(
        session, candidates, buyer_id="buyer-1", seller_id="seller-1", good_name="cloud_compute", day=1
    )

    assert tx is None


def test_walked_away_negotiation_returns_none():
    def buyer_decide(session):
        return _action(DecisionAction.WALK_AWAY)

    def seller_decide(session):
        return _action(DecisionAction.OFFER)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)
    candidates = [_candidate()]

    tx = build_transaction_from_negotiation(
        session, candidates, buyer_id="buyer-1", seller_id="seller-1", good_name="cloud_compute", day=1
    )

    assert tx is None


def test_max_rounds_reached_negotiation_returns_none():
    def buyer_decide(session):
        return _action(DecisionAction.COUNTER_OFFER)

    def seller_decide(session):
        return _action(DecisionAction.COUNTER_OFFER)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=4)
    candidates = [_candidate()]

    tx = build_transaction_from_negotiation(
        session, candidates, buyer_id="buyer-1", seller_id="seller-1", good_name="cloud_compute", day=1
    )

    assert tx is None
