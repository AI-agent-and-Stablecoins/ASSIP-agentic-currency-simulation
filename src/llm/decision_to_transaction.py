"""Turns a completed NegotiationSession into a settleable Transaction.

adapt_decision (decision_adapter.py) validates a single Decision against the
full supported currency/chain universe -- a per-turn, in-negotiation check.
This module runs after negotiation has finished: it takes the session's final
ACCEPTed offer and validates the accepted (currency_symbol, chain_name) pair
against the *exact* candidate list that was offered this round (from
generate_candidates), a stricter anti-hallucination check than adapt_decision
provides, since a model could in principle accept a currency/chain that is
supported in general but was never actually presented as a candidate for this
trade.

Returns None (never raises) for any non-deal outcome: the negotiation didn't
end in ACCEPT, or it did but the accepted currency/chain isn't among the
offered candidates. This mirrors the existing rule-based path in
src/simulation/timestep.py, where `if agreed_price is None: continue` is
already the idiom for "no deal, move on" -- keeping this function's contract
symmetric with that lets a future caller use the same `if tx is None:
continue` pattern regardless of which negotiation engine produced the
session.
"""

from src.blockchain.routing_engine import CurrencyChainOption
from src.negotiation.llm_negotiation_engine import NegotiationSession, NegotiationStatus
from src.transactions.transaction import Transaction


def build_transaction_from_negotiation(
    session: NegotiationSession,
    candidates: list[CurrencyChainOption],
    buyer_id: str,
    seller_id: str,
    good_name: str,
    day: int,
) -> Transaction | None:
    if session.status != NegotiationStatus.ACCEPTED or session.current_offer is None:
        return None

    accepted_offer = session.current_offer
    matching_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.currency_symbol == accepted_offer.currency_symbol
            and candidate.chain_name == accepted_offer.chain_name
        ),
        None,
    )
    if matching_candidate is None:
        return None

    expected_value = session.initial_offer.price if session.initial_offer is not None else accepted_offer.price

    return Transaction(
        buyer_id=buyer_id,
        seller_id=seller_id,
        good_name=good_name,
        currency_symbol=accepted_offer.currency_symbol,
        chain_name=accepted_offer.chain_name,
        gas_fee=matching_candidate.gas_fee,
        expected_value=expected_value,
        paid_value=accepted_offer.price,
        timestep=day,
    )
