import pytest

from src.llm.decision_adapter import DecisionValidationError, NegotiationAction, adapt_decision
from src.llm.decision_schema import Decision, DecisionAction


def _decision(**overrides) -> Decision:
    defaults = dict(
        action=DecisionAction.OFFER,
        proposed_currency="USDC",
        proposed_chain="ethereum",
        amount=1.0,
        price=100.0,
        reasoning="test reasoning",
    )
    defaults.update(overrides)
    return Decision(**defaults)


def test_adapt_valid_decision_produces_negotiation_action():
    action = adapt_decision(_decision(), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})

    assert isinstance(action, NegotiationAction)
    assert action.currency_symbol == "USDC"
    assert action.chain_name == "ethereum"
    assert action.price == 100.0


def test_adapt_invalid_currency_raises_with_reason():
    with pytest.raises(DecisionValidationError) as exc_info:
        adapt_decision(_decision(proposed_currency="NOTACOIN"), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})

    assert "currency" in str(exc_info.value).lower()


def test_adapt_accept_with_insufficient_funds_raises():
    with pytest.raises(DecisionValidationError):
        adapt_decision(
            _decision(action=DecisionAction.ACCEPT, price=5000.0), {"USDC"}, {"ethereum"}, {"USDC": 100.0}
        )
