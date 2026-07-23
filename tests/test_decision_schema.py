from src.llm.decision_schema import Decision, DecisionAction, validate_decision


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


def test_valid_offer_passes():
    result = validate_decision(_decision(), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is True


def test_unsupported_currency_rejected():
    result = validate_decision(_decision(proposed_currency="NOTACOIN"), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is False
    assert "currency" in result.reason.lower()


def test_unsupported_chain_rejected():
    result = validate_decision(_decision(proposed_chain="notachain"), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is False
    assert "chain" in result.reason.lower()


def test_nonpositive_amount_rejected():
    result = validate_decision(_decision(amount=0.0), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is False


def test_nonpositive_price_rejected():
    result = validate_decision(_decision(price=-5.0), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is False


def test_accept_with_insufficient_funds_rejected():
    result = validate_decision(
        _decision(action=DecisionAction.ACCEPT, price=500.0), {"USDC"}, {"ethereum"}, {"USDC": 100.0}
    )
    assert result.is_valid is False
    assert "funds" in result.reason.lower()


def test_reject_and_walk_away_are_always_valid_regardless_of_funds():
    assert validate_decision(_decision(action=DecisionAction.REJECT), {"USDC"}, {"ethereum"}, {}).is_valid is True
    assert validate_decision(_decision(action=DecisionAction.WALK_AWAY), {"USDC"}, {"ethereum"}, {}).is_valid is True
