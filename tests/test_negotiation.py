from src.negotiation.negotiation_engine import negotiate

SUPPORTED = {"USDC"}


def test_negotiation_terminates_within_max_rounds_even_with_wide_gap():
    max_rounds = 5
    _, log = negotiate(
        buyer_opening_price=10.0,
        seller_opening_price=1000.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        true_price=500.0,
        supported_currencies=SUPPORTED,
        max_rounds=max_rounds,
        agreement_tolerance=0.001,
        concession_rate=0.1,
    )

    assert len(log.offers) <= 2 * max_rounds + 1
    assert log.outcome in {"accepted", "timeout"}


def test_negotiation_rejects_unsupported_currency():
    agreed_price, log = negotiate(
        buyer_opening_price=100.0,
        seller_opening_price=120.0,
        currency_symbol="NOTACOIN",
        chain_name="ethereum",
        true_price=100.0,
        supported_currencies=SUPPORTED,
    )

    assert agreed_price is None
    assert log.outcome == "rejected"


def test_negotiation_rejects_negative_opening_price():
    agreed_price, log = negotiate(
        buyer_opening_price=-10.0,
        seller_opening_price=120.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        true_price=100.0,
        supported_currencies=SUPPORTED,
    )

    assert agreed_price is None
    assert log.outcome == "rejected"


def test_negotiation_reaches_agreement_when_openings_are_close():
    agreed_price, log = negotiate(
        buyer_opening_price=99.0,
        seller_opening_price=100.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        true_price=100.0,
        supported_currencies=SUPPORTED,
        agreement_tolerance=0.05,
    )

    assert agreed_price is not None
    assert log.outcome == "accepted"
