import pytest

from src.blockchain.chain import load_chain_universe
from src.blockchain.routing_engine import generate_candidates
from src.currencies.currency import load_currency_universe
from src.economy.shocks import ShockEvent, ShockType
from src.economy.trust import TrustLedger, TrustParams


def _params() -> TrustParams:
    return TrustParams(lambda_shock=0.5, lambda_recover=0.03, lambda_contagion=0.1, rolling_window_days=30)


def test_generate_candidates_uses_static_values_without_a_trust_ledger():
    currencies = load_currency_universe()
    chains = load_chain_universe()

    candidates = generate_candidates({"USDT": 100.0}, currencies, chains)

    assert candidates[0].peg_error == currencies["USDT"].peg_error


def test_generate_candidates_reflects_trust_ledger_effective_peg_error():
    currencies = load_currency_universe()
    chains = load_chain_universe()
    ledger = TrustLedger(currencies, _params())
    ledger.update([ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")])

    candidates = generate_candidates({"USDT": 100.0}, currencies, chains, trust_ledger=ledger)

    assert candidates[0].peg_error == pytest.approx(currencies["USDT"].peg_error + 0.08)
    assert candidates[0].peg_error != currencies["USDT"].peg_error


def test_generate_candidates_reflects_trust_ledger_effective_liquidity_score():
    currencies = load_currency_universe()
    chains = load_chain_universe()
    ledger = TrustLedger(currencies, _params())
    ledger.update([ShockEvent(day=0, type=ShockType.LIQUIDITY_CRUNCH, magnitude=0.3, target_currency="USDC")])

    candidates = generate_candidates({"USDC": 100.0}, currencies, chains, trust_ledger=ledger)

    assert candidates[0].liquidity_score < currencies["USDC"].liquidity_score
