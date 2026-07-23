import json

import httpx

from src.agents.wallet import Wallet
from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.agent_reasoning import (
    AgentUtilityContext,
    TransactionContext,
    build_decision_context,
    decide,
)
from src.llm.llm_router import OPENROUTER_BASE_URL, RetryConfig, load_model_roster
from src.transactions.settlement import settle
from src.transactions.transaction import Transaction, TransactionStatus


def _decision_json() -> str:
    return json.dumps(
        {
            "action": "ACCEPT",
            "proposed_currency": "USDC",
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 100.0,
            "reasoning": "accepting the offer",
        }
    )


def test_llm_decision_never_mutates_wallet_before_settlement():
    buyer_wallet = Wallet(balances={"USDC": 1000.0})
    seller_wallet = Wallet(balances={"USDC": 0.0})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": _decision_json()}}]})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()

    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances=dict(buyer_wallet.balances),
    )
    candidates = [
        CurrencyChainOption(
            currency_symbol="USDC",
            chain_name="ethereum",
            governance_score=0.95,
            liquidity_score=0.97,
            peg_error=0.0003,
            gas_fee=2.5,
            finality_seconds=12.0,
            genius_compliant=True,
        )
    ]
    macro = MacroState()
    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, TransactionContext(is_cross_border=False)
    )

    outcome = decide(
        "buyer", context, roster, client, {"USDC"}, {"ethereum"}, retry_config=RetryConfig(sleep_fn=lambda s: None)
    )

    # The LLM call and decision-adaptation must never touch the wallet.
    assert buyer_wallet.balances["USDC"] == 1000.0
    assert seller_wallet.balances["USDC"] == 0.0

    # Only the existing deterministic settlement path may move money.
    tx = Transaction(
        buyer_id="buyer-1",
        seller_id="seller-1",
        good_name="cloud_compute",
        currency_symbol=outcome.negotiation_action.currency_symbol,
        chain_name=outcome.negotiation_action.chain_name,
        gas_fee=2.5,
        expected_value=100.0,
        paid_value=outcome.negotiation_action.price,
        timestep=0,
    )
    settle(tx, buyer_wallet, seller_wallet)

    assert tx.status == TransactionStatus.SETTLED
    assert buyer_wallet.balances["USDC"] == 900.0
    assert seller_wallet.balances["USDC"] == 100.0
