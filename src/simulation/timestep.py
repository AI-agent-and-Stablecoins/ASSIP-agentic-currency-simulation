"""Runs one simulation day: the 13-step rule-based lifecycle (steps 1-13 of
project_instructions.md section 6; LLM reasoning at step 6 is skipped since
Phase 1 has no LLM calls; step 14 -- persistence -- is the caller's job so
this stays testable without a database).
"""

import random

from pydantic import BaseModel, Field

from src.agents.buyer_agent import BuyerAgent
from src.agents.seller_agent import SellerAgent
from src.blockchain.routing_engine import generate_candidates
from src.economy.shocks import apply_shock
from src.market.pricing_engine import true_price
from src.negotiation.conversation_history import ConversationLog
from src.negotiation.negotiation_engine import negotiate
from src.simulation.environment import Environment
from src.simulation.scheduler import agent_activation_order
from src.transactions.settlement import settle
from src.transactions.transaction import Transaction, TransactionStatus
from src.transactions.validation import validate_transaction


class TimestepResult(BaseModel):
    day: int
    transactions: list[Transaction] = Field(default_factory=list)
    negotiations: list[ConversationLog] = Field(default_factory=list)


def run_timestep(
    env: Environment,
    day: int,
    rng: random.Random,
    max_negotiation_rounds: int = 10,
    agreement_tolerance: float = 0.01,
    concession_rate: float = 0.3,
) -> TimestepResult:
    # Steps 1-2: update macroeconomic state and prices from any shocks due today.
    for shock in env.event_queue.pop_due(day):
        env.macro_state = apply_shock(env.macro_state, shock)
    env.refresh_exchange_rates()

    result = TimestepResult(day=day)
    env.marketplace.clear_listings()

    sellers = [a for a in env.agents.values() if isinstance(a, SellerAgent)]
    buyers = {a.agent_id: a for a in env.agents.values() if isinstance(a, BuyerAgent)}

    for seller in sellers:
        for good in env.goods:
            price = true_price(good)
            asking = seller.asking_price(price)
            env.marketplace.post_listing(seller.agent_id, good, asking)

    # Step 3: select active agents (buyers act each day; sellers already listed above).
    active_buyers = agent_activation_order(buyers, day, rng)

    for buyer in active_buyers:
        for good in env.goods:
            # Step 4: agent observes the environment (available listings).
            listings = env.marketplace.find_counterparties(good.name, exclude_agent_id=buyer.agent_id)
            if not listings:
                continue
            listing = listings[0]
            seller = env.agents[listing.seller_id]

            # Steps 5, 7-8: compute utility, choose currency and blockchain.
            candidates = generate_candidates(buyer.wallet.balances, env.currencies, env.chains, env.liquidity_pools)
            if not candidates:
                continue
            chosen = buyer.choose_currency_and_chain(candidates)

            # Step 9: negotiate.
            buyer_open = buyer.opening_offer_price(listing.true_price)
            seller_open = seller.asking_price(listing.true_price)
            agreed_price, log = negotiate(
                buyer_opening_price=buyer_open,
                seller_opening_price=seller_open,
                currency_symbol=chosen.currency_symbol,
                chain_name=chosen.chain_name,
                true_price=listing.true_price,
                supported_currencies=set(env.currencies.keys()),
                max_rounds=max_negotiation_rounds,
                agreement_tolerance=agreement_tolerance,
                concession_rate=concession_rate,
            )
            result.negotiations.append(log)
            if agreed_price is None:
                continue

            tx = Transaction(
                buyer_id=buyer.agent_id,
                seller_id=seller.agent_id,
                good_name=good.name,
                currency_symbol=chosen.currency_symbol,
                chain_name=chosen.chain_name,
                gas_fee=chosen.gas_fee,
                expected_value=listing.true_price,
                paid_value=agreed_price,
                timestep=day,
            )

            # Step 10: validate.
            validation = validate_transaction(tx, buyer.wallet, env.currencies)
            if not validation.is_valid:
                tx.status = TransactionStatus.FAILED
                result.transactions.append(tx)
                continue

            # Step 11: settle payment.
            settle(tx, buyer.wallet, seller.wallet)
            env.ledger.record(tx)
            result.transactions.append(tx)

            # Step 12: update memory/preferences.
            success = tx.status == TransactionStatus.SETTLED
            buyer.update_memory(chosen.currency_symbol, success)
            seller.update_memory(chosen.currency_symbol, success)

    # Step 13 (recording metrics) reads env.ledger/result after the fact --
    # see metrics/*.py, which query the ledger rather than being called inline here.
    return result
