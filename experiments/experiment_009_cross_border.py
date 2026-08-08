"""Experiment 009: Cross-border (US buyer x European merchant).

Research question: when a USD-holding buyer trades with a EUR-holding
merchant, how much Euro-stablecoin adoption happens, and how does a buyer's
home_chain (where its funds are natively deployed -- src/agents/base_agent.py)
trade off against bridging cost/convenience when routing payments?

This exercises src/blockchain/bridges.py's bridge cost/time model and
MultiAttributeWeights.cross_border_convenience_weight end to end:
    - A buyer with home_chain="ethereum" (expensive native gas, $2.50) is
      compared against one with home_chain="arbitrum" (cheap native gas,
      $0.15), both weighting cross_border_convenience at the same level.
    - The expensive-home-chain buyer is expected to bridge away to a cheaper
      chain despite paying a convenience penalty for doing so (the fee gap
      dominates); the cheap-home-chain buyer is expected to just stay home.

Custom agent profiles (US buyer holding USD-pegged currencies with a
home_chain, EU merchant holding EUR-pegged currencies) are built in-memory
via experiments._common.build_custom_environment rather than added to
configs/agent_profiles/*.yaml, since neither profile exists there.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (  # noqa: E402
    AgentProfileConfig,
    MultiAttributeWeights,
    build_custom_environment,
    make_run_dir,
    print_header,
    print_kv,
    run_environment,
    save_run_artifacts,
    standard_metrics,
)
from metrics.chain_selection import chain_usage_share  # noqa: E402
from src.simulation.simulation_runner import SimulationConfig  # noqa: E402

EXPERIMENT_NAME = "experiment_009_cross_border"
EUR_PEGGED_SYMBOLS = {"EURC", "EURT"}

_CROSS_BORDER_WEIGHTS = MultiAttributeWeights(
    governance_weight=0.20,
    liquidity_weight=0.20,
    gas_fee_weight=0.20,
    volatility_weight=0.10,
    compliance_weight=0.10,
    cross_border_convenience_weight=0.20,
)


def _us_buyer_profile(home_chain: str) -> AgentProfileConfig:
    return AgentProfileConfig(
        name=f"us_buyer_{home_chain}",
        agent_class="buyer",
        risk_tolerance="medium",
        utility_type="multi_attribute",
        weights=_CROSS_BORDER_WEIGHTS,
        initial_wallet={"USDC": 1500.0, "USDT": 500.0},
        home_chain=home_chain,
        daily_income={"USDC": 400.0},
    )


def _eu_merchant_profile() -> AgentProfileConfig:
    return AgentProfileConfig(
        name="eu_merchant",
        agent_class="seller",
        risk_tolerance="medium",
        utility_type="multi_attribute",
        weights=MultiAttributeWeights(),
        initial_wallet={"EURC": 1000.0, "EURT": 500.0},
    )


def _switching_rate(transactions) -> float:
    """Fraction of consecutive same-buyer transactions where the currency changed."""
    by_buyer: dict[str, list] = defaultdict(list)
    for tx in sorted(transactions, key=lambda t: t.timestep):
        by_buyer[tx.buyer_id].append(tx.currency_symbol)

    switches, total_pairs = 0, 0
    for symbols in by_buyer.values():
        for prev, curr in zip(symbols, symbols[1:]):
            total_pairs += 1
            if prev != curr:
                switches += 1
    return switches / total_pairs if total_pairs else 0.0


def _run(home_chain: str, num_buyers: int, num_merchants: int, num_days: int, seed: int):
    env = build_custom_environment(
        "baseline",
        {
            "buyer": (_us_buyer_profile(home_chain), num_buyers),
            "seller": (_eu_merchant_profile(), num_merchants),
        },
    )
    timesteps = run_environment(env, num_days=num_days, random_seed=seed)
    return env, timesteps


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--num-buyers", type=int, default=10)
    parser.add_argument("--num-merchants", type=int, default=5)
    parser.add_argument("--num-days", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print_header("Experiment 009: Cross-border (US buyer x European merchant)")

    for home_chain in ("ethereum", "arbitrum"):
        env, timesteps = _run(home_chain, args.num_buyers, args.num_merchants, args.num_days, args.seed)
        metrics = standard_metrics(env, timesteps)
        eur_share = sum(v for symbol, v in metrics["market_share"].items() if symbol in EUR_PEGGED_SYMBOLS)
        switching_rate = _switching_rate(env.ledger.history())
        chain_usage = chain_usage_share(env.ledger)

        print_kv(f"condition: home_chain={home_chain}", "")
        print_kv("  market_share", metrics["market_share"])
        print_kv("  chain_usage_share", chain_usage)
        print_kv("  stayed_home_share", chain_usage.get(home_chain, 0.0))
        print_kv("  eur_pegged_adoption_share", eur_share)
        print_kv("  currency_switching_rate", switching_rate)
        print_kv("  transaction_success_rate", metrics["transaction_success_rate"])

        record_config = SimulationConfig(
            agent_mix={f"us_buyer_{home_chain}": args.num_buyers, "eu_merchant": args.num_merchants},
            num_days=args.num_days,
            scenario="baseline",
            random_seed=args.seed,
        )
        paths = make_run_dir(f"{EXPERIMENT_NAME}__home_{home_chain}")
        save_run_artifacts(
            paths,
            record_config,
            env,
            timesteps,
            metrics,
            extra_metadata={
                "research_question": "Does a buyer bridge away from an expensive home chain despite a cross-border convenience penalty, and how much EUR-pegged currency gets adopted?",
                "home_chain": home_chain,
                "chain_usage_share": chain_usage,
                "eur_pegged_adoption_share": eur_share,
                "currency_switching_rate": switching_rate,
            },
        )
        print(f"  saved to {paths['base']}\n")

    print(
        "note: only the buyer's own home_chain/bridging cost is modeled -- currency choice "
        "still depends only on what the BUYER holds (routing_engine.generate_candidates() "
        "reads buyer.wallet.balances), not on what the seller prefers, since settlement "
        "doesn't require matching chains between buyer and seller in this backend version."
    )


if __name__ == "__main__":
    main()
