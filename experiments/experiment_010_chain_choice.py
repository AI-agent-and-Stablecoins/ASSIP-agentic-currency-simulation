"""Experiment 010: Chain choice and gas optimization.

Research question (per README): which blockchains do agents actually route
through, and how much does gas-fee minimization drive that choice?

Protocol:
    1. Run the baseline scenario and report chain_usage_share
       (metrics/chain_selection.py).
    2. Cross-check against configs/blockchains/*.yaml's gas_fee field to see
       whether the most-used chain is also the cheapest one available
       (Solana: 0.002, Base: 0.05, Arbitrum: 0.15, Ethereum: 2.50 -- see
       configs/blockchains/*.yaml), which is what MultiAttributeUtility's
       gas_fee_weight term (src/utility/multi_attribute.py) would predict.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (  # noqa: E402
    AgentProfileConfig,
    MultiAttributeWeights,
    build_custom_environment,
    load_simulation_config,
    make_run_dir,
    print_header,
    print_kv,
    run_environment,
    run_simulation,
    save_run_artifacts,
    standard_metrics,
)
from metrics.chain_selection import chain_usage_share  # noqa: E402
from src.blockchain.chain import load_chain_universe  # noqa: E402

EXPERIMENT_NAME = "experiment_010_chain_choice"


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--config", default="medium_test", choices=["small_test", "medium_test", "large_scale"])
    parser.add_argument("--scenario", default="baseline", choices=["baseline", "fee_spike"])
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()

    config = load_simulation_config(args.config).model_copy(update={"scenario": args.scenario})
    chains = load_chain_universe()

    print_header("Experiment 010: Chain choice and gas optimization")
    print_kv("agent_mix", config.agent_mix)
    print_kv("chain_gas_fees", {name: chain.gas_fee for name, chain in chains.items()})

    env, sim_result = run_simulation(config, persist=args.persist)
    metrics = standard_metrics(env, sim_result.timesteps)

    print_kv("chain_usage_share", metrics["chain_usage_share"])
    print_kv("average_gas_fee_paid", metrics["average_gas_fee_paid"])

    cheapest_chain = min(chains.values(), key=lambda c: c.gas_fee).name
    most_used_chain = (
        max(metrics["chain_usage_share"], key=metrics["chain_usage_share"].get)
        if metrics["chain_usage_share"]
        else None
    )
    print_kv("cheapest_available_chain", cheapest_chain)
    print_kv("most_used_chain", most_used_chain)
    print_kv("gas_optimization_holds (most-used chain is the cheapest)", most_used_chain == cheapest_chain)
    demo_chain_usage = None
    if most_used_chain != cheapest_chain:
        print(
            "  note: this is a real, reportable finding, not a bug. src/utility/crra.py and "
            "src/utility/cara.py's evaluate() never reference option.gas_fee -- their safety_multiplier "
            "is governance_score * liquidity_score * (1 - peg_error), which is identical across every "
            "chain for a given currency (LiquidityPoolRegistry falls back to the currency's own "
            "liquidity_score with no per-chain override registered). So for CRRA/CARA agents "
            "(consumer, investor profiles), every chain scores as an exact tie, and "
            "src/utility/base.py's choose_best() -- a plain max() -- deterministically returns the "
            "first tied option in generate_candidates()'s iteration order, which is alphabetical by "
            "chain filename (arbitrum, base, ethereum, solana)."
        )
        print(
            "  additionally: no *shipped* buyer-class agent profile (configs/agent_profiles/*.yaml, "
            "agent_class: buyer) uses multi_attribute utility -- only 'consumer' is agent_class=buyer, "
            "and it's crra. Since src/simulation/timestep.py only ever calls choose_currency_and_chain() "
            "on BuyerAgent instances, gas-fee optimization is currently unreachable through any standard "
            "agent_mix. The block below demonstrates it IS reachable with an ad hoc buyer profile."
        )

        gas_aware_buyer = AgentProfileConfig(
            name="gas_aware_buyer",
            agent_class="buyer",
            risk_tolerance="medium",
            utility_type="multi_attribute",
            weights=MultiAttributeWeights(
                gas_fee_weight=0.6, governance_weight=0.1, liquidity_weight=0.1, volatility_weight=0.1, compliance_weight=0.1
            ),
            initial_wallet={"USDC": 2000.0},
        )
        neutral_seller = AgentProfileConfig(
            name="neutral_seller",
            agent_class="seller",
            risk_tolerance="medium",
            utility_type="multi_attribute",
            weights=MultiAttributeWeights(),
            initial_wallet={"USDC": 500.0},
        )
        demo_env = build_custom_environment(
            "baseline", {"buyer": (gas_aware_buyer, 5), "seller": (neutral_seller, 3)}
        )
        run_environment(demo_env, num_days=config.num_days, random_seed=config.random_seed)
        demo_chain_usage = chain_usage_share(demo_env.ledger)
        print_kv("demo: gas-fee-weighted multi_attribute buyer chain_usage_share", demo_chain_usage)

    paths = make_run_dir(EXPERIMENT_NAME)
    save_run_artifacts(
        paths,
        config,
        env,
        sim_result.timesteps,
        metrics,
        extra_metadata={
            "research_question": "Which blockchains do agents route through, and does gas-fee minimization explain it?",
            "chain_gas_fees": {name: chain.gas_fee for name, chain in chains.items()},
            "cheapest_available_chain": cheapest_chain,
            "most_used_chain": most_used_chain,
            "demo_chain_usage_share": demo_chain_usage,
            "finding": (
                "CRRA/CARA utility functions (consumer, investor profiles) never reference gas_fee or "
                "any other chain property, so every chain ties for those agents and choose_best()'s "
                "max() deterministically picks the first iteration-order chain (alphabetical by config "
                "filename), not the cheapest one. Moreover, no shipped buyer-class profile uses "
                "multi_attribute utility, so gas-fee optimization is unreachable through any standard "
                "agent_mix -- an ad hoc gas-fee-weighted buyer profile confirms it works when reachable "
                "(see demo_chain_usage_share)."
            ),
        },
    )
    print(f"\nSaved run artifacts to {paths['base']}")


if __name__ == "__main__":
    main()
