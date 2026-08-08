"""Experiment 006: Gas fee shock.

Research question: how much do agents care about transaction fees -- when
gas fees spike 5x (configs/scenarios/fee_spike.yaml, day 8), does the
average fee agents actually pay rise accordingly, and does chain usage
shift toward cheaper chains?

src/economy/shocks.py's apply_shock() intentionally leaves FEE_SPIKE as a
no-op on MacroState (fee spikes are a blockchain-layer event, not a
macroeconomic one) -- src/simulation/timestep.py's run_timestep() is the
real caller that multiplies every ChainConfig.gas_fee by the shock's
magnitude directly, once, on the day it fires. This experiment compares
pre- vs post-shock average_gas_fee_paid to show that wiring taking effect,
and separately checks whether chain usage actually shifts toward the
cheapest chain.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (  # noqa: E402
    load_simulation_config,
    make_run_dir,
    print_header,
    print_kv,
    run_simulation,
    save_run_artifacts,
    standard_metrics,
)
from metrics.gas_fee_sensitivity import average_gas_fee_paid  # noqa: E402
from src.economy.shocks import load_scenario  # noqa: E402

EXPERIMENT_NAME = "experiment_006_fee_shock"
SHOCK_DAY = 8


def _avg_fee_for_days(transactions, days: set[int]) -> float:
    relevant = [tx for tx in transactions if tx.timestep in days and tx.status.value == "settled"]
    if not relevant:
        return 0.0
    return sum(tx.gas_fee for tx in relevant) / len(relevant)


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--config", default="medium_test", choices=["small_test", "medium_test", "large_scale"])
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()

    config = load_simulation_config(args.config).model_copy(update={"scenario": "fee_spike"})
    scenario = load_scenario("fee_spike")
    fee_spike_magnitude = next(
        (shock.magnitude for shock in scenario.shocks if shock.type.value == "fee_spike"), None
    )

    print_header("Experiment 006: Gas fee shock")
    print_kv("agent_mix", config.agent_mix)
    print_kv("configured fee_spike magnitude (multiplier)", fee_spike_magnitude)

    env, sim_result = run_simulation(config, persist=args.persist)
    metrics = standard_metrics(env, sim_result.timesteps)

    num_days = len(sim_result.timesteps)
    pre_days = set(range(0, SHOCK_DAY))
    post_days = set(range(SHOCK_DAY, num_days))
    transactions = env.ledger.history()
    avg_fee_pre = _avg_fee_for_days(transactions, pre_days)
    avg_fee_post = _avg_fee_for_days(transactions, post_days)

    print_kv("chain_usage_share", metrics["chain_usage_share"])
    print_kv("average_gas_fee_paid (whole run)", metrics["average_gas_fee_paid"])
    print_kv("average_gas_fee_paid (pre-shock, days < 8)", avg_fee_pre)
    print_kv("average_gas_fee_paid (post-shock, days >= 8)", avg_fee_post)
    print_kv("fee_spike_ratio_observed (post / pre)", (avg_fee_post / avg_fee_pre) if avg_fee_pre else None)
    print_kv("fee_spike_ratio_configured", fee_spike_magnitude)

    print(
        "  note: chain_usage_share doesn't shift toward the cheapest chain here because "
        "every buyer in this agent_mix is CRRA/CARA-utility (consumer, investor), and "
        "those utility functions never reference gas_fee at all -- see "
        "experiment_010_chain_choice.py, which demonstrates gas-optimizing routing IS "
        "reachable once a multi_attribute-utility buyer profile is used."
    )

    paths = make_run_dir(EXPERIMENT_NAME)
    save_run_artifacts(
        paths,
        config,
        env,
        sim_result.timesteps,
        metrics,
        extra_metadata={
            "research_question": "Does a gas fee spike raise the average fee agents actually pay, and shift chain choice?",
            "shock_day": SHOCK_DAY,
            "fee_spike_ratio_configured": fee_spike_magnitude,
            "avg_gas_fee_paid_pre_shock": avg_fee_pre,
            "avg_gas_fee_paid_post_shock": avg_fee_post,
            "fee_spike_ratio_observed": (avg_fee_post / avg_fee_pre) if avg_fee_pre else None,
        },
    )
    print(f"\nSaved run artifacts to {paths['base']}")


if __name__ == "__main__":
    main()

