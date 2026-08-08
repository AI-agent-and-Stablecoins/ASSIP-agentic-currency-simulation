"""Experiment 001: Baseline -- normal economy, no shocks.

Research question: in a shock-free economy, which currencies do agents
converge on, and how quickly does that preference stabilize?

Protocol:
    1. Load configs/simulation/medium_test.yaml (20 consumers, 10 merchants,
       2 banks, 5 investors) with configs/scenarios/baseline.yaml (no shocks).
    2. Run it unmodified via SimulationRunner.
    3. Collect the standard metrics report.
    4. Save metadata/transactions/negotiations/metrics to outputs/.

This experiment does not modify any backend logic -- it only loads an
existing config and calls the existing simulation/metrics modules.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (
    load_simulation_config,
    make_run_dir,
    print_header,
    print_kv,
    run_simulation,
    save_run_artifacts,
    standard_metrics,
)

EXPERIMENT_NAME = "experiment_001_baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--config", default="medium_test", choices=["small_test", "medium_test", "large_scale"])
    parser.add_argument("--persist", action="store_true", help="Also write results to the database")
    args = parser.parse_args()

    config = load_simulation_config(args.config)

    print_header("Experiment 001: Baseline (normal economy)")
    print_kv("scenario", config.scenario)
    print_kv("agent_mix", config.agent_mix)
    print_kv("num_days", config.num_days)

    env, sim_result = run_simulation(config, persist=args.persist)
    metrics = standard_metrics(env, sim_result.timesteps)

    dominant_currency = (
        max(metrics["market_share"], key=metrics["market_share"].get) if metrics["market_share"] else None
    )

    print_kv("market_share", metrics["market_share"])
    print_kv("dominant_currency", dominant_currency)
    print_kv("transaction_success_rate", metrics["transaction_success_rate"])
    print_kv("wealth_distribution", metrics["wealth_distribution"])

    paths = make_run_dir(EXPERIMENT_NAME)
    save_run_artifacts(
        paths,
        config,
        env,
        sim_result.timesteps,
        metrics,
        extra_metadata={"research_question": "Which currency dominates a shock-free economy, and how fast?"},
    )
    print(f"\nSaved run artifacts to {paths['base']}")


if __name__ == "__main__":
    main()
