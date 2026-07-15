"""Experiment 002: Inflation shock (+10% on day 10).

Research question: does gold gain market share during inflation, and how
does that compare against the same economy with no shock at all?

Protocol:
    1. Run the same agent population/seed under configs/scenarios/baseline.yaml
       (control) and configs/scenarios/inflation_shock.yaml (treatment: +10%
       inflation fired at day 10 -- see src/economy/shocks.py).
    2. Compare gold-backed currency (PAXG, XAUT) market share pre- vs
       post-shock in the treatment run, and against the control run's
       trajectory over the same window.
    3. Save both runs' artifacts plus a comparison summary.

Uses only existing scenario configs and metrics -- no new shock or
currency logic is introduced here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (
    Environment,
    SimulationResult,
    load_simulation_config,
    make_run_dir,
    print_header,
    print_kv,
    run_simulation,
    save_json,
    save_run_artifacts,
    standard_metrics,
)

EXPERIMENT_NAME = "experiment_002_inflation"
GOLD_BACKED_SYMBOLS = {"PAXG", "XAUT"}


def _gold_share(market_share: dict[str, float]) -> float:
    return sum(share for symbol, share in market_share.items() if symbol in GOLD_BACKED_SYMBOLS)


def _gold_share_series(adoption_curve_series: dict[int, dict[str, float]]) -> dict[int, float]:
    return {day: _gold_share(shares) for day, shares in adoption_curve_series.items()}


def _run(config_name: str, scenario: str, args) -> tuple[Environment, SimulationResult, dict]:
    config = load_simulation_config(config_name).model_copy(update={"scenario": scenario})
    env, sim_result = run_simulation(config, persist=args.persist)
    metrics = standard_metrics(env, sim_result.timesteps)
    return env, sim_result, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--config", default="medium_test", choices=["small_test", "medium_test", "large_scale"])
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()

    print_header("Experiment 002: Inflation shock (+10%, day 10)")

    print_kv("condition", "control (baseline, no shock)")
    control_env, control_result, control_metrics = _run(args.config, "baseline", args)
    print_kv("gold_share_series (control)", _gold_share_series(control_metrics["adoption_curve_series"]))

    print_kv("condition", "treatment (inflation_shock)")
    treatment_env, treatment_result, treatment_metrics = _run(args.config, "inflation_shock", args)
    treatment_gold_series = _gold_share_series(treatment_metrics["adoption_curve_series"])
    print_kv("gold_share_series (treatment)", treatment_gold_series)

    shock_day = 10
    pre_shock_days = [d for d in treatment_gold_series if d < shock_day]
    post_shock_days = [d for d in treatment_gold_series if d >= shock_day]
    pre_avg = sum(treatment_gold_series[d] for d in pre_shock_days) / len(pre_shock_days) if pre_shock_days else 0.0
    post_avg = sum(treatment_gold_series[d] for d in post_shock_days) / len(post_shock_days) if post_shock_days else 0.0

    print_kv("avg_gold_share_pre_shock", pre_avg)
    print_kv("avg_gold_share_post_shock", post_avg)
    print_kv("gold_share_delta", post_avg - pre_avg)

    paths = make_run_dir(EXPERIMENT_NAME)
    save_run_artifacts(
        paths,
        load_simulation_config(args.config).model_copy(update={"scenario": "inflation_shock"}),
        treatment_env,
        treatment_result.timesteps,
        treatment_metrics,
        extra_metadata={
            "research_question": "Does gold gain market share during a 10% inflation shock?",
            "control_scenario": "baseline",
            "treatment_scenario": "inflation_shock",
            "shock_day": shock_day,
            "avg_gold_share_pre_shock": pre_avg,
            "avg_gold_share_post_shock": post_avg,
            "gold_share_delta": post_avg - pre_avg,
        },
    )
    save_json(
        paths["metrics"] / "control_metrics.json",
        {"gold_share_series": _gold_share_series(control_metrics["adoption_curve_series"]), **control_metrics},
    )
    print(f"\nSaved run artifacts to {paths['base']}")


if __name__ == "__main__":
    main()

