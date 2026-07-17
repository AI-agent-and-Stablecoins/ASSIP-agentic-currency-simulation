"""Experiment 008: Liquidity vs. fees.

Research question (per README): do risk-averse agents prefer liquidity over
fee savings?

Protocol:
    1. Run configs/simulation/medium_test.yaml (which already mixes
       consumer/merchant/bank/investor -- see configs/agent_profiles/*.yaml)
       under the fee_spike scenario, so fee pressure is part of the picture.
    2. Group the resulting ledger by each agent's risk_profile field
       (configs/agent_profiles/*.yaml's risk_tolerance: low/medium/high,
       carried onto BaseAgent.risk_profile by agent_factory.build_agent).
    3. Compare metrics/liquidity_sensitivity.py's liquidity_sensitivity()
       and metrics/gas_fee_sensitivity.py's average_gas_fee_paid() within
       each risk-profile group.

This only groups/filters the existing Ledger (via experiments._common's
ledger_subset helper) and calls existing metrics/*.py functions -- no new
metric or utility logic is added.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (  # noqa: E402
    agent_ids_by_predicate,
    ledger_subset,
    load_simulation_config,
    make_run_dir,
    print_header,
    print_kv,
    run_simulation,
    save_json,
    save_run_artifacts,
    standard_metrics,
)
from metrics.gas_fee_sensitivity import average_gas_fee_paid  # noqa: E402
from metrics.liquidity_sensitivity import liquidity_sensitivity  # noqa: E402

EXPERIMENT_NAME = "experiment_008_liquidity_vs_fees"
RISK_PROFILES = ("low", "medium", "high")


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--config", default="medium_test", choices=["small_test", "medium_test", "large_scale"])
    parser.add_argument("--scenario", default="fee_spike", choices=["baseline", "fee_spike"])
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()

    config = load_simulation_config(args.config).model_copy(update={"scenario": args.scenario})

    print_header("Experiment 008: Liquidity vs. fees, by risk profile")
    print_kv("agent_mix", config.agent_mix)
    print_kv("scenario", args.scenario)

    env, sim_result = run_simulation(config, persist=args.persist)
    metrics = standard_metrics(env, sim_result.timesteps)

    by_risk_profile: dict[str, dict[str, float]] = {}
    for risk_profile in RISK_PROFILES:
        agent_ids = agent_ids_by_predicate(env, lambda agent, rp=risk_profile: agent.risk_profile == rp)
        if not agent_ids:
            continue
        subset = ledger_subset(env.ledger, agent_ids)
        by_risk_profile[risk_profile] = {
            "num_agents": len(agent_ids),
            "liquidity_sensitivity": liquidity_sensitivity(subset, env.currencies),
            "average_gas_fee_paid": average_gas_fee_paid(subset),
        }
        print_kv(f"risk_profile={risk_profile}", by_risk_profile[risk_profile])
        if by_risk_profile[risk_profile]["num_agents"] > 0 and len(subset.history()) == 0:
            print(
                f"    note: {risk_profile}-risk agents recorded zero settled transactions. In this "
                "backend version src/simulation/timestep.py only activates BuyerAgent/SellerAgent "
                "instances each day (see agent_activation_order/timestep.py) -- InvestorAgent and "
                "BankAgent are otherwise-passive Phase 1 classes (see their docstrings), so a "
                "risk_profile made up mostly of investors will show 0.0 here as an artifact of that, "
                "not necessarily a preference finding."
            )

    if (
        "low" in by_risk_profile
        and "high" in by_risk_profile
        and by_risk_profile["low"]["num_agents"] > 0
        and by_risk_profile["high"]["num_agents"] > 0
    ):
        low_has_data = ledger_subset(
            env.ledger, agent_ids_by_predicate(env, lambda agent: agent.risk_profile == "low")
        ).history()
        high_has_data = ledger_subset(
            env.ledger, agent_ids_by_predicate(env, lambda agent: agent.risk_profile == "high")
        ).history()
        if low_has_data and high_has_data:
            hypothesis_holds = (
                by_risk_profile["low"]["liquidity_sensitivity"] >= by_risk_profile["high"]["liquidity_sensitivity"]
            )
            print_kv(
                "hypothesis 'risk-averse (low-risk) agents prefer liquidity over fee savings' holds",
                hypothesis_holds,
            )
        else:
            print_kv(
                "hypothesis check",
                "skipped -- one or both groups had zero settled transactions (see note above)",
            )

    paths = make_run_dir(EXPERIMENT_NAME)
    save_run_artifacts(
        paths,
        config,
        env,
        sim_result.timesteps,
        metrics,
        extra_metadata={
            "research_question": "Do risk-averse agents prefer liquidity over fee savings?",
            "by_risk_profile": by_risk_profile,
        },
    )
    save_json(paths["metrics"] / "by_risk_profile.json", by_risk_profile)
    print(f"\nSaved run artifacts to {paths['base']}")


if __name__ == "__main__":
    main()
