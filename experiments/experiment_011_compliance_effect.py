"""Experiment 011: Compliance effect (GENIUS Act).

Research question (per README): do governance-aware agents prefer
GENIUS-Act-compliant USDC over non-compliant USDT?

Protocol:
    1. Build an ad hoc "governance_aware" buyer profile (multi_attribute
       utility, governance_weight and compliance_weight both raised) holding
       an equal split of USDC (compliant, governance 0.95) and USDT
       (non-compliant, governance 0.55).
    2. Compare against a "governance_blind" buyer profile (governance_weight
       and compliance_weight near zero, weight shifted to liquidity, where
       USDT's 0.98 liquidity_score edges out USDC's 0.97) holding the same
       wallet.
    3. Report market_share and metrics/compliance_effects.py's
       compliance_adoption_share() for both conditions.

This is the more direct, non-LLM counterpart to experiment_007's governance
proxy -- it isolates compliance_weight specifically rather than
governance_weight, and asks the compliance-flavored version of the same
underlying question.
"""

import argparse
import sys
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
from metrics.compliance_effects import compliance_adoption_share  # noqa: E402
from src.simulation.simulation_runner import SimulationConfig  # noqa: E402

EXPERIMENT_NAME = "experiment_011_compliance_effect"
_BASE_WALLET = {"USDC": 1000.0, "USDT": 1000.0}


def _buyer_profile(name: str, governance_weight: float, compliance_weight: float) -> AgentProfileConfig:
    liquidity_weight = 1.0 - governance_weight - compliance_weight - 0.10 - 0.05
    weights = MultiAttributeWeights(
        governance_weight=governance_weight,
        compliance_weight=compliance_weight,
        liquidity_weight=max(liquidity_weight, 0.0),
        gas_fee_weight=0.10,
        volatility_weight=0.05,
    )
    return AgentProfileConfig(
        name=name,
        agent_class="buyer",
        risk_tolerance="medium",
        utility_type="multi_attribute",
        weights=weights,
        initial_wallet=dict(_BASE_WALLET),
    )


def _run(profile: AgentProfileConfig, num_agents: int, num_days: int, seed: int):
    seller = AgentProfileConfig(
        name="neutral_seller",
        agent_class="seller",
        risk_tolerance="medium",
        utility_type="multi_attribute",
        weights=MultiAttributeWeights(),
        initial_wallet=dict(_BASE_WALLET),
    )
    env = build_custom_environment(
        "baseline", {"buyer": (profile, num_agents), "seller": (seller, max(1, num_agents // 2))}
    )
    timesteps = run_environment(env, num_days=num_days, random_seed=seed)
    return env, timesteps


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--num-days", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print_header("Experiment 011: Compliance effect (GENIUS Act)")

    aware_profile = _buyer_profile("compliance_aware", governance_weight=0.35, compliance_weight=0.35)
    blind_profile = _buyer_profile("compliance_blind", governance_weight=0.0, compliance_weight=0.0)

    aware_env, aware_timesteps = _run(aware_profile, args.num_agents, args.num_days, args.seed)
    aware_metrics = standard_metrics(aware_env, aware_timesteps)
    aware_compliance_share = compliance_adoption_share(aware_env.ledger, aware_env.currencies)
    print_kv("condition", "compliance_aware (governance_weight=0.35, compliance_weight=0.35)")
    print_kv("market_share", aware_metrics["market_share"])
    print_kv("compliance_adoption_share", aware_compliance_share)

    blind_env, blind_timesteps = _run(blind_profile, args.num_agents, args.num_days, args.seed)
    blind_metrics = standard_metrics(blind_env, blind_timesteps)
    blind_compliance_share = compliance_adoption_share(blind_env.ledger, blind_env.currencies)
    print_kv("condition", "compliance_blind (governance_weight=0.0, compliance_weight=0.0)")
    print_kv("market_share", blind_metrics["market_share"])
    print_kv("compliance_adoption_share", blind_compliance_share)

    print_kv("compliance_adoption_share_delta (aware - blind)", aware_compliance_share - blind_compliance_share)
    print_kv(
        "hypothesis 'governance-aware agents prefer USDC over USDT' holds",
        aware_metrics["market_share"].get("USDC", 0.0) > blind_metrics["market_share"].get("USDC", 0.0),
    )

    record_config = SimulationConfig(
        agent_mix={"compliance_aware_buyer": args.num_agents, "neutral_seller": max(1, args.num_agents // 2)},
        num_days=args.num_days,
        scenario="baseline",
        random_seed=args.seed,
    )

    paths_aware = make_run_dir(f"{EXPERIMENT_NAME}__compliance_aware")
    save_run_artifacts(
        paths_aware,
        record_config,
        aware_env,
        aware_timesteps,
        aware_metrics,
        extra_metadata={
            "research_question": "Do governance-aware agents prefer GENIUS-Act-compliant USDC over non-compliant USDT?",
            "condition": "compliance_aware",
            "compliance_adoption_share": aware_compliance_share,
        },
    )
    paths_blind = make_run_dir(f"{EXPERIMENT_NAME}__compliance_blind")
    save_run_artifacts(
        paths_blind,
        record_config,
        blind_env,
        blind_timesteps,
        blind_metrics,
        extra_metadata={"condition": "compliance_blind", "compliance_adoption_share": blind_compliance_share},
    )
    print(f"\nSaved run artifacts to {paths_aware['base']} and {paths_blind['base']}")


if __name__ == "__main__":
    main()
