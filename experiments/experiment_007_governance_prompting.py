"""Experiment 007: Governance-aware vs governance-blind agents.

Research question (per README): does making governance salient to an agent
increase adoption of higher-governance currencies (USDC) relative to
lower-governance ones (USDT)?

Status / scope note: the README frames this as "prompt mentions governance"
vs "prompt ignores governance" -- an LLM-prompting A/B test. That requires
src/llm/agent_reasoning.py's generate_reasoning(), which is a Phase 2 stub
(raises NotImplementedError; see experiment_005_model_comparison.py). Phase 1
has no prompting path at all, so there is nothing to A/B on the actual
prompt text yet.

This experiment instead uses the closest mechanism Phase 1 actually has for
"how much does an agent weigh governance in its decision": the
MultiAttributeUtility's governance_weight (src/utility/multi_attribute.py).
Two custom in-memory agent profiles are built (not added to configs/,
per this layer's scope boundary), both holding an equal split of USDT
(low governance 0.55, high liquidity 0.98) and TDUSD (high governance
0.90, low liquidity 0.35) -- a genuine tradeoff pair, unlike USDC/USDT
where USDC dominates on every axis and masks any governance effect:
    - "governance_aware": governance_weight=0.70, liquidity_weight=0.10.
    - "governance_blind": governance_weight=0.0, liquidity_weight=0.80.
gas_fee/volatility/compliance weights are held fixed and small across both
conditions so the comparison isolates the governance/liquidity tradeoff
rather than confounding it with compliance_weight (TDUSD happens to be
GENIUS-compliant and USDT isn't). This is a faithful Phase 1 proxy for the
research question -- "does weighting governance more heavily shift currency
choice" -- without inventing an LLM prompting mechanism that doesn't exist
in this backend version.
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
from metrics.currency_usage import market_share  # noqa: E402
from src.simulation.simulation_runner import SimulationConfig  # noqa: E402

EXPERIMENT_NAME = "experiment_007_governance_prompting"

_BASE_WALLET = {"USDT": 1000.0, "TDUSD": 1000.0}


def _profile(name: str, governance_weight: float, agent_class: str = "buyer") -> AgentProfileConfig:
    """governance_weight trades directly against liquidity_weight (0.8 - governance_weight),
    holding gas_fee/volatility/compliance weights fixed and small, so any shift in currency
    choice between conditions is attributable to the governance/liquidity tradeoff itself
    rather than to compliance_weight (USDT is GENIUS-non-compliant, TDUSD is compliant --
    letting compliance_weight vary alongside governance_weight would confound the two)."""
    liquidity_weight = 0.8 - governance_weight
    weights = MultiAttributeWeights(
        governance_weight=governance_weight,
        liquidity_weight=liquidity_weight,
        gas_fee_weight=0.10,
        volatility_weight=0.05,
        compliance_weight=0.05,
    )
    return AgentProfileConfig(
        name=name,
        agent_class=agent_class,
        risk_tolerance="medium",
        utility_type="multi_attribute",
        weights=weights,
        initial_wallet=dict(_BASE_WALLET),
    )


def _run_condition(profile: AgentProfileConfig, num_agents: int, num_days: int, seed: int):
    seller_profile = _profile("neutral_seller", governance_weight=0.2, agent_class="seller")
    env = build_custom_environment(
        "baseline",
        {"buyer": (profile, num_agents), "seller": (seller_profile, max(1, num_agents // 2))},
    )
    timesteps = run_environment(env, num_days=num_days, random_seed=seed)
    return env, timesteps


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--num-agents", type=int, default=15)
    parser.add_argument("--num-days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print_header("Experiment 007: Governance-aware vs governance-blind agents")

    aware_profile = _profile("governance_aware", governance_weight=0.70)
    blind_profile = _profile("governance_blind", governance_weight=0.0)

    aware_env, aware_timesteps = _run_condition(aware_profile, args.num_agents, args.num_days, args.seed)
    aware_metrics = standard_metrics(aware_env, aware_timesteps)
    print_kv("condition", "governance_aware (governance_weight=0.70, liquidity_weight=0.10)")
    print_kv("market_share", aware_metrics["market_share"])
    print_kv("governance_preference", aware_metrics["governance_preference"])

    blind_env, blind_timesteps = _run_condition(blind_profile, args.num_agents, args.num_days, args.seed)
    blind_metrics = standard_metrics(blind_env, blind_timesteps)
    print_kv("condition", "governance_blind (governance_weight=0.0, liquidity_weight=0.80)")
    print_kv("market_share", blind_metrics["market_share"])
    print_kv("governance_preference", blind_metrics["governance_preference"])

    tdusd_aware = aware_metrics["market_share"].get("TDUSD", 0.0)
    tdusd_blind = blind_metrics["market_share"].get("TDUSD", 0.0)
    print_kv("TDUSD_share_delta (aware - blind)", tdusd_aware - tdusd_blind)

    # Use a SimulationConfig purely as a reproducibility record (this run used
    # build_custom_environment/run_environment directly, since the ad hoc
    # profiles above aren't in configs/agent_profiles/*.yaml).
    record_config = SimulationConfig(
        agent_mix={"governance_aware_buyer": args.num_agents, "neutral_seller": max(1, args.num_agents // 2)},
        num_days=args.num_days,
        scenario="baseline",
        random_seed=args.seed,
    )

    paths_aware = make_run_dir(f"{EXPERIMENT_NAME}__governance_aware")
    save_run_artifacts(
        paths_aware,
        record_config,
        aware_env,
        aware_timesteps,
        aware_metrics,
        extra_metadata={
            "research_question": "Does weighting governance more heavily in agent utility shift adoption toward the higher-governance, lower-liquidity currency (TDUSD over USDT)?",
            "condition": "governance_aware",
            "tdusd_share_aware": tdusd_aware,
            "tdusd_share_blind": tdusd_blind,
            "tdusd_share_delta": tdusd_aware - tdusd_blind,
            "scope_note": (
                "The README's framing ('prompt mentions governance' vs 'prompt ignores governance') "
                "is a Phase 2 LLM-prompting A/B test; src/llm/agent_reasoning.py is a NotImplementedError "
                "stub in this backend version. This experiment uses governance_weight in "
                "MultiAttributeUtility as the Phase 1 proxy for governance salience."
            ),
        },
    )
    paths_blind = make_run_dir(f"{EXPERIMENT_NAME}__governance_blind")
    save_run_artifacts(
        paths_blind,
        record_config,
        blind_env,
        blind_timesteps,
        blind_metrics,
        extra_metadata={"condition": "governance_blind"},
    )
    print(f"\nSaved run artifacts to {paths_aware['base']} and {paths_blind['base']}")


if __name__ == "__main__":
    main()
