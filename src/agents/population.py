"""Reproducible 100-agent population generator for Phase 3's full-scale
run. See docs/superpowers/specs/2026-07-30-phase3-plan3-agent-population-design.md
for the role/zone/CARA/model-assignment design this implements.

Has no network dependency: the model-candidate list passed in is assumed
to already be preflight-verified (src.llm.llm_router.verify_model_candidates)
by the caller -- this module only samples from it.
"""

import random

from src.agents.agent_factory import build_agent, cara_utility_fields, load_agent_profiles
from src.agents.base_agent import BaseAgent

ROLE_COUNTS = {
    "consumer": 35,
    "merchant": 35,
    "bank": 10,
    "investor": 10,
    "institution": 10,
}

CARA_ELIGIBLE_ROLES = {"consumer", "bank", "investor"}

CARA_SAMPLE_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def generate_agent_population(seed: int, model_candidates: list[str]) -> list[BaseAgent]:
    if not model_candidates:
        raise ValueError("generate_agent_population requires at least one verified model candidate")

    rng = random.Random(seed)
    profiles = load_agent_profiles()

    total_agents = sum(ROLE_COUNTS.values())
    zones = ["USD"] * (total_agents // 2) + ["EUR"] * (total_agents // 2)
    rng.shuffle(zones)

    shuffled_models = list(model_candidates)
    rng.shuffle(shuffled_models)

    population: list[BaseAgent] = []
    slot_index = 0
    for profile_name, count in ROLE_COUNTS.items():
        profile = profiles[profile_name]
        for _ in range(count):
            cara_override = None
            if profile_name in CARA_ELIGIBLE_ROLES:
                a = rng.choice(CARA_SAMPLE_VALUES)
                cara_override = cara_utility_fields(a)

            assigned_model = shuffled_models[slot_index % len(shuffled_models)]
            deterministic_id = f"{profile_name}-seed{seed}-{slot_index:03d}"
            agent = build_agent(
                profile,
                currency_zone=zones[slot_index],
                assigned_model=assigned_model,
                cara_override=cara_override,
                agent_id=deterministic_id,
            )
            population.append(agent)
            slot_index += 1

    return population


HYPOTHESIS_ROLE_COUNTS = {
    "consumer": 40,
    "bank": 8,
    "investor": 8,
    "merchant": 35,
    "institution": 9,
}

RISK_AVERSION_COHORTS = [0.0, 2.0, 4.0, 6.0]

HYPOTHESIS_EIS = 1.0

HYPOTHESIS_UTILITY_TYPES = {"crra", "cara", "epstein_zin_proxy"}


def generate_hypothesis_population(seed: int, model_candidates: list[str], utility_type: str) -> list[BaseAgent]:
    if not model_candidates:
        raise ValueError("generate_hypothesis_population requires at least one verified model candidate")
    if utility_type not in HYPOTHESIS_UTILITY_TYPES:
        raise ValueError(f"utility_type must be one of {HYPOTHESIS_UTILITY_TYPES}, got {utility_type!r}")

    rng = random.Random(seed)
    profiles = load_agent_profiles()

    total_agents = sum(HYPOTHESIS_ROLE_COUNTS.values())
    zones = ["USD"] * (total_agents // 2) + ["EUR"] * (total_agents // 2)
    rng.shuffle(zones)

    shuffled_models = list(model_candidates)
    rng.shuffle(shuffled_models)

    population: list[BaseAgent] = []
    slot_index = 0
    for profile_name, count in HYPOTHESIS_ROLE_COUNTS.items():
        profile = profiles[profile_name]
        if profile_name in CARA_ELIGIBLE_ROLES:
            profile = profile.model_copy(update={"eis": HYPOTHESIS_EIS}) if utility_type == "epstein_zin_proxy" else profile
            cohort_assignment = [RISK_AVERSION_COHORTS[i % len(RISK_AVERSION_COHORTS)] for i in range(count)]
            rng.shuffle(cohort_assignment)

        for i in range(count):
            cara_override = None
            if profile_name in CARA_ELIGIBLE_ROLES:
                if utility_type == "cara":
                    cara_override = cara_utility_fields(cohort_assignment[i])
                else:
                    cara_override = (utility_type, cohort_assignment[i])

            assigned_model = shuffled_models[slot_index % len(shuffled_models)]
            deterministic_id = f"{profile_name}-seed{seed}-{slot_index:03d}"
            agent = build_agent(
                profile,
                currency_zone=zones[slot_index],
                assigned_model=assigned_model,
                cara_override=cara_override,
                agent_id=deterministic_id,
            )
            population.append(agent)
            slot_index += 1

    return population
