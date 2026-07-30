import pytest

from src.agents.population import generate_agent_population


CANDIDATE_MODELS = [f"vendor/model-{i}" for i in range(30)]  # fewer than 100, forces reuse


def test_generates_exactly_100_agents():
    population = generate_agent_population(seed=0, model_candidates=CANDIDATE_MODELS)

    assert len(population) == 100


def test_role_composition_matches_spec():
    population = generate_agent_population(seed=0, model_candidates=CANDIDATE_MODELS)

    counts = {}
    for agent in population:
        counts[agent.profile_name] = counts.get(agent.profile_name, 0) + 1

    assert counts == {"consumer": 35, "merchant": 35, "bank": 10, "investor": 10, "institution": 10}


def test_currency_zone_is_50_50_split():
    population = generate_agent_population(seed=0, model_candidates=CANDIDATE_MODELS)

    zones = [agent.currency_zone for agent in population]
    assert zones.count("USD") == 50
    assert zones.count("EUR") == 50
    assert all(zone in ("USD", "EUR") for zone in zones)


def test_cara_eligible_agents_get_individualized_a_others_stay_none():
    population = generate_agent_population(seed=0, model_candidates=CANDIDATE_MODELS)

    cara_eligible = [a for a in population if a.profile_name in ("consumer", "bank", "investor")]
    multi_attribute = [a for a in population if a.profile_name in ("merchant", "institution")]

    assert len(cara_eligible) == 55
    assert all(a.cara_coefficient is not None for a in cara_eligible)
    assert all(a.cara_coefficient in {-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0} for a in cara_eligible)
    # Genuine variance, not one value repeated 55 times (extremely unlikely with 8 buckets/55 draws).
    assert len({a.cara_coefficient for a in cara_eligible}) > 1

    assert len(multi_attribute) == 45
    assert all(a.cara_coefficient is None for a in multi_attribute)
    assert all(a.utility_type == "multi_attribute" for a in multi_attribute)


def test_a_equals_zero_builds_risk_neutral_utility():
    from src.utility.risk_neutral import RiskNeutralUtility

    # Seed sweep: find a seed that actually samples a==0.0 for at least one
    # CARA-eligible agent (8 buckets, 55 draws per population -- a seed
    # producing zero zeros across a handful of seeds would be a red flag).
    found = False
    for seed in range(20):
        population = generate_agent_population(seed=seed, model_candidates=CANDIDATE_MODELS)
        for agent in population:
            if agent.profile_name in ("consumer", "bank", "investor") and agent.cara_coefficient == 0.0:
                assert agent.utility_type == "risk_neutral"
                assert isinstance(agent.utility_fn, RiskNeutralUtility)
                found = True
    assert found, "expected at least one a==0.0 draw across 20 seeds x 55 draws each"


def test_model_assignment_uses_only_provided_candidates():
    population = generate_agent_population(seed=0, model_candidates=CANDIDATE_MODELS)

    assigned = {agent.assigned_model for agent in population}
    assert assigned.issubset(set(CANDIDATE_MODELS))
    assert all(agent.assigned_model is not None for agent in population)


def test_model_assignment_reuses_models_when_candidates_fewer_than_100():
    population = generate_agent_population(seed=0, model_candidates=CANDIDATE_MODELS)

    from collections import Counter
    counts = Counter(agent.assigned_model for agent in population)
    assert len(counts) == len(CANDIDATE_MODELS)  # every candidate used at least once
    assert max(counts.values()) >= 4  # 100 agents / 30 models -> some models get several agents


def test_same_seed_is_fully_reproducible():
    population_a = generate_agent_population(seed=42, model_candidates=CANDIDATE_MODELS)
    population_b = generate_agent_population(seed=42, model_candidates=CANDIDATE_MODELS)

    zones_a = [a.currency_zone for a in population_a]
    zones_b = [a.currency_zone for a in population_b]
    cara_a = [a.cara_coefficient for a in population_a]
    cara_b = [a.cara_coefficient for a in population_b]
    models_a = [a.assigned_model for a in population_a]
    models_b = [a.assigned_model for a in population_b]

    assert zones_a == zones_b
    assert cara_a == cara_b
    assert models_a == models_b


def test_different_seeds_produce_different_populations():
    population_a = generate_agent_population(seed=1, model_candidates=CANDIDATE_MODELS)
    population_b = generate_agent_population(seed=2, model_candidates=CANDIDATE_MODELS)

    zones_a = [a.currency_zone for a in population_a]
    zones_b = [a.currency_zone for a in population_b]
    assert zones_a != zones_b


def test_empty_model_candidates_raises_loudly():
    with pytest.raises(ValueError):
        generate_agent_population(seed=0, model_candidates=[])
