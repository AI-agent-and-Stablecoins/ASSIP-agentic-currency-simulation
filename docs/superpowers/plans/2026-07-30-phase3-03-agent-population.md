# Phase 3 Plan 3: Agent Population Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reproducible, seeded 100-agent population generator
specified in `docs/superpowers/specs/2026-07-30-phase3-plan3-agent-population-design.md`:
per-agent currency zone (50 USD/50 EUR), per-agent nominal CARA coefficient
`a` for the 55 CARA-eligible agents (consumer/bank/investor) with the
`a==0 -> risk_neutral` branch, per-agent LLM model assignment from a
preflight-verified ~99-model candidate pool, and the schema/model changes
needed to carry these three new per-agent traits through `BaseAgent` and
`AgentRecord`.

**Architecture:** A new pure function `generate_agent_population(seed,
model_candidates) -> list[BaseAgent]` (`src/agents/population.py`) is the
single entry point Plan 4 will call. It has no network dependency — the
network-dependent preflight check is a separate function,
`verify_model_candidates()` (`src/llm/llm_router.py`), that the caller runs
first and whose surviving `available` list is passed in as
`model_candidates`. `agent_factory.build_agent()` gains three optional,
keyword-only parameters so every existing call site (tests,
`Environment.build`'s count-based path, `experiments/experiment_007..011_*.py`)
is completely unaffected. `BaseAgent` and `AgentRecord` each gain three new
nullable/optional fields.

**Tech Stack:** Pydantic >=2.6, SQLAlchemy >=2.0, httpx >=0.27 (existing,
for the preflight check's `MockTransport`-based tests), no new dependencies.

## Global Constraints

- Python >=3.12, Pydantic >=2.6, SQLAlchemy >=2.0 — no new dependencies
  without checking with the user first.
- This is the final data-collection phase per the approved master spec
  (`docs/superpowers/specs/2026-07-29-phase3-full-scale-simulation-design.md`)
  and this plan's own design spec — do not add scope beyond what those two
  documents describe without checking with the user first.
- Every existing call site of `build_agent()` and `Environment.build()` must
  keep working completely unchanged — all three new `build_agent()`
  parameters are optional and keyword-only, defaulting to today's exact
  behavior.
- `cara_coefficient` (new, nominal, static) is a **separate field** from the
  existing `risk_aversion` (the value the utility function actually uses) —
  do not merge or alias these two fields. See design spec Sec 3.1.
- The **55 CARA-eligible agents** are consumer (35) + bank (10) + investor
  (10) — these three roles get their `utility_type`/`risk_aversion`
  overridden per-agent. Merchant (35) and institution (10) keep their
  existing `multi_attribute` weights completely untouched. See design spec
  Sec 2.3 for why this split, not "all 100" or "only bank."
- `a == 0.0` must build `RiskNeutralUtility` via
  `build_utility_function("risk_neutral", ...)`, never
  `build_utility_function("cara", risk_aversion=0.0, ...)` (which raises
  inside `CARAUtility.__init__`). The nominal `a = 0.0` is still recorded on
  `cara_coefficient` regardless.
- `generate_agent_population` takes an already-verified `model_candidates:
  list[str]` and has zero network/`httpx` dependency itself — the preflight
  check is a separate function the caller runs first.
- Task order matters: Task 1 (model roster + preflight) and Task 2
  (`BaseAgent` fields) must both land before Task 4 (`agent_factory`
  extension) and Task 5 (`generate_agent_population`), since both later
  tasks construct `BaseAgent`s with the new fields and assign verified model
  IDs.

---

## File Structure

- **Create:** `configs/llm/model_roster_full.yaml` (already created — 99
  entries, `id`/`label`/`name` per vendor, see file for full content)
- **Modify:** `src/llm/llm_router.py` (`ModelCandidate`,
  `ModelCandidateRoster`, `load_model_candidate_roster()`,
  `verify_model_candidates()`)
- **Modify:** `src/agents/base_agent.py` (`BaseAgent.currency_zone`,
  `.assigned_model`, `.cara_coefficient`; extend `build_llm_context()`)
- **Modify:** `src/llm/agent_reasoning.py` (`AgentUtilityContext` gains
  `currency_zone`, `assigned_model`, `cara_coefficient`)
- **Modify:** `database/models.py` (`AgentRecord` gains `currency_zone`,
  `assigned_model`, `cara_coefficient` nullable columns)
- **Modify:** `database/repository.py` (`AgentRepository.upsert_agent` passes
  the three new fields through)
- **Modify:** `src/agents/agent_factory.py` (`build_agent()` gains
  `currency_zone`, `assigned_model`, `cara_override` optional kwargs)
- **Create:** `src/agents/population.py` (`generate_agent_population()`)
- **Test:** new `tests/test_llm_router.py` extensions, new
  `tests/test_agents.py` extensions, new `tests/test_agent_reasoning.py`
  extensions, new `tests/test_llm_persistence.py` extensions, new
  `tests/test_agent_factory.py` (check first whether agent_factory already
  has a dedicated test file — the research report didn't confirm one; if
  `tests/test_agents.py` already covers `build_agent`, extend that instead),
  new `tests/test_population.py`

---

### Task 1: Full model roster config + preflight verification (collect-all-failures variant)

**Files:**
- Already created: `configs/llm/model_roster_full.yaml`
- Modify: `src/llm/llm_router.py`
- Test: extend `tests/test_llm_router.py`

**Interfaces:**
- Produces: `ModelCandidate` (Pydantic: `id: str`, `label: str`, `name:
  str`), `ModelCandidateRoster` (Pydantic: `models: list[ModelCandidate]`),
  `load_model_candidate_roster(path: Path = MODEL_ROSTER_FULL_PATH) ->
  ModelCandidateRoster`, `verify_model_candidates(candidate_ids: list[str],
  client: httpx.Client) -> tuple[list[str], list[str]]` (returns
  `(available, unavailable)`, both from `candidate_ids`, order-preserving).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm_router.py` (reuse the file's existing
`_client_with_models(available_ids)` helper — do not duplicate it):

```python
from src.llm.llm_router import (
    ModelCandidate,
    ModelCandidateRoster,
    load_model_candidate_roster,
    verify_model_candidates,
)


def test_loads_full_model_candidate_roster():
    roster = load_model_candidate_roster()
    assert len(roster.models) == 99
    ids = [entry.id for entry in roster.models]
    assert len(set(ids)) == 99  # all unique
    labels = [entry.label for entry in roster.models]
    assert len(set(labels)) == 99  # all unique
    assert any(entry.name == "GPT-5" for entry in roster.models)
    assert any(entry.name == "WizardLM" for entry in roster.models)


def test_verify_model_candidates_returns_all_available():
    roster = load_model_candidate_roster()
    all_ids = [entry.id for entry in roster.models]
    client = _client_with_models(all_ids)

    available, unavailable = verify_model_candidates(all_ids, client)

    assert set(available) == set(all_ids)
    assert unavailable == []


def test_verify_model_candidates_collects_all_failures_without_raising():
    roster = load_model_candidate_roster()
    all_ids = [entry.id for entry in roster.models]
    # Simulate 3 stale/deprecated IDs missing from OpenRouter's live roster.
    missing = {all_ids[0], all_ids[1], all_ids[2]}
    present_ids = [i for i in all_ids if i not in missing]
    client = _client_with_models(present_ids)

    available, unavailable = verify_model_candidates(all_ids, client)

    assert set(unavailable) == missing
    assert set(available) == set(present_ids)
    assert len(available) + len(unavailable) == len(all_ids)


def test_verify_model_candidates_preserves_input_order_in_available():
    client = _client_with_models(["b", "a", "c"])

    available, unavailable = verify_model_candidates(["a", "b", "c"], client)

    assert available == ["a", "b", "c"]
    assert unavailable == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_router.py -v`
Expected: FAIL with `ImportError: cannot import name 'ModelCandidate'`.

- [ ] **Step 3: Add MODEL_ROSTER_FULL_PATH, ModelCandidate, ModelCandidateRoster, load_model_candidate_roster, verify_model_candidates**

In `src/llm/llm_router.py`, find the existing `MODELS_CONFIG_PATH`
constant and `ModelEntry`/`ModelRosterConfig`/`load_model_roster` — add
alongside them (do not modify the existing ones, this is new, parallel
code for the separate ~99-model concept):

```python
MODEL_ROSTER_FULL_PATH = CONFIG_ROOT / "llm" / "model_roster_full.yaml"


class ModelCandidate(BaseModel):
    id: str
    label: str
    name: str


class ModelCandidateRoster(BaseModel):
    models: list[ModelCandidate]


def load_model_candidate_roster(path: Path = MODEL_ROSTER_FULL_PATH) -> ModelCandidateRoster:
    return load_yaml_as(path, ModelCandidateRoster)


def verify_model_candidates(candidate_ids: list[str], client: httpx.Client) -> tuple[list[str], list[str]]:
    """Preflight check for a large candidate pool: unlike verify_model_roster
    (which raises on the first missing model, correct for Phase 2's small
    fixed roster), this collects every result and returns
    (available, unavailable) so the caller can exclude failures and report
    them, rather than aborting entirely on one stale ID."""
    response = client.get("/models")
    response.raise_for_status()
    available_ids = {entry["id"] for entry in response.json()["data"]}

    available = [candidate_id for candidate_id in candidate_ids if candidate_id in available_ids]
    unavailable = [candidate_id for candidate_id in candidate_ids if candidate_id not in available_ids]
    return available, unavailable
```

(Check `CONFIG_ROOT`, `BaseModel`, `load_yaml_as`, `Path`, `httpx` are
already imported in this file — they are, since `ModelRosterConfig`/
`verify_model_roster` already use all of them; no new imports needed.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_router.py -v`
Expected: PASS (all tests in the file, including the 4 new ones).

- [ ] **Step 5: Commit**

```bash
git add configs/llm/model_roster_full.yaml src/llm/llm_router.py tests/test_llm_router.py
git commit -m "feat: add full ~99-model candidate roster and collect-all-failures preflight check"
```

---

### Task 2: BaseAgent gains currency_zone, assigned_model, cara_coefficient

**Files:**
- Modify: `src/agents/base_agent.py`
- Modify: `src/llm/agent_reasoning.py` (`AgentUtilityContext`)
- Test: extend `tests/test_agents.py`, extend `tests/test_agent_reasoning.py`

**Interfaces:**
- Produces: `BaseAgent.currency_zone: str | None = None`,
  `.assigned_model: str | None = None`, `.cara_coefficient: float | None =
  None`; `AgentUtilityContext.currency_zone`/`.assigned_model`/
  `.cara_coefficient` (same types/defaults); `build_llm_context()` passes
  them through.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agents.py` (read the file first for its existing
`BaseAgent`-construction helper/pattern and match it — likely constructs a
`BaseAgent` directly with a `Wallet`/utility_fn, or via `build_agent`):

```python
def test_base_agent_defaults_new_population_fields_to_none():
    agent = build_agent(load_agent_profiles()["consumer"])

    assert agent.currency_zone is None
    assert agent.assigned_model is None
    assert agent.cara_coefficient is None


def test_base_agent_accepts_population_fields():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)
    agent.currency_zone = "EUR"
    agent.assigned_model = "anthropic/claude-sonnet-5"
    agent.cara_coefficient = 1.5

    assert agent.currency_zone == "EUR"
    assert agent.assigned_model == "anthropic/claude-sonnet-5"
    assert agent.cara_coefficient == 1.5


def test_build_llm_context_carries_population_fields():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)
    agent.currency_zone = "USD"
    agent.assigned_model = "openai/gpt-5"
    agent.cara_coefficient = 0.5

    context = agent.build_llm_context()

    assert context.currency_zone == "USD"
    assert context.assigned_model == "openai/gpt-5"
    assert context.cara_coefficient == 0.5


def test_build_llm_context_defaults_population_fields_to_none():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)

    context = agent.build_llm_context()

    assert context.currency_zone is None
    assert context.assigned_model is None
    assert context.cara_coefficient is None
```

(Adjust imports/fixture usage to match whatever `tests/test_agents.py`
already imports for `build_agent`/`load_agent_profiles` — check the file's
top before writing, do not assume the exact import path shown above.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py -v`
Expected: FAIL with `AttributeError` (no `currency_zone` attribute) or a
Pydantic validation error when setting an undeclared field.

- [ ] **Step 3: Add the three fields to BaseAgent**

In `src/agents/base_agent.py`, add after the existing `multi_attribute_weights`
field:

```python
    multi_attribute_weights: MultiAttributeWeights | None = None
    currency_zone: str | None = None
    assigned_model: str | None = None
    cara_coefficient: float | None = None
```

In `build_llm_context()`, find the existing `AgentUtilityContext(...)`
construction call and add the three new fields to it:

```python
            currency_zone=self.currency_zone,
            assigned_model=self.assigned_model,
            cara_coefficient=self.cara_coefficient,
```

- [ ] **Step 4: Add the three fields to AgentUtilityContext**

In `src/llm/agent_reasoning.py`, find the `AgentUtilityContext` class and
add the same three fields with the same defaults, in the same style as its
existing optional fields (e.g. `risk_aversion: float | None = None`):

```python
    currency_zone: str | None = None
    assigned_model: str | None = None
    cara_coefficient: float | None = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_agents.py tests/test_agent_reasoning.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass — confirms the three new
optional `BaseAgent`/`AgentUtilityContext` fields don't break any existing
construction call site.

- [ ] **Step 7: Commit**

```bash
git add src/agents/base_agent.py src/llm/agent_reasoning.py tests/test_agents.py tests/test_agent_reasoning.py
git commit -m "feat: add currency_zone/assigned_model/cara_coefficient fields to BaseAgent"
```

---

### Task 3: AgentRecord schema + AgentRepository wiring

**Files:**
- Modify: `database/models.py` (`AgentRecord`)
- Modify: `database/repository.py` (`AgentRepository.upsert_agent`)
- Test: extend `tests/test_llm_persistence.py` (check first whether agent
  persistence already has its own test file — if `AgentRepository`/
  `AgentRecord` are tested elsewhere, extend that file instead)

**Interfaces:**
- Produces: `AgentRecord.currency_zone: str | None`,
  `.assigned_model: str | None`, `.cara_coefficient: float | None` (all
  nullable columns); `AgentRepository.upsert_agent` persists all three from
  the passed `BaseAgent`.

- [ ] **Step 1: Write the failing test**

First, grep the test suite for existing `AgentRepository`/`upsert_agent`
coverage (`grep -rn "upsert_agent\|AgentRepository" tests/`) and extend
whatever file already covers it; if none exists, create
`tests/test_agent_persistence.py` following the existing
`sqlite:///:memory:` + `Base.metadata.create_all(engine)` + `Session(engine)`
pattern used throughout `tests/test_*_persistence.py`:

```python
def test_upsert_agent_persists_population_fields():
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)
    agent.currency_zone = "EUR"
    agent.assigned_model = "anthropic/claude-sonnet-5"
    agent.cara_coefficient = 2.0

    repo.upsert_agent(agent)
    session.commit()

    row = session.get(AgentRecord, agent.agent_id)
    assert row.currency_zone == "EUR"
    assert row.assigned_model == "anthropic/claude-sonnet-5"
    assert row.cara_coefficient == 2.0


def test_upsert_agent_allows_none_population_fields():
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)  # currency_zone/assigned_model/cara_coefficient all None

    repo.upsert_agent(agent)
    session.commit()

    row = session.get(AgentRecord, agent.agent_id)
    assert row.currency_zone is None
    assert row.assigned_model is None
    assert row.cara_coefficient is None
```

- [ ] **Step 2: Run test to verify it fails**

Run the test file (path depends on Step 1's placement decision).
Expected: FAIL — `AgentRecord` has no `currency_zone` column, or
`upsert_agent` doesn't set it (so the assertion on the persisted row fails).

- [ ] **Step 3: Add the three nullable columns to AgentRecord**

In `database/models.py`, inside `AgentRecord`, add after `risk_profile`:

```python
    risk_profile: Mapped[str] = mapped_column(String)
    currency_zone: Mapped[str | None] = mapped_column(String, nullable=True)
    assigned_model: Mapped[str | None] = mapped_column(String, nullable=True)
    cara_coefficient: Mapped[float | None] = mapped_column(Float, nullable=True)
```

(Confirm `Float` is already imported in `database/models.py` — it is, used
by `TransactionRecord`/others.)

- [ ] **Step 4: Wire the fields through AgentRepository.upsert_agent**

In `database/repository.py`, in `AgentRepository.upsert_agent`, extend the
`AgentRecord(...)` construction:

```python
            record = AgentRecord(
                id=agent.agent_id,
                agent_class=agent.agent_class,
                profile_name=agent.profile_name,
                risk_profile=agent.risk_profile,
                currency_zone=agent.currency_zone,
                assigned_model=agent.assigned_model,
                cara_coefficient=agent.cara_coefficient,
                created_at=datetime.now(timezone.utc),
            )
```

- [ ] **Step 5: Run test to verify it passes**

Run the test file from Step 1.
Expected: PASS (both tests).

- [ ] **Step 6: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass.

- [ ] **Step 7: Commit**

```bash
git add database/models.py database/repository.py tests/
git commit -m "feat: add currency_zone/assigned_model/cara_coefficient columns to AgentRecord"
```

---

### Task 4: agent_factory.build_agent() gains optional per-agent overrides

**Files:**
- Modify: `src/agents/agent_factory.py`
- Test: extend `tests/test_agents.py`

**Interfaces:**
- Produces: `build_agent(profile: AgentProfileConfig, *, currency_zone:
  str | None = None, assigned_model: str | None = None, cara_override:
  tuple[str, float] | None = None) -> BaseAgent`. `cara_override`, when
  provided, is `(utility_type, risk_aversion)` — supersedes
  `profile.utility_type`/`profile.risk_aversion` for utility-function
  construction only; `cara_coefficient` on the returned agent is set
  separately (see Step 3).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agents.py`:

```python
def test_build_agent_with_no_overrides_behaves_exactly_as_before():
    profile = load_agent_profiles()["consumer"]

    agent = build_agent(profile)

    assert agent.currency_zone is None
    assert agent.assigned_model is None
    assert agent.cara_coefficient is None
    assert agent.utility_type == profile.utility_type
    assert agent.risk_aversion == profile.risk_aversion


def test_build_agent_accepts_currency_zone_and_assigned_model():
    profile = load_agent_profiles()["consumer"]

    agent = build_agent(profile, currency_zone="EUR", assigned_model="openai/gpt-5")

    assert agent.currency_zone == "EUR"
    assert agent.assigned_model == "openai/gpt-5"


def test_build_agent_cara_override_supersedes_profile_utility():
    profile = load_agent_profiles()["consumer"]  # profile.utility_type == "crra"

    agent = build_agent(profile, cara_override=("cara", 1.5))

    assert agent.utility_type == "cara"
    assert agent.risk_aversion == 1.5
    assert agent.cara_coefficient == 1.5
    assert isinstance(agent.utility_fn, CARAUtility)


def test_build_agent_cara_override_risk_neutral_branch():
    profile = load_agent_profiles()["bank"]  # profile.utility_type == "cara"

    agent = build_agent(profile, cara_override=("risk_neutral", None))

    assert agent.utility_type == "risk_neutral"
    assert isinstance(agent.utility_fn, RiskNeutralUtility)
    assert agent.cara_coefficient == 0.0  # nominal a is still recorded even though utility_type switched
```

(Check the file's existing imports for `CARAUtility`/`RiskNeutralUtility` —
add them if not already imported; match whatever import style the file
already uses for `build_agent`/`load_agent_profiles`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py -v`
Expected: FAIL with `TypeError: build_agent() got an unexpected keyword argument 'currency_zone'`.

- [ ] **Step 3: Extend build_agent**

In `src/agents/agent_factory.py`, replace `build_agent`:

```python
def build_agent(
    profile: AgentProfileConfig,
    *,
    currency_zone: str | None = None,
    assigned_model: str | None = None,
    cara_override: tuple[str, float | None] | None = None,
) -> BaseAgent:
    agent_cls = _AGENT_CLASSES[profile.agent_class]

    if cara_override is not None:
        utility_type, risk_aversion = cara_override
        nominal_cara = risk_aversion if risk_aversion is not None else 0.0
    else:
        utility_type, risk_aversion = profile.utility_type, profile.risk_aversion
        nominal_cara = None

    utility_fn = build_utility_function(utility_type, risk_aversion, profile.weights, profile.eis)
    wallet = Wallet(balances=dict(profile.initial_wallet))
    return agent_cls(
        agent_id=generate_id(profile.agent_class),
        agent_class=profile.agent_class,
        profile_name=profile.name,
        risk_profile=profile.risk_tolerance,
        wallet=wallet,
        utility_fn=utility_fn,
        utility_type=utility_type,
        risk_aversion=risk_aversion,
        eis=profile.eis,
        multi_attribute_weights=profile.weights,
        currency_zone=currency_zone,
        assigned_model=assigned_model,
        cara_coefficient=nominal_cara,
    )
```

(Note: when `cara_override=("risk_neutral", None)`, `risk_aversion` passed
to `build_utility_function`/stored on the agent is `None` — matching how
`RiskNeutralUtility` ignores it — but `cara_coefficient` is explicitly `0.0`,
the nominal `a` that triggered the risk-neutral branch, per Step 3's test.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents.py -v`
Expected: PASS (all tests, including the 4 new ones).

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass — confirms every existing
`build_agent(profile)` call site (positional-only, no new kwargs) is
completely unaffected.

- [ ] **Step 6: Commit**

```bash
git add src/agents/agent_factory.py tests/test_agents.py
git commit -m "feat: add optional currency_zone/assigned_model/cara_override params to build_agent"
```

---

### Task 5: generate_agent_population — the population constructor

**Files:**
- Create: `src/agents/population.py`
- Test: new `tests/test_population.py`

**Interfaces:**
- Consumes: `build_agent` (Task 4), `load_agent_profiles` (existing),
  `ModelCandidateRoster`/`load_model_candidate_roster` (Task 1, for the
  human-facing candidate ID list — the function itself just takes
  `model_candidates: list[str]`).
- Produces: `generate_agent_population(seed: int, model_candidates:
  list[str]) -> list[BaseAgent]`. Deterministic for a given `(seed,
  model_candidates)` pair.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_population.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_population.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.population'`.

- [ ] **Step 3: Implement generate_agent_population**

Create `src/agents/population.py`:

```python
"""Reproducible 100-agent population generator for Phase 3's full-scale
run. See docs/superpowers/specs/2026-07-30-phase3-plan3-agent-population-design.md
for the role/zone/CARA/model-assignment design this implements.

Has no network dependency: the model-candidate list passed in is assumed
to already be preflight-verified (src.llm.llm_router.verify_model_candidates)
by the caller -- this module only samples from it.
"""

import random

from src.agents.agent_factory import build_agent
from src.agents.base_agent import BaseAgent
from src.agents.agent_factory import load_agent_profiles

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
                cara_override = ("risk_neutral", None) if a == 0.0 else ("cara", a)

            assigned_model = shuffled_models[slot_index % len(shuffled_models)]
            agent = build_agent(
                profile,
                currency_zone=zones[slot_index],
                assigned_model=assigned_model,
                cara_override=cara_override,
            )
            population.append(agent)
            slot_index += 1

    return population
```

(Note: `cara_override=("risk_neutral", None)` on the `a == 0.0` branch relies
on Task 4's `build_agent` setting `cara_coefficient=0.0` in that exact case
— confirm this still holds; if Task 4's implementation differs, adjust this
call accordingly rather than re-deriving the branch here.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_population.py -v`
Expected: PASS (all 10 tests). If `test_a_equals_zero_builds_risk_neutral_utility`
is flaky (no seed in `range(20)` happens to draw `a==0.0` for a
CARA-eligible agent across 55 draws each — astronomically unlikely with 8
roughly-equal buckets, but not impossible), widen the seed range rather than
weakening the assertion.

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agents/population.py tests/test_population.py
git commit -m "feat: add generate_agent_population for the reproducible 100-agent population"
```

---

## What comes after this plan

1. **Matrix runner / experiment orchestration (Plan 4)** — calls
   `verify_model_candidates()` once before any run starts (reporting
   `unavailable` to the user), then `generate_agent_population(seed,
   available)` once per seed; persists one `AgentRecord` per agent via
   `AgentRepository.upsert_agent` at run start; implements the loss-driven
   `a`-adaptation formula (master spec Sec 3.3) as a per-timestep mechanism
   writing to `AgentStateRecord.cara_coefficient`; wires the 100-agent
   population into the master simulation + 7 domestic + 7 cross-border
   sandboxes, 365 days, 5 seeds each.
2. **Econometrics engine (Plan 5)** — H1–H5 regression outputs, regressing
   on `AgentRecord.cara_coefficient` (nominal) and/or
   `AgentStateRecord.cara_coefficient` (time-varying) depending on which the
   hypothesis calls for, plus `currency_zone` and `assigned_model` as
   covariates.
