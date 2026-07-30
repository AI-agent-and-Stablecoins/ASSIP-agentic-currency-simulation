# Phase 3 Plan 3: Agent Population Generation — Design Spec

**Status:** Design spec, not yet implemented.
**Scope:** Builds the static, per-agent-individualized 100-agent population
(role, currency zone, nominal CARA coefficient `a`, assigned LLM model) that
every later Phase 3 component (matrix runner, econometrics engine) consumes.
Grounded in `docs/superpowers/specs/2026-07-29-phase3-full-scale-simulation-design.md`
§3 (the authoritative source) and the actual current codebase state,
confirmed by direct inspection (see "Current state" under each section).
Nothing in this spec should be modified, added to, or reinterpreted without
checking back with the user first.

---

## 1. What this plan does NOT do

- **Loss-driven `a` adaptation** (master spec §3.3's `a_{t+1} = min(a_max, a_t +
  eta_risk * Loss_t / W_real_t)` formula): this requires a per-timestep
  realized-loss computation, a real-wealth denominator, and persistence into
  `agent_states.cara_coefficient` (a per-timestep table whose write path is
  explicitly Plan 4's job per
  `docs/superpowers/plans/2026-07-29-phase3-01-foundation-persistence.md`'s
  own "what comes after" section: "Matrix runner ... writing into
  simulation_runs, timestep_logs, and agent_states"). This plan establishes
  only the **nominal, static** `a` per agent and a mutable slot for Plan 4 to
  later update each timestep — it does not implement the adaptation formula
  itself. Recorded explicitly here so it is a documented hand-off, not
  something rediscovered as "missing" in a later review.
- **The matrix runner / experiment orchestration** (Plan 4) — this plan only
  produces a `list[BaseAgent]` (or equivalent) population; wiring that
  population into a real 365-day run across the master sim + 14 sandboxes is
  Plan 4's job.
- **The econometrics engine** (Plan 5).

---

## 2. Role composition and per-agent trait assignment

### 2.1 Role composition (100 agents, unchanged from master spec §3.1)

| Role | Count | Existing profile file |
|---|---|---|
| Consumer (buyer) | 35 | `configs/agent_profiles/consumer.yaml` |
| Merchant (seller) | 35 | `configs/agent_profiles/merchant.yaml` |
| Bank | 10 | `configs/agent_profiles/bank.yaml` |
| Investor | 10 | `configs/agent_profiles/investor.yaml` |
| Institution | 10 | `configs/agent_profiles/institution.yaml` |

**Current state (confirmed by reading all 5 files in full):** `institution.yaml`
sets `agent_class: investor`, so institution and investor agents both
instantiate `InvestorAgent` today, distinguished only by `profile_name`. This
plan does not change that — `agent_mix`-style role-count construction already
supports it via `Environment.build`'s existing `profile_name -> count` dict.

### 2.2 Home currency zone

Each of the 100 agents gets `currency_zone: Literal["USD", "EUR"]`, **50/50
split, independent of role** (both US-zone and EU-zone consumers exist, etc).
Assignment: build a list of 50 `"USD"` + 50 `"EUR"` labels, shuffle with the
population's seeded `random.Random`, zip onto the 100 agent slots in
construction order.

**Current state:** no `currency_zone` concept exists anywhere in `src/` or
`configs/` (confirmed by grep — zero hits). New field on `BaseAgent` and a
new column on `AgentRecord`.

### 2.3 Per-agent nominal CARA coefficient `a`

**Which agents get an individualized `a` — resolving an ambiguity in the
master spec:** §3.3 says both "each of the 100 agents is individually
assigned an `a`" and, two sentences later, "only the CARA-eligible roles' `a`
is individualized" while multi-attribute roles "stay as currently
configured." These two statements are only consistent under one reading,
which this plan adopts: **"CARA-eligible roles" means roles whose profile
uses a wealth-utility function at all (`crra`/`cara`), not roles whose YAML
happens to already say `utility_type: cara` today.** Concretely:

- **Consumer, Bank, Investor** (currently `crra`, `cara`, `crra` respectively
  per the actual YAML) — all three are converted to **individualized CARA**:
  each of these 55 agents (35 + 10 + 10) gets its own sampled `a`, overriding
  the role's flat `risk_aversion` value from the YAML profile.
- **Merchant, Institution** (currently `multi_attribute`) — these 45 agents
  (35 + 10) **keep their existing role-level `multi_attribute` weights
  unchanged**. `multi_attribute` has no `risk_aversion` parameter to
  individualize in the first place — the "stays as currently configured"
  clause refers to exactly this group.

This gives 55/100 agents genuine within-sample CARA variance for H1–H3's
regression (not "5 discrete role clusters," per the master spec's own stated
goal), while leaving the multi_attribute roles' weight vectors untouched, per
the master spec's explicit carve-out. **This resolution is flagged here for
the user to override if a different reading was intended** — it is the only
internally-consistent reading of §3.3's two sentences, not a guess made in
the absence of any signal.

**Sampling:** for each of the 55 CARA-eligible agents, sample `a` uniformly
(with replacement) from `{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0}` using the
population's seeded `random.Random`.

**The `a == 0` branch:** `CARAUtility.__init__` raises `ValueError` when
`risk_aversion == 0` (confirmed: `src/utility/cara.py`, guards `== 0`
exactly, no tolerance). Per the master spec, an agent sampled at `a = 0.0`
must be built with `utility_type = "risk_neutral"`
(`build_utility_function("risk_neutral", ...)` → `RiskNeutralUtility()`,
confirmed to take no constructor params) instead of `"cara"`. **The nominal
`a = 0.0` is still recorded as a plain field** on the agent and in
`AgentRecord`/`agent_states`, regardless of which utility class actually
backs `utility_fn` — the econometrics engine regresses on the recorded `a`,
not on `utility_type`.

**Current state:** confirmed via `src/utility/utility_factory.py` that
`build_utility_function("cara", risk_aversion=0.0, ...)` would raise inside
`CARAUtility.__init__` (the factory itself only guards `risk_aversion is
None`, not `== 0`) — so this branch must be handled by the population
constructor, not the factory or `CARAUtility` itself.

### 2.4 Per-agent LLM model assignment

**Source data:** the user-supplied `Phase 4 Model List.md` (pasted into
conversation, not a repo file) lists ~99 models by human-readable name across
17 vendors (OpenAI 10, Anthropic 7, Google 7, xAI 4, Meta 8, DeepSeek 8, Qwen
8, Mistral 8, Cohere 4, Microsoft 4, Google Gemma 5, MiniMax 3, Moonshot 3,
Alibaba 3, NVIDIA 2, Amazon 3, Miscellaneous 12). This plan translates each
name to a best-effort OpenRouter model ID (standard `vendor-prefix/model-slug`
convention, e.g. `"Claude Opus 4"` → `anthropic/claude-opus-4`, `"Llama 4
Maverick"` → `meta-llama/llama-4-maverick`). **These ID guesses are expected
to be imperfect** — some of the listed models (several are prior-generation,
e.g. Anthropic's Claude 3/3.5/3.7/4 family, GPT-4.1/4o predating GPT-5) may
have been deprecated/renamed on OpenRouter by run time, or the slug
convention may not match exactly. This is not a defect to fix by hand-tuning
IDs further — it is exactly what the preflight verification step (§2.4.1
below) exists to catch: wrong/stale IDs are excluded from the sampling pool
and reported to the user, never silently kept or silently dropped after the
fact.

The full name → best-effort-ID mapping lives in a new config,
`configs/llm/model_roster_full.yaml` (distinct from the existing 5-model
`configs/llm/models.yaml`, which is Phase 2's separate A/B-comparison roster
for `agent_reasoning.py`'s `_model_ids_for_policy` — orthogonal to this
one-fixed-model-per-agent concept and not modified by this plan).

#### 2.4.1 Preflight verification (new function, not a reuse of `verify_model_roster`)

**Current state:** `src/llm/llm_router.py`'s existing `verify_model_roster()`
takes a roster and an `httpx.Client`, does one `GET /models`, and **raises on
the first missing model** — it has no return value and does not report which
models passed. Confirmed by reading the full function body. This is the
correct behavior for Phase 2's fixed 5-model roster (any missing model there
is a hard configuration error), but wrong for Plan 3's "verify ~99
candidates, exclude failures, report survivors" requirement.

**New function**, `src/llm/llm_router.py` (or a new
`src/agents/population.py`-local helper — task breakdown decides placement):
`verify_model_candidates(candidate_ids: list[str], client: httpx.Client) ->
tuple[list[str], list[str]]` — returns `(available, unavailable)`, doing the
same single `GET /models` call and set-membership check as
`verify_model_roster`, but collecting **all** results instead of raising on
the first miss. Population construction uses `available` as the sampling
pool and must report `unavailable` (candidate ID + originating human-readable
name) to the user before any assignment happens — never silently drop them.

#### 2.4.2 Assignment

Shuffle the `available` pool (seeded `random.Random`), assign round-robin (or
one-per-slot then wrap) to fill exactly 100 agent slots — "a handful of
models get 2 agents to reach 100" per the master spec, since the available
pool is expected to be under 100 after preflight exclusions. If preflight
leaves **zero** available models (e.g. no `OPENROUTER_API_KEY`/network in a
test environment), population construction must fail loudly with a clear
error, not silently fall back to an empty assignment — tests exercise this
path with a fake/mock `httpx.Client` transport instead of a live network call
(matching the existing test pattern for `verify_model_roster` — check
`tests/test_llm_router.py` for the transport-mocking convention already in
use before inventing a new one).

**Current state:** `SimulationRunRecord.model_roster_summary` (Plan 1) and
its docstring already anticipated "one model per agent, not one per run" —
confirmed in `database/models.py`. No per-agent field to record the actual
assignment exists yet; this plan adds it.

---

## 3. Schema and model changes

### 3.1 `BaseAgent` (`src/agents/base_agent.py`)

Add three fields (all required at construction, no defaults, since every
agent in the 100-agent population must have them — existing single-profile
`build_agent()` callers, e.g. existing tests and `Environment.build`'s
count-based path, need a sensible default so they don't break; see §4):

```python
currency_zone: str | None = None       # "USD" | "EUR"; None for legacy single-profile construction
assigned_model: str | None = None      # OpenRouter model ID; None for legacy construction
cara_coefficient: float | None = None  # nominal a; distinct from the existing risk_aversion field
```

`cara_coefficient` is deliberately a **new, separate field** from the
existing `risk_aversion: float | None` — `risk_aversion` remains whatever
value the utility function was actually constructed with (including the
role-flat value for merchant/institution's `multi_attribute` agents, which
have no CARA `a` at all), while `cara_coefficient` is specifically the
recorded nominal `a` for econometrics, populated only for the 55 CARA-eligible
agents (`None` for merchant/institution agents). Reusing `risk_aversion` for
this would conflate "the parameter the utility function actually uses" with
"the trait H1-H3 regresses on," which happen to be the same 55/100 agents
but are conceptually distinct fields the population constructor must not
merge.

`build_llm_context()` extends `AgentUtilityContext` construction to pass
these three new fields through (the method already does a lazy import of
`AgentUtilityContext` from `src.llm.agent_reasoning` — extend that
construction call, extend `AgentUtilityContext` itself to accept them).

### 3.2 `AgentRecord` (`database/models.py`)

Add three nullable columns, matching `BaseAgent`'s new fields exactly:

```python
currency_zone: Mapped[str | None] = mapped_column(String, nullable=True)
assigned_model: Mapped[str | None] = mapped_column(String, nullable=True)
cara_coefficient: Mapped[float | None] = mapped_column(Float, nullable=True)
```

Nullable because existing tests/experiments construct single agents via
`build_agent()` outside the 100-agent population path (e.g. `Environment.build`'s
existing count-based construction) and must continue to work unchanged (see
§4) — this plan does not require every `AgentRecord` in the system to have
these fields populated, only the ones produced by the new population
constructor.

Note this is a different table from `AgentStateRecord` (Plan 1), which
already has a **per-timestep** `cara_coefficient` column for the
loss-driven-adaptation value (§1, deferred to Plan 4). `AgentRecord`'s new
`cara_coefficient` is the one-row-per-agent **nominal/initial** value;
`AgentStateRecord.cara_coefficient` is the time-varying value Plan 4 will
write once per agent per day. Two columns with the same name in two
different tables serving two different purposes — documented here explicitly
so it isn't mistaken for redundancy in a later review.

### 3.3 `AgentProfileConfig` / role YAMLs

**No changes.** The five `configs/agent_profiles/*.yaml` files stay exactly
as they are — the population constructor reads each agent's role profile for
its `initial_wallet`, `risk_tolerance`, and (for merchant/institution)
`multi_attribute` weights, but overrides `utility_type`/`risk_aversion` for
the 55 CARA-eligible agents at construction time rather than mutating the
YAML or `AgentProfileConfig` schema.

---

## 4. Population constructor (new code, new entry point)

**Current state:** `src/agents/agent_factory.py`'s `build_agent(profile)`
builds exactly one agent from one profile, with no per-agent override
parameters at all — confirmed by reading the full 69-line file.
`Environment.build(scenario_name, agent_mix: dict[str, int])` only supports
uniform, count-based construction (every agent of a role gets identical
`risk_aversion`, no zone, no model). **Neither of these is modified in a
breaking way** — both stay exactly as they are today for existing callers
(existing tests, `experiments/experiment_007..011_*.py`, which all use
`Environment.build`'s count-based path).

**New code:**

- `src/agents/agent_factory.py`: extend `build_agent(profile, *,
  currency_zone=None, assigned_model=None, cara_override=None) -> BaseAgent`
  with three new **optional, keyword-only** parameters, defaulting to
  today's exact behavior when omitted (so every existing call site — direct
  calls in tests, `Environment.build`'s loop — is untouched and unaffected).
  When `cara_override` is provided (a `(utility_type, risk_aversion)` tuple,
  or `None` to mean "use the profile's own configured values unchanged"),
  it supersedes `profile.utility_type`/`profile.risk_aversion` for
  utility-function construction; `cara_coefficient` on the resulting
  `BaseAgent` is always set from the raw sampled `a`, independent of
  `cara_override`'s risk-neutral branching.
- New module `src/agents/population.py`: `generate_agent_population(seed:
  int, model_candidates: list[str]) -> list[BaseAgent]` — the single
  entry point Plan 4 will call. Builds all 100 agents (role counts from
  §2.1, currency zone from §2.2, CARA sampling + `a==0` branch from §2.3,
  model assignment from §2.4/§2.4.2), all seeded from one `random.Random(seed)`
  so the whole population is exactly reproducible given a seed — required
  by the master spec's provenance requirements (§9, `random_seed` capture).
  Does **not** call `verify_model_candidates` itself (that requires a live
  `httpx.Client`/API key) — takes the already-verified `model_candidates`
  list as a parameter, so the function itself has no network dependency and
  is fully unit-testable. The preflight call is a separate step the caller
  (Plan 4's matrix runner, or this plan's own CLI/script entry point — task
  breakdown decides) performs first and passes the surviving list in.

---

## 5. What Plan 4 inherits from this plan

- `generate_agent_population(seed, model_candidates) -> list[BaseAgent]` —
  call once per run (or once per seed, per the master spec's 5-seeds-per-cell
  matrix) to get a reproducible 100-agent population.
- `verify_model_candidates(candidate_ids, client) -> (available,
  unavailable)` — call once before the full run matrix starts, report
  `unavailable` to the user, feed `available` into `generate_agent_population`.
- The loss-driven `a`-adaptation mechanism (§1) — entirely Plan 4's to design
  and implement, using `BaseAgent.cara_coefficient` as the nominal starting
  point and `AgentStateRecord.cara_coefficient` as the per-day persisted
  value.
- `AgentRecord`'s new `currency_zone`/`assigned_model`/`cara_coefficient`
  columns are populated once, at population-construction/persistence time
  (Plan 4 writes one `AgentRecord` per agent when a run starts) — this plan
  does not itself add a repository call site, since no run-orchestration code
  exists yet to call it from.
