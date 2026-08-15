# Runner Wiring Implementation Plan (Pivot, final sub-project)

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Use subagent-driven execution with review checkpoints per task, plus a final whole-plan review once all tasks are done.

**Goal:** Build `run_hypothesis_matrix` per `docs/superpowers/specs/2026-08-14-runner-wiring-design.md`: a new, parallel runner that drives `build_hypothesis_cell_specs()`'s 24 cells × 3 utility functions × seeds through a real day-loop, persists every day via the existing `persist_full_timestep`, then runs the appropriate post-run analysis (`holdings_by_cohort` for H1, `cohort_indifference_points` for H2-H11) and persists those results to two new tables.

**Architecture:** Six tasks, roughly in dependency order:
1. Extract shared checkpoint helpers to `src/simulation/checkpointing.py` (mechanical refactor of `matrix_runner.py`, no behavior change).
2. `HypothesisCellSpec.key` property.
3. `build_hypothesis_event_scenario` + a `_scenario_for` dispatcher.
4. New DB tables + repositories (`CohortHoldingsRecord`, `IndifferencePointRecord`).
5. Fix `cohort_indifference_points`'s per-agent model resolution (spec §6's bug fix).
6. `run_hypothesis_matrix` itself, wiring everything together, plus end-to-end tests.

**Tech Stack:** Python 3.12, pydantic 2.x, SQLAlchemy 2.x, pytest, httpx (mocked in tests). No new dependencies.

## Global Constraints

- Follow the spec exactly: `docs/superpowers/specs/2026-08-14-runner-wiring-design.md`.
- Every test run during implementation must be targeted (the specific new/changed test files plus their nearest existing neighbors), never the full ~600-test suite — cap at ~5 minutes.
- `DATABASE_URL="sqlite:///./assip.db"` must prefix every pytest invocation that touches anything importing `database.repository`/`database.models` (existing `.env` has a typo'd `DATABASE_URL`; do not edit `.env`).
- No comments beyond what the codebase already uses at each touched call site; new tests follow existing style (plain `assert`, no docstrings on trivial tests).
- Task 1 must be verified behavior-preserving (existing `matrix_runner.py` tests pass unchanged) before any task that builds on it.

---

### Task 1: Extract checkpoint helpers to `src/simulation/checkpointing.py`

**Files:**
- Create: `src/simulation/checkpointing.py`
- Modify: `src/simulation/matrix_runner.py`

**Steps:**

- [ ] **Step 1**: Create `src/simulation/checkpointing.py` containing, moved verbatim from `src/simulation/matrix_runner.py` (only renamed to drop the leading underscore, since this is now a shared public module): `CellSeedCheckpoint` (was `_CellSeedCheckpoint`), `checkpoint_path` (was `_checkpoint_path`), `save_checkpoint` (was `_save_checkpoint`), `load_checkpoint` (was `_load_checkpoint`), `delete_checkpoint` (was `_delete_checkpoint`). Needs `pickle`, `random` (for `CellSeedCheckpoint.rng`'s type), `pathlib.Path`, `pydantic.BaseModel`/`ConfigDict`, and `src.simulation.environment.Environment`, `src.simulation.timestep.TimestepResult` imports (same as `matrix_runner.py` already has for these).

- [ ] **Step 2**: In `matrix_runner.py`: remove the five moved definitions; add `from src.simulation.checkpointing import CellSeedCheckpoint, checkpoint_path, save_checkpoint, load_checkpoint, delete_checkpoint`; update every call site (`_load_checkpoint(...)` → `load_checkpoint(...)`, `_CellSeedCheckpoint(...)` → `CellSeedCheckpoint(...)`, etc. — mechanical rename only, no logic change).

- [ ] **Step 3**: Run the existing matrix-runner test suite to prove this is behavior-preserving.

Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_matrix_runner.py -q`
Expected: PASS, identical to pre-refactor (same count, same names) — if this file doesn't exist under that exact name, find it via `grep -rl "run_matrix" tests/*.py | grep -v hypothesis`.

- [ ] **Step 4: Commit**

```bash
git add src/simulation/checkpointing.py src/simulation/matrix_runner.py
git commit -m "refactor: extract matrix_runner's checkpoint helpers to a shared module"
```

---

### Task 2: `HypothesisCellSpec.key`

**Files:**
- Modify: `src/economy/hypothesis_scenarios.py`
- Modify: `tests/test_hypothesis_scenarios.py`

**Steps:**

- [ ] **Step 1**: Add to `HypothesisCellSpec` (per the spec's §2 code block) a `key` property: `hypothesis`, plus `"_cb"` if `cross_border`, plus `"_" + event_shock` if `event_shock is not None`, joined.

- [ ] **Step 2**: Add a test asserting `len({spec.key for spec in build_hypothesis_cell_specs()}) == len(build_hypothesis_cell_specs()) == 24` (uniqueness), plus a couple of direct spot-checks (e.g. the H1 baseline spec's key is `"H1"`, the H1 cross-border spec's key is `"H1_cb"`, an H1 event spec's key is `"H1_depeg_event"` or `"H1_bank_failure"`).

- [ ] **Step 3**: Run tests.

Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_hypothesis_scenarios.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/economy/hypothesis_scenarios.py tests/test_hypothesis_scenarios.py
git commit -m "feat: add HypothesisCellSpec.key for unique per-cell run identity"
```

---

### Task 3: `build_hypothesis_event_scenario` + `_scenario_for`

**Files:**
- Modify: `src/economy/hypothesis_scenarios.py`
- Modify: `tests/test_hypothesis_scenarios.py`

**Interfaces:**
- Consumes: `ScenarioConfig`, `ShockEvent`, `ShockType`, `load_scenario` (`src/economy/shocks.py`, already exist).
- Produces: `build_hypothesis_event_scenario(spec: HypothesisCellSpec, base_scenario: ScenarioConfig) -> ScenarioConfig` and `scenario_for(spec: HypothesisCellSpec, base_scenario: ScenarioConfig) -> ScenarioConfig`.

**Steps:**

- [ ] **Step 1**: Implement `build_hypothesis_event_scenario`, mirroring `src/economy/sandbox_scenarios.py`'s `build_sandbox_scenario` shape: keep `base_scenario`'s macro-level shocks only (`[s for s in base_scenario.shocks if s.target_currency is None]`), append one `ShockEvent(day=_EVENT_DAY, type=ShockType(spec.event_shock), magnitude=_EVENT_MAGNITUDE[spec.event_shock], target_currency=spec.event_target_currency)`, return `base_scenario.model_copy(update={"name": f"{spec.key}_event", "shocks": [...]}, deep=True)`.
  - `_EVENT_DAY = 200` (module constant — clear of every `configs/scenarios/master_simulation.yaml` shock day, all of which are ≤190 or ≥210; confirm this against that file before picking the exact value).
  - `_EVENT_MAGNITUDE = {ShockType.DEPEG_EVENT.value: 0.15, ShockType.BANK_FAILURE.value: 0.25}` (matching `master_simulation.yaml`'s own existing magnitude for each shock type).
  - Note for the implementer: `BANK_FAILURE` with a `target_currency` set is meaningful, not a no-op — confirmed by reading `src/economy/trust.py`'s `TrustLedger.update` (~line 98), which applies a currency-specific trust-severity effect to `shock.target_currency` for any shock type with a non-`None` target, in addition to `apply_shock`'s global `confidence_index` effect (which ignores `target_currency` for `BANK_FAILURE`). Both effects are already-existing machinery; this task only constructs the `ShockEvent`.

- [ ] **Step 2**: Implement `scenario_for(spec, base_scenario)`: return `base_scenario` unmodified if `spec.event_shock is None`, else `build_hypothesis_event_scenario(spec, base_scenario)`.

- [ ] **Step 3**: Tests: `build_hypothesis_event_scenario` on an H1 depeg-event spec produces a `ScenarioConfig` with exactly one shock whose `target_currency` is non-`None` (the event one, matching `spec.event_target_currency` and `spec.event_shock`), and every shock from `base_scenario` that had a `target_currency` is gone; the 5 (or however many) macro-level shocks are intact (same days/types/magnitudes as `base_scenario`'s). `scenario_for` returns `base_scenario` itself (identity or equal, implementer's choice matching existing conventions) for a baseline/cross-border spec, and the event-scenario result for an event spec.

- [ ] **Step 4**: Run tests.

Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_hypothesis_scenarios.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/economy/hypothesis_scenarios.py tests/test_hypothesis_scenarios.py
git commit -m "feat: add build_hypothesis_event_scenario for event-based hypothesis cells"
```

---

### Task 4: New DB tables + repositories

**Files:**
- Modify: `database/models.py`
- Modify: `database/repository.py`
- Create: `tests/test_cohort_holdings_repository.py`
- Create: `tests/test_indifference_point_repository.py`

**Steps:**

- [ ] **Step 1**: Add `CohortHoldingsRecord`/`IndifferencePointRecord` to `database/models.py` exactly per the spec's §7 code block (composite primary keys as specified; `ForeignKey("simulation_runs.run_id")` on `run_id` in both, matching `TimestepLogRecord`'s pattern).

- [ ] **Step 2**: Add `CohortHoldingsLogEntry`/`CohortHoldingsRepository` and `IndifferencePointLogEntry`/`IndifferencePointRepository` to `database/repository.py`, mirroring `SimulationRunLogEntry`/`SimulationRunRepository`'s exact minimal shape (plain pydantic entry with the same fields as the record minus any server-generated ones; repository `__init__(self, session)`, one `.record(entry)` method doing `self.session.add(RecordClass(**entry.model_dump()))`).

- [ ] **Step 3**: Tests (one file per table, matching this codebase's one-file-per-repository-concern convention): build an in-memory sqlite engine + `Base.metadata.create_all`, insert a `SimulationRunRecord` (required by the FK), record one entry via each new repository, commit, query it back, assert every field round-trips. Also assert a duplicate `(run_id, risk_aversion_cohort, currency_symbol)` (or the `IndifferencePointRecord` equivalent) raises `IntegrityError` on commit — matching how this codebase already tests composite-key uniqueness elsewhere (see `AgentRecord`'s docstring/tests for the pattern).

- [ ] **Step 4**: Run tests.

Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_cohort_holdings_repository.py tests/test_indifference_point_repository.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add database/models.py database/repository.py tests/test_cohort_holdings_repository.py tests/test_indifference_point_repository.py
git commit -m "feat: add CohortHoldingsRecord/IndifferencePointRecord persistence"
```

---

### Task 5: Per-agent model resolution in `cohort_indifference_points`

**Files:**
- Modify: `src/economy/equivalence_framework.py`
- Modify: `tests/test_equivalence_framework.py`

**Steps:**

- [ ] **Step 1**: Drop the `model_id: str` parameter from `cohort_indifference_points` and `_agent_indifference_point`. Inside `_agent_indifference_point`, resolve the model from the agent itself: `agent_context = agent.build_llm_context()` already exists — use `agent_context.assigned_model` as the `model_id` argument to `call_model_for_switch`. Raise a clear `ValueError` if `agent_context.assigned_model` is `None` (mirrors `run_timestep`'s existing `use_llm=True requires every agent to have an assigned_model` guard, `src/simulation/timestep.py` ~line 699) rather than passing `None` through to `call_model_for_switch` and getting an unrelated failure deeper in the call stack.

- [ ] **Step 2**: Update every `cohort_indifference_points(env, comparison, "vendor/model", client)` call in `tests/test_equivalence_framework.py` to drop the now-removed positional argument (`cohort_indifference_points(env, comparison, client)`). Since every hypothesis-sim test agent is built via `generate_hypothesis_population(seed, model_candidates, utility_type)` with a single-element `model_candidates` list in these tests, every agent's `assigned_model` is already `"vendor/model"` — no test fixture changes needed beyond the call signature.

- [ ] **Step 3**: Run tests.

Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_equivalence_framework.py tests/test_switch_elicitation.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/economy/equivalence_framework.py tests/test_equivalence_framework.py
git commit -m "fix: resolve each agent's own assigned_model in cohort_indifference_points"
```

---

### Task 6: `run_hypothesis_matrix`

**Files:**
- Create: `src/simulation/hypothesis_matrix_runner.py`
- Create: `tests/test_hypothesis_matrix_runner.py`

**Interfaces:**
- Consumes: everything listed in the spec's §4/§6/§8 — `build_hypothesis_cell_specs`, `HYPOTHESIS_CURRENCIES`, `HYPOTHESIS_UTILITY_TYPES`, `generate_hypothesis_population`, `seed_restricted_wallets`, `scenario_for`, `EQUIVALENCE_COMPARISONS`, `holdings_by_cohort`, `cohort_indifference_points`, `CrossZoneMarketplace` (from `matrix_runner.py`, reused as-is), `persist_full_timestep`, `SimulationRunLogEntry`/`SimulationRunRepository`, `CohortHoldingsLogEntry`/`CohortHoldingsRepository`, `IndifferencePointLogEntry`/`IndifferencePointRepository`, `checkpointing.py`'s shared helpers (Task 1), `compute_config_hash`/`compute_git_commit_hash`/`model_roster_summary_for` (`src/simulation/provenance.py`), `run_timestep`.
- Produces: `HypothesisCellResult` (pydantic `BaseModel`, per spec §8), `run_hypothesis_matrix(...)` per the spec's §8 signature exactly.

**Steps:**

- [ ] **Step 1**: Implement `HypothesisCellResult`.

- [ ] **Step 2**: Implement `run_hypothesis_matrix`'s outer loop: resolve `available_models` (reuse `matrix_runner._resolve_available_models`, or inline the same one-line preflight-verify-if-client-given logic — implementer's call, but don't duplicate the whole function body if it's cheap to import), compute `git_commit_hash`/`prompt_version_hash`/`config_hash` once, load `real_currency_universe` and `base_scenario = load_scenario("master_simulation")` once, then `for utility_type in (utility_types or HYPOTHESIS_UTILITY_TYPES): for spec in (all specs filtered by hypotheses if given): for seed in seeds:` — three nested loops (the extra `utility_type` axis versus `run_matrix`'s two).

- [ ] **Step 3**: Per (utility_type, spec, seed): build `run_id = f"{matrix_run_id}-{spec.key}-{utility_type}-seed{seed}"`; checkpoint-or-fresh-build branch mirroring `run_matrix`'s exact structure (§0/§4 of the spec) but using `generate_hypothesis_population(seed, available_models, utility_type)`, `restricted_currencies = {s: real_currency_universe[s] for s in HYPOTHESIS_CURRENCIES[spec.hypothesis]}`, `scenario_for(spec, base_scenario)`, and unconditionally `env.currency_chain_pins = spec.chain_pins or {}` and `seed_restricted_wallets(...)` (unlike `run_matrix`, which only seeds wallets when `spec.currencies is not None` — every hypothesis-sim cell always restricts currencies, so this is unconditional here). `if spec.cross_border: env.marketplace = CrossZoneMarketplace(env.agents)`.

- [ ] **Step 4**: Day loop: identical shape to `run_matrix`'s (`run_timestep(..., use_llm=True, ...)`, `persist_full_timestep`, progress_callback with the extra `utility_type` argument, checkpoint save per day via `checkpointing.save_checkpoint`).

- [ ] **Step 5**: On day-loop completion: `checkpointing.delete_checkpoint(...)`, then the post-run analysis phase (spec §6): if `spec.hypothesis == "H1"`, call `holdings_by_cohort(env)` and persist one `CohortHoldingsLogEntry` per `(cohort, currency)`; else, for each `comparison in EQUIVALENCE_COMPARISONS[spec.hypothesis]`, call `cohort_indifference_points(env, comparison, client=openrouter_client)` (per Task 5's new signature) and persist one `IndifferencePointLogEntry` per cohort, tagged with `comparison.hypothesis`/`comparison.fixed_currency`/`comparison.varied_currency`/`comparison.varied_field`. `session.commit()` after this phase's rows are added.

- [ ] **Step 6**: Build `HypothesisCellResult`, append to `results`; on any exception in the try block (day loop OR analysis phase), `session.rollback()` and append `(spec.key, seed, utility_type, exc)` to `failures`, matching `run_matrix`'s per-cell/seed isolation.

- [ ] **Step 7**: Tests, end-to-end against an in-memory sqlite session with `mock_openrouter_client`/`mock_switch_threshold_client` (`tests/llm_test_helpers.py`):
  - An H1 cell (small `num_days`, e.g. 5-10, one seed, one utility_type) run through `run_hypothesis_matrix` produces `CohortHoldingsRecord` rows queryable from the session, matching what a direct `holdings_by_cohort(env)` call against the same (checkpointed-then-reloaded, or freshly rebuilt with the same seed) environment would produce.
  - An H3 cell similarly produces `IndifferencePointRecord` rows.
  - `env.currency_chain_pins` is non-empty and correct for an H5/H8/H10/H11 cell (the exact regression test the spec's §9 calls for) — assert it directly on the `env` object mid-construction, or via a small seam that exposes it (implementer's call on how to reach in and check without over-exposing internals — a unit test directly on the population/env-construction helper extracted from Step 3 is fine if that's cleaner than asserting through the full runner).
  - A cell/seed/utility_type interrupted after N days (via `checkpoint_dir`, killing the loop early through test scaffolding — mirror however `tests/test_matrix_runner.py`'s existing resume test triggers this) resumes correctly on a second `run_hypothesis_matrix` call with the same `matrix_run_id`.
  - `SimulationRunRecord` rows use `run_id`s containing the utility_type component (spot-check one).
  - Keep every test's `num_days` small (5-10) and cell/seed/utility_type counts minimal — this suite must run in well under 5 minutes.

- [ ] **Step 8**: Run tests.

Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_hypothesis_matrix_runner.py -q`
Expected: PASS.

- [ ] **Step 9**: Run the full targeted regression set for everything this plan touched.

Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_hypothesis_matrix_runner.py tests/test_hypothesis_scenarios.py tests/test_cohort_holdings_repository.py tests/test_indifference_point_repository.py tests/test_equivalence_framework.py tests/test_switch_elicitation.py tests/test_equilibrium_holdings.py tests/test_matrix_runner.py tests/test_wallet_seeding.py tests/test_hypothesis_sim_integration.py -q`
Expected: PASS, all green.

- [ ] **Step 10: Commit**

```bash
git add src/simulation/hypothesis_matrix_runner.py tests/test_hypothesis_matrix_runner.py
git commit -m "feat: wire hypothesis-sandbox sims into a persisted, checkpointed runner"
```

---

### Final: whole-plan review

After all 6 tasks are committed, dispatch a whole-plan review (opus) covering the full diff from before Task 1 to after Task 6, per this session's established pattern (sub-projects A and C both received one). Report findings to the user before considering this sub-project done.
