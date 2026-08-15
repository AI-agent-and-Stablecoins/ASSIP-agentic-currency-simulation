# Runner Wiring — Design Spec (Pivot, final sub-project)

## 0. Why this spec exists

Continuing the research-methodology pivot from `New info.pdf`. Sub-projects A (hypothesis-sandbox mechanism), B (equilibrium-holdings measurement), and C (equivalence framework, incorporating D's elicitation primitive) are built and merged. E (the old econometrics engine) is archived to `src/legacy/econometrics/`. This spec covers the last deferred piece: wiring the hypothesis-sandbox sims into `run_matrix`'s persisted/checkpointed batch machinery, so a hypothesis-sim run persists to the database the same way every other cell in this codebase does, instead of only ever being built by hand inside a test (`tests/test_hypothesis_sim_integration.py`, and every `*_env()` helper in `tests/test_equilibrium_holdings.py`/`tests/test_equivalence_framework.py`).

Three design forks were resolved by explicit user decision before writing this spec (2026-08-14):

- **Runner shape**: a new, parallel runner function (`run_hypothesis_matrix`), not an extension of `run_matrix`/`_CellSpec`. The two diverge too deeply (different population generator, real vs. synthetic currencies, no chain-pin/event-shock concept in the existing cell spec, a mandatory `use_llm=True` with no meaningful deterministic mode) to unify without destabilizing the existing, heavily-tested 13-cell path. The new runner reuses `persist_full_timestep` and the day-level checkpoint pattern rather than duplicating them.
- **Analysis persistence**: new database tables. `holdings_by_cohort`/`cohort_indifference_points`' own design specs deferred this deliberately; this is the moment that gap gets filled, since nothing in `database/models.py` today can hold either shape.
- **Elicitation-phase crash recovery**: accept a full restart of the post-run search/elicitation phase on crash, rather than building a second, independent checkpointing concept for it. Matches the user's standing instruction not to weigh compute/resource-cost tradeoffs.

## 1. Scope: one new module, `src/simulation/hypothesis_matrix_runner.py`

`run_hypothesis_matrix` drives `build_hypothesis_cell_specs()`'s 24 `HypothesisCellSpec`s × 3 utility functions (`HYPOTHESIS_UTILITY_TYPES` = crra/cara/epstein_zin_proxy) × `seeds` — 72 (cell × utility-type) combinations per seed. It does not modify `run_matrix`/`_CellSpec`/the 13-cell path's behavior at all, except for the mechanical, behavior-preserving refactor in §3.

## 2. Cell/run identity

`HypothesisCellSpec` (`src/economy/hypothesis_scenarios.py`) has no analogue of `_CellSpec.key` today — `build_hypothesis_cell_specs()`'s 24 specs are distinguished only by their field values. Add a computed property (a small, additive change to this already-merged dataclass — no stored field, no serialization change):

```python
@dataclass(frozen=True)
class HypothesisCellSpec:
    ...
    @property
    def key(self) -> str:
        parts = [self.hypothesis]
        if self.cross_border:
            parts.append("cb")
        if self.event_shock is not None:
            parts.append(self.event_shock)
        return "_".join(parts)
```

This gives each of the 24 specs a unique key (`"H1"`, `"H1_cb"`, `"H1_depeg_event"`, `"H1_bank_failure"`, ...), mirroring `_CellSpec.key` exactly.

`run_id` scheme, adding the missing utility-type axis `_CellSpec`/`run_matrix` has no equivalent of:

```
{matrix_run_id}-{spec.key}-{utility_type}-seed{seed}
```

e.g. `hyp-2026-08-20-H3-crra-seed0`.

## 3. Shared checkpoint infrastructure: extract, don't duplicate

`_CellSeedCheckpoint`, `_checkpoint_path`, `_save_checkpoint`, `_load_checkpoint`, `_delete_checkpoint` (`src/simulation/matrix_runner.py` lines ~454-495) are already generic — keyed only by `run_id`, no `_CellSpec`-specific logic. Move them, unchanged, to a new module `src/simulation/checkpointing.py` (dropping the leading underscore since they're now a shared public surface: `CellSeedCheckpoint`, `checkpoint_path`, `save_checkpoint`, `load_checkpoint`, `delete_checkpoint`). `matrix_runner.py` imports them from there instead of defining them locally — a pure, behavior-preserving refactor (every existing call site's arguments are unchanged; only the import source and the four/five names' underscore prefix change). `hypothesis_matrix_runner.py` imports the same shared names.

**Verification**: `matrix_runner.py`'s full existing test suite must pass unchanged after this extraction, proving it's truly behavior-preserving before any new hypothesis-sim code is added on top.

## 4. Environment/population construction, per cell/seed/utility_type

Mirrors `run_matrix`'s no-checkpoint branch shape (population → `build_from_population` → wallet seeding → optional marketplace swap), with every piece specific to hypothesis-sims:

```python
population = generate_hypothesis_population(seed, available_models, utility_type)
cell_scenario = _scenario_for(spec, base_scenario)  # see below
env = Environment.build_from_population(
    "master_simulation", population, currencies=restricted_currencies(spec), scenario=cell_scenario
)
env.currency_chain_pins = spec.chain_pins or {}
seed_restricted_wallets(env.agents, restricted_currencies(spec), real_currency_universe, env.macro_state.peg_reference_rates)
if spec.cross_border:
    env.marketplace = CrossZoneMarketplace(env.agents)  # reused as-is from matrix_runner.py
```

`env.currency_chain_pins = spec.chain_pins or {}` is the missing wiring the research surfaced: sub-project A added `Environment.currency_chain_pins` and threaded it through `generate_candidates`, but nothing anywhere ever sets it from a runner. Without this line, H5/H8/H10/H11's chain-pinning (the whole point of those four hypotheses — forcing the gas-fee tradeoff) silently never takes effect in a real run.

`restricted_currencies(spec)` resolves `HYPOTHESIS_CURRENCIES[spec.hypothesis]` from the real currency universe (already exactly how every hypothesis-sim test builds its `restricted` dict today — `tests/test_equivalence_framework.py`'s `_hypothesis_env` is the canonical pattern this generalizes).

**Scenario construction** (`_scenario_for`, new, in `src/economy/hypothesis_scenarios.py`):
- `spec.event_shock is None` (16 of 24 specs: 11 baseline + 5 cross-border): use `base_scenario` (the unmodified `master_simulation` `ScenarioConfig`) verbatim — same macro backdrop as the master cell, no hypothesis-specific shocks. Currency-universe restriction and (for cross-border specs) marketplace behavior are the only things that vary, consistent with sub-project A's original intent.
- `spec.event_shock is not None` (8 event-based specs): a new `build_hypothesis_event_scenario(spec, base_scenario) -> ScenarioConfig`, mirroring `build_sandbox_scenario`'s shape (`src/economy/sandbox_scenarios.py`) but simpler — one `ShockEvent(day=<fixed day>, type=ShockType(spec.event_shock), magnitude=<TBD>, target_currency=spec.event_target_currency)` appended to `base_scenario`'s 5 macro-level shocks (its currency-targeted shocks are dropped, same reasoning as `build_sandbox_scenario`: they target real-universe symbols that may not even be in this hypothesis's restricted currency set). Shock day and magnitude picked to match existing conventions (`build_sandbox_scenario`'s `_DEPEG_MAGNITUDE = 0.15` precedent for `DEPEG_EVENT`; a comparable value for `BANK_FAILURE` from `master_simulation.yaml`'s own existing entry) — exact values are an implementation-time detail, not a design fork.

## 5. `use_llm` is unconditionally `True` — no `dry_run`/`exercise_llm_path` duality

Per sub-project A's binding decision (no deterministic mode is meaningful for a hypothesis-sim — CRRA/CARA/EpsteinZinProxy are monotone transforms of one wealth scalar under the deterministic path), `run_hypothesis_matrix` has no `dry_run`/`exercise_llm_path`/`mock_llm_decision` parameters at all. Instead it takes a required `openrouter_client: httpx.Client` (no default, no `None`) — the caller supplies either a real client or a mocked one (e.g. `tests/llm_test_helpers.py`'s `mock_openrouter_client`/`mock_switch_threshold_client`) directly, exactly as `cohort_indifference_points`/`call_model_for_switch` already do. `polygon_client` stays optional (live-price fetch degrades gracefully already, per existing convention). There is no need to replicate `run_matrix`'s more elaborate default-safe gate: requiring an explicit client with no default already forces the caller to decide.

## 6. Post-run analysis phase

Immediately after a cell/seed/utility_type's day-loop completes (env still in memory; the day-loop's own checkpoint is deliberately kept alive through this phase, deleted only once it too commits successfully — see the implementation note below):

- **H1 only** (`spec.hypothesis == "H1"`, all 4 of its specs: baseline, cross-border, 2 event variants): call `holdings_by_cohort(env)`, persist one `CohortHoldingsRecord` row per `(cohort, currency)` pair.
- **H2-H11** (the other 20 specs): for each `EquivalenceComparison` in `EQUIVALENCE_COMPARISONS[spec.hypothesis]` (1 row for H3-H11, 2 for H2), call `cohort_indifference_points(env, comparison, client)` (see the model-resolution fix below for why this no longer takes a `model_id`), persist one `IndifferencePointRecord` row per cohort.

**Implementation note (added after the Task 6 whole-plan review found a gap in the "accept restart" decision below)**: the analysis phase itself still has no checkpointing of its own, exactly as decided — but the day-loop's *existing* checkpoint is kept alive until the analysis phase's own `session.commit()` succeeds, rather than being deleted the moment the day loop finishes. A crash during analysis therefore resumes into an already-exhausted day range (a no-op) and simply retries the analysis phase, without needing to redo the day loop at all — strictly better than the "full restart" this section originally called for, at no extra checkpointing-concept cost. See `src/simulation/hypothesis_matrix_runner.py`'s module docstring for the full mechanism, including a related, narrower gap it also documents (a crash between one day's persistence commit and that day's checkpoint write, shared with `run_matrix`'s identical pre-existing pattern).

**Bug found during this investigation, fixed as part of this wiring**: `cohort_indifference_points`'s `model_id` parameter is a single fixed string used for every agent's `call_model_for_switch` call, but `generate_hypothesis_population` assigns each agent its own model round-robin from `model_candidates` (`shuffled_models[slot_index % len(shuffled_models)]`, `src/agents/population.py` line 123) — exactly like every other population generator in this codebase. A multi-model hypothesis-sim run would silently elicit every agent's switch decision from the wrong model. Fix: drop the `model_id` parameter from `cohort_indifference_points`/`_agent_indifference_point`; read `agent.assigned_model` per agent instead (already available via `agent.build_llm_context().assigned_model`), matching how every other per-agent LLM call in this codebase already resolves its model. This is a one-parameter, backward-incompatible signature change to already-merged sub-project C code — `tests/test_equivalence_framework.py`'s calls update accordingly.

## 7. New database tables

`database/models.py`:

```python
class CohortHoldingsRecord(Base):
    __tablename__ = "cohort_holdings"

    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"), primary_key=True)
    risk_aversion_cohort: Mapped[float] = mapped_column(Float, primary_key=True)
    currency_symbol: Mapped[str] = mapped_column(String, primary_key=True)
    pct_of_wealth: Mapped[float] = mapped_column(Float)


class IndifferencePointRecord(Base):
    __tablename__ = "indifference_points"

    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"), primary_key=True)
    hypothesis: Mapped[str] = mapped_column(String, primary_key=True)
    fixed_currency: Mapped[str] = mapped_column(String)
    varied_currency: Mapped[str] = mapped_column(String, primary_key=True)
    varied_field: Mapped[str] = mapped_column(String)
    risk_aversion_cohort: Mapped[float] = mapped_column(Float, primary_key=True)
    compensation: Mapped[float] = mapped_column(Float)
```

(`varied_currency`, not `fixed_currency`, is part of `IndifferencePointRecord`'s key alongside `hypothesis`/`risk_aversion_cohort` — H2 is the one hypothesis with two comparisons sharing a `hypothesis` value, and they're distinguished by `varied_currency`, EURC vs. PAXG.)

`database/repository.py`: `CohortHoldingsLogEntry`/`CohortHoldingsRepository` and `IndifferencePointLogEntry`/`IndifferencePointRepository`, each mirroring `SimulationRunLogEntry`/`SimulationRunRepository`'s existing minimal shape (a plain pydantic entry, a repository whose `.record()` does one `session.add(...)`).

## 8. `run_hypothesis_matrix` signature

```python
def run_hypothesis_matrix(
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    openrouter_client: httpx.Client,
    session: Session,
    matrix_run_id: str,
    polygon_client: httpx.Client | None = None,
    utility_types: list[str] | None = None,  # None -> all of HYPOTHESIS_UTILITY_TYPES
    hypotheses: list[str] | None = None,     # None -> all 24 specs; filters build_hypothesis_cell_specs() by .hypothesis
    progress_callback: Callable[[str, int, str, int], None] | None = None,  # (cell_key, seed, utility_type, day)
    checkpoint_dir: Path | None = None,
    llm_max_workers: int = 1,
) -> tuple[list[HypothesisCellResult], list[tuple[str, int, str, Exception]]]:
```

`HypothesisCellResult` mirrors `MatrixCellResult`'s shape (`run_id`, `cell_key`, `seed`, `utility_type`, `hypothesis`, `is_cross_border`, `num_days_completed`, `total_transactions`, `total_llm_decisions`) plus whichever of `holdings_by_cohort`'s or `cohort_indifference_points`' return value this cell produced, for immediate caller inspection alongside what's now durably persisted. `failures` gains a `utility_type` element (`(cell_key, seed, utility_type, exception)`) versus `MatrixCellResult`'s tuple shape, since this runner's identity has one more axis.

## 9. Testing

- `HypothesisCellSpec.key` is unique across all 24 specs from `build_hypothesis_cell_specs()`.
- The checkpointing extraction (§3) is proven behavior-preserving: `matrix_runner.py`'s existing checkpoint/resume tests pass unchanged after the refactor.
- `env.currency_chain_pins` is actually populated after `run_hypothesis_matrix` builds a chain-pinned cell (H5/H8/H10/H11) — a regression test for the exact gap this spec's §4 closes.
- `build_hypothesis_event_scenario` produces a `ScenarioConfig` with exactly one currency-targeted shock (the event one), the 5 macro-level shocks intact, and no leftover real-universe-targeted shocks from `base_scenario`.
- One real end-to-end test per major path (H1 cell → `CohortHoldingsRecord` rows persisted correctly; H3 cell → `IndifferencePointRecord` rows persisted correctly), using `mock_openrouter_client`/`mock_switch_threshold_client` against an in-memory sqlite session — matching every prior sub-project's end-to-end test convention.
- Resume: a hypothesis-sim cell/seed/utility_type interrupted mid-day-loop resumes correctly from its checkpoint (reusing `matrix_runner.py`'s own resume tests' pattern against the shared `checkpointing.py` helpers).
- The elicitation-phase-crash-means-full-restart behavior (§0's third decision) is documented, not specially tested — there's no new mechanism to verify beyond "a crash during the post-run analysis phase leaves no checkpoint and the next `run_hypothesis_matrix` call re-runs that cell/seed/utility_type's day-loop from scratch, exactly like any run_id with no checkpoint and no `SimulationRunRecord` row" (already covered by the resume test above).

## 10. Out of scope (this spec)

- Actually executing all 72 cell/utility-type combinations at real scale (365 days × real LLM calls × real spend) — this spec builds the mechanism; running it is a separate, later decision.
- `distributed_matrix_runner`-style multi-process partitioning for the hypothesis matrix (`run_matrix`'s `cell_keys` hook enables this today for the 13-cell path; an equivalent for `run_hypothesis_matrix` is a natural follow-on, not required here).
- Any dashboard/reporting UI surfacing `CohortHoldingsRecord`/`IndifferencePointRecord` data — this spec only makes the data durable and queryable.
- Renaming or otherwise touching `run_matrix`'s own `dry_run`/`exercise_llm_path` parameters or behavior.
