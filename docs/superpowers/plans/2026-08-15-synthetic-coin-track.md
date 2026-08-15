# Synthetic-Coin Track Implementation Plan

> **For agentic workers:** Use subagent-driven execution with review checkpoints per task, plus a final whole-plan review once all tasks are done.

**Goal:** Build a second, parallel hypothesis-sandbox track using fully-controlled synthetic coins per `docs/superpowers/specs/2026-08-15-synthetic-coin-track-design.md`, without touching anything in the already-shipped real-coin track (sub-projects A/B/C, runner-wiring).

**Architecture:** Six tasks:
1. `bid_ask_spread` optional field on `CurrencyConfig`.
2. Synthetic gas-fee chains (programmatic, not YAML-loaded — never touches `load_chain_universe()`'s real universe).
3. `src/currencies/synthetic_hypothesis_currencies.py` — the 5-dimension grid + per-hypothesis coin builder.
4. `src/economy/synthetic_hypothesis_scenarios.py` — cell specs for the 11 synthetic hypotheses (baseline-only, per spec §8).
5. Discrete-level switch search (reuses `IndifferencePointRecord`/reporting unchanged — only the search algorithm is new).
6. Runner wiring: `track` parameter on `run_hypothesis_matrix`, dispatch logic, reporting generalization.

**Tech Stack:** Python 3.12, pydantic 2.x, pytest, httpx (mocked in tests). No new dependencies.

## Global Constraints

- Follow the spec exactly: `docs/superpowers/specs/2026-08-15-synthetic-coin-track-design.md`.
- **Never modify**: `src/economy/hypothesis_scenarios.py`'s existing real-coin functions/constants, `src/economy/equivalence_framework.py`'s existing functions, `configs/currencies/*.yaml`, `configs/blockchains/*.yaml`, or any already-shipped test for the real-coin track. Every new piece is additive, in new files (or new, optional, non-breaking fields on shared base classes).
- Every test run during implementation must be targeted, never the full suite — cap at ~5 minutes.
- `DATABASE_URL="sqlite:///./assip.db"` must prefix every pytest invocation touching `database.*`.

---

### Task 1: `bid_ask_spread` field

**Files:** Modify `src/currencies/currency.py`; Modify `tests/test_currency_config.py` (or nearest existing currency-config test file — find it first).

- [ ] Add `bid_ask_spread: float | None = Field(default=None, ge=0.0)` to `CurrencyConfig`. Optional, defaults to `None` so every existing real `configs/currencies/*.yaml` file (none of which set this field) still loads unchanged.
- [ ] Test: `load_currency_universe()` still loads every real currency with `bid_ask_spread is None`; a `StablecoinConfig` constructed with `bid_ask_spread=0.0001` round-trips correctly.
- [ ] Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_currency_config.py -q` (or wherever this lands) plus a full `load_currency_universe()` smoke check.
- [ ] Commit: `feat: add optional bid_ask_spread field to CurrencyConfig`

---

### Task 2: Synthetic gas-fee chains

**Files:** Create `src/currencies/synthetic_hypothesis_currencies.py` (this task just adds the chain constants at the top; Task 3 adds the rest to the same file — one implementer can do both, or split; your call).

- [ ] Define, as plain Python (NOT YAML, NOT touching `configs/blockchains/`): `SYNTHETIC_GAS_LEVELS = (0.01, 0.05, 0.10)` and `SYNTHETIC_CHAINS: dict[str, ChainConfig]` = `{"synthetic_gas_low": ChainConfig(name="synthetic_gas_low", throughput=1000.0, gas_fee=0.01, finality_seconds=5.0), "synthetic_gas_mid": ChainConfig(..., gas_fee=0.05, ...), "synthetic_gas_high": ChainConfig(..., gas_fee=0.10, ...)}` (pick sensible throughput/finality placeholders — not economically meaningful, just valid `ChainConfig` instances).
- [ ] Test: each of the 3 configs validates, `gas_fee` values are exactly 0.01/0.05/0.10, and `load_chain_universe()`'s real result is completely unaffected (still exactly `{ethereum, arbitrum, base, solana}`, proving these synthetic chains never leak into the real universe).
- [ ] Run targeted tests.
- [ ] Commit (can combine with Task 3's commit if built together).

---

### Task 3: The 5-dimension grid + per-hypothesis coin builder

**Files:** Modify/extend `src/currencies/synthetic_hypothesis_currencies.py`; Create `tests/test_synthetic_hypothesis_currencies.py`.

**Interfaces:**
```python
GOVERNANCE_LEVELS = (0.0, 1.0)  # low, high
MEDIUM_LEVELS = ("USD", "EUR", "XAU")
BID_ASK_SPREAD_LEVELS = (0.0001, 0.0005, 0.0010)  # 0.01% / 0.05% / 0.10%
VOLATILITY_LEVELS = (0.001, 0.004, 0.008)  # 0.1% / 0.4% / 0.8% -- maps to peg_error
GAS_FEE_LEVELS = SYNTHETIC_GAS_LEVELS  # from Task 2

SYNTHETIC_DIMENSION_PAIRS: dict[str, tuple[str, str] | tuple[str]] = {
    "H1": ("medium",),
    "H2": ("governance", "medium"),
    "H3": ("governance", "liquidity"),
    "H4": ("governance", "volatility"),
    "H5": ("governance", "gas_fee"),
    "H6": ("medium", "liquidity"),
    "H7": ("medium", "volatility"),
    "H8": ("medium", "gas_fee"),
    "H9": ("liquidity", "volatility"),
    "H10": ("liquidity", "gas_fee"),
    "H11": ("volatility", "gas_fee"),
}

# Neutral fixed value for a dimension when it's NOT one of a hypothesis's tested pair.
NEUTRAL_FIXED_VALUES = {"governance": 1.0, "medium": "USD", "liquidity": 0.0005, "volatility": 0.004, "gas_fee": 0.05}

def build_synthetic_hypothesis_currencies(hypothesis: str) -> tuple[dict[str, CurrencyConfig], dict[str, str]]:
    """Returns (currencies, chain_pins) for `hypothesis`'s full cross-product
    grid (per spec §3's table), holding every untested dimension at
    NEUTRAL_FIXED_VALUES. chain_pins maps each currency symbol to its
    assigned chain name ONLY for currencies whose gas_fee level is fixed at
    a specific value by this call (every grid coin gets a chain pin -- gas
    fee is always determined, whether tested or held neutral)."""
```

- [ ] Implement `build_synthetic_hypothesis_currencies`: for each hypothesis, enumerate the cross-product of its tested dimensions' level-tuples (or, for H1, just the 3 medium levels alone); for each combination, construct a `StablecoinConfig` (peg USD/EUR) or `GoldBackedConfig` (peg XAU) with a deterministic symbol (e.g. `f"SYN_{hypothesis}_{index:02d}"` or a descriptive one like `f"SYN_{hypothesis}_HIGOV_0.05SPR"` — implementer's call, but must be deterministic and unique within a hypothesis's grid), governance_score/genius_compliant/peg_error/bid_ask_spread set per the combination (or `NEUTRAL_FIXED_VALUES` for untested dimensions), `issuer_risk` held at one shared constant (e.g. `0.10`) across every coin in every hypothesis (matching `sandbox_currencies.py`'s "hold everything else constant" convention), `redemption_mechanism`/`custodian`/`gold_reserve_oz` filled with placeholder values matching `sandbox_currencies.py`'s `_SYNTHETIC_REDEMPTION`/`_SYNTHETIC_CUSTODIAN` style. Every coin's `chain_pins` entry points to whichever of Task 2's 3 synthetic chains matches its gas-fee level.
- [ ] Tests: H1 has exactly 3 currencies (one per medium level); H3 has exactly 6 (2 governance × 3 liquidity), each with the right `governance_score`/`bid_ask_spread` combination and all sharing the same `peg`/`peg_error`/`issuer_risk` (the untested dimensions truly held constant); H6 has exactly 9; every coin across every hypothesis has a `chain_pins` entry resolving to a real `SYNTHETIC_CHAINS` key with the correct `gas_fee`; symbols are unique within each hypothesis's grid.
- [ ] Run targeted tests.
- [ ] Commit: `feat: add synthetic 5-dimension currency grid for the synthetic-coin track`

---

### Task 4: Synthetic cell specs

**Files:** Create `src/economy/synthetic_hypothesis_scenarios.py`; Create `tests/test_synthetic_hypothesis_scenarios.py`.

**Interfaces:**
```python
@dataclass(frozen=True)
class SyntheticHypothesisCellSpec:
    hypothesis: str
    currencies: dict[str, CurrencyConfig]
    chain_pins: dict[str, str]

    @property
    def key(self) -> str:
        return f"{self.hypothesis}_synthetic"

def build_synthetic_hypothesis_cell_specs() -> list[SyntheticHypothesisCellSpec]:
    """One spec per hypothesis (H1-H11), baseline-only -- no cross-border or
    event variants for this track (per spec §8)."""
```

- [ ] Implement, calling `build_synthetic_hypothesis_currencies(hypothesis)` from Task 3 for each of H1-H11.
- [ ] Tests: exactly 11 specs, one per hypothesis, unique keys, each spec's currencies match what Task 3's builder produces directly.
- [ ] Run targeted tests.
- [ ] Commit: `feat: add SyntheticHypothesisCellSpec and its 11-hypothesis builder`

---

### Task 5: Discrete-level switch search

**Files:** Create `src/economy/synthetic_switch_search.py`; Create `tests/test_synthetic_switch_search.py`.

**Interfaces** (mirrors `equivalence_framework.py`'s `_agent_indifference_point`/`cohort_indifference_points` exactly, so it can plug into the SAME `IndifferencePointRecord`/reporting pipeline unchanged — only the search algorithm differs):
```python
@dataclass(frozen=True)
class SyntheticEquivalenceComparison:
    hypothesis: str
    fixed_currency: str
    varied_currency: str
    varied_field: str  # "governance_score" | "bid_ask_spread" | "peg_error" | "gas_fee"
    levels: tuple[float, ...]  # e.g. BID_ASK_SPREAD_LEVELS, ordered low-to-high value (not attractiveness)

def synthetic_equivalence_comparisons_for(hypothesis: str, currencies: dict[str, CurrencyConfig]) -> list[SyntheticEquivalenceComparison]:
    """H1 has none (medium-alone, holdings-only per spec §6). Every other
    hypothesis has exactly one comparison: fixed_currency = the coin at the
    grid's "best on the varied dimension, worst on the other tested
    dimension" corner; varied_currency = the coin at "worst on the varied
    dimension, worst on the other tested dimension" through "best on both" --
    concretely, compare the LOW value of the OTHER tested dimension held
    constant, varying the SECOND tested dimension across its `levels`."""

def _agent_discrete_switch_point(agent, comparison, fixed_traits, varied_other_traits, client) -> float:
    """Same role as equivalence_framework._agent_indifference_point, but
    asks the switch question at each of comparison.levels directly (one
    call per level, not a multi-round binary search) and returns the
    lowest level (per _HIGHER_IS_BETTER's direction) at which the agent
    says it would switch -- or the most extreme level if the agent never
    switches at any tested level (report that boundary, don't extrapolate
    past it)."""

def cohort_discrete_switch_points(env, comparison, client) -> dict[float, float]:
    """Same role/shape as equivalence_framework.cohort_indifference_points
    (per-agent point -> cohort mean of (point - fixed_value)), built on
    _agent_discrete_switch_point instead of the continuous search."""
```

- [ ] Implement, reusing `_HIGHER_IS_BETTER`-equivalent direction logic (governance_score/liquidity... wait, for this track the fields are `governance_score` (higher better) and `bid_ask_spread`/`peg_error`/`gas_fee` (lower better) -- define a small `_HIGHER_IS_BETTER` dict local to this module, matching `equivalence_framework.py`'s exact values for the overlapping field names (`peg_error`: False, `gas_fee`: False) plus `governance_score`: True, `bid_ask_spread`: False.
- [ ] Reuse `render_switch_prompt`/`call_model_for_switch` (`src/llm/switch_elicitation.py`, `src/llm/llm_router.py`) exactly as-is -- no changes needed there, this is purely a different search algorithm calling the same elicitation primitive.
- [ ] Per-agent model resolution via `agent.build_llm_context().assigned_model` (same fix as Task 5 of the runner-wiring plan -- do NOT reintroduce a `model_id` parameter here).
- [ ] Tests: a threshold-based mock client (reuse/adapt `tests/llm_test_helpers.py`'s `mock_switch_threshold_client` pattern) proves `_agent_discrete_switch_point` picks the correct one of 3 levels for both a higher-is-better field (governance_score) and a lower-is-better field (bid_ask_spread); `cohort_discrete_switch_points`' cohort-mean arithmetic is correct (mock `_agent_discrete_switch_point` directly, same pattern as `tests/test_equivalence_framework.py`'s cohort-mean test); confirm exactly 3 `call_model_for_switch` calls happen per agent (not 7-10), proving the cost reduction.
- [ ] Run targeted tests.
- [ ] Commit: `feat: add discrete-level switch search for the synthetic-coin track`

---

### Task 6: Runner wiring + reporting generalization

**Files:** Modify `src/simulation/hypothesis_matrix_runner.py`; Modify `src/reporting/hypothesis_tables.py`; Modify `tests/test_hypothesis_matrix_runner.py`; Modify `tests/test_hypothesis_tables.py`.

- [ ] Add `track: str = "real"` parameter to `run_hypothesis_matrix` (`"real"` or `"synthetic"`; raise `ValueError` for anything else). When `track == "synthetic"`: use `build_synthetic_hypothesis_cell_specs()` (Task 4) instead of `build_hypothesis_cell_specs()`; use `generate_hypothesis_population`'s SAME population mechanism (no change needed there -- it already takes any `currencies` dict via `Environment.build_from_population`); every hypothesis (including H1) gets `holdings_by_cohort` persisted (not just H1 -- per spec §6, "every hypothesis, not just H1"); every hypothesis EXCEPT H1 additionally gets `cohort_discrete_switch_points` (Task 5) run for each of `synthetic_equivalence_comparisons_for(hypothesis, spec.currencies)`, persisted to the SAME `IndifferencePointRecord` table unchanged. `env.currency_chain_pins = spec.chain_pins` (every synthetic coin always has a chain pin, unlike the real track's `spec.chain_pins or {}`).
- [ ] `run_id` scheme becomes `f"{matrix_run_id}-{track}-{spec.key}-{utility_type}-seed{seed}"` for BOTH tracks (a mechanical, behavior-preserving change for `track="real"` callers only if you thread a literal `"real"` in at that same position -- check whether this breaks any already-committed real-track test asserting the OLD run_id shape without a track segment; if so, decide with the reviewer whether to version the real track's run_id too or keep it exactly as-is and only add the segment for `track="synthetic"`. Flag this explicitly in your task report -- it's the one place this task could accidentally touch the real track's already-shipped behavior.)
- [ ] `build_equilibrium_holdings_table` (`src/reporting/hypothesis_tables.py`): generalize the hardcoded `_H1_ZONE_LABELS = {"USDC": "USD", "EURC": "Euro", "PAXG": "gold"}` lookup to accept an optional `zone_labels: dict[str, str] | None = None` parameter (falling back to the symbol itself when a symbol isn't in the map) so it works for synthetic coins' generated symbols too, without breaking the existing H1 real-coin call sites (which pass nothing and get the old default).
- [ ] Tests: an end-to-end synthetic-track test (H3, mocked client, tiny `num_days`) proves `track="synthetic"` produces `CohortHoldingsRecord` rows (H3 itself, not just H1) AND `IndifferencePointRecord` rows from the discrete search, with `run_id`s containing `"synthetic"`; a real-track regression test confirms `track="real"` (or omitted) still produces the exact same behavior as before this task (re-run the existing `test_hypothesis_matrix_runner.py` suite unchanged and confirm 100% pass).
- [ ] Run: `DATABASE_URL="sqlite:///./assip.db" .venv/bin/python -m pytest tests/test_hypothesis_matrix_runner.py tests/test_hypothesis_tables.py tests/test_synthetic_hypothesis_currencies.py tests/test_synthetic_hypothesis_scenarios.py tests/test_synthetic_switch_search.py -q`
- [ ] Commit: `feat: wire the synthetic-coin track into run_hypothesis_matrix`

---

### Final: whole-plan review

After all 6 tasks are committed, dispatch a whole-plan review (opus) covering the full diff, with special attention to: (1) zero behavior change to the real-coin track when `track="real"` (the reviewer should diff `test_hypothesis_matrix_runner.py`'s pre-existing tests byte-for-byte against their pre-Task-6 versions and confirm every one still passes unchanged), (2) the neutral-fixed-value choices in Task 3 actually hold every untested dimension constant within a hypothesis's grid (a real factor-isolation correctness check), (3) the discrete search's direction logic for each field type. Report findings to the user before considering this done.
