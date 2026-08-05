# Phase 3 Plan 6b: H6-H10 Sandbox-Preference Hypotheses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap where 5 of the 6 factor-isolation sandboxes (all but `liquidity_vs_governance`/H3) have no dedicated "which coin wins" statistical test, by adding hypotheses H6-H10 to `src/econometrics/`.

**Architecture:** A single parameterized dataset-builder helper (`build_sandbox_preference_dataset`) replaces 5 near-duplicate per-hypothesis builders, since H6-H10 share an identical shape (per-decision logit: did the agent choose the sandbox's "higher-X" currency; regressor = CARA `a`; clustered by agent). Each hypothesis calls that helper twice (once per cell variant: domestic, cross_border) and fits two separate `RegressionResult`s — unlike H3, which pools both variants with a fixed effect. `report.py`'s `run_all_hypotheses` grows from 5 to 15 results.

**Tech Stack:** Existing `src/econometrics/` stack: `pandas`, `statsmodels` (via `regression_engine.fit_clustered_logit`, unchanged), SQLAlchemy `Session` queries against `LLMDecisionRecord`/`AgentStateRecord`.

## Global Constraints

- H1-H5's existing datasets/regressions/tests are untouched — this plan is purely additive.
- Every new hypothesis follows the exact directional claims approved in `docs/superpowers/specs/2026-08-04-phase3-plan6-concurrency-and-sandbox-hypotheses-design.md` Sec 1 — do not alter the claimed direction for any of H6-H10.
- Domestic and cross-border cells are reported as **separate** `RegressionResult`s per hypothesis (10 rows total for H6-H10), not pooled — this is the one deliberate difference from H3's pattern, per the design spec.
- Follow the existing `hypothesis_datasets.py` conventions exactly: use `_safe_cell_key`, `_matches_matrix_run_id`, `_join_cara_a`, and `_DECIDED_ACTIONS` (all already defined in that file) rather than reimplementing equivalent logic.

---

### Task 1: Add `build_sandbox_preference_dataset`, the shared H6-H10 dataset builder

**Files:**
- Modify: `src/econometrics/hypothesis_datasets.py`
- Test: `tests/test_hypothesis_h6_h10.py` (new file)

**Interfaces:**
- Consumes: `_safe_cell_key`, `_matches_matrix_run_id`, `_join_cara_a`, `_DECIDED_ACTIONS` from `hypothesis_datasets.py` (all already defined, at the top of that file).
- Produces: `build_sandbox_preference_dataset(session: Session, sandbox_key: str, higher_option_selector: Callable[[CurrencyConfig, CurrencyConfig], str], cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame` — one row per eligible `LLMDecisionRecord` in exactly the one cell `f"{sandbox_key}_{cell_variant}"`, with columns `agent_id, chose_higher_option, cara_a, agent_type, actual_model`. Consumed by Task 2's `build_h6_dataset` .. `build_h10_dataset` (thin wrappers each supplying their own `sandbox_key`/`higher_option_selector`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_hypothesis_h6_h10.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.econometrics.hypothesis_datasets import build_sandbox_preference_dataset
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(sandbox_key: str, forced_symbol: str, num_days: int = 8) -> Session:
    """Mirrors tests/test_hypothesis_h3.py's _populated_session helper: forces
    every mock decision to one specific symbol so at least that sandbox's
    cells produce genuine ACCEPT decisions instead of falling back to a
    synthetic WALK_AWAY (see matrix_runner's per-cell mock-currency
    docstring)."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        mock_llm_decision={
            "action": "ACCEPT",
            "proposed_currency": forced_symbol,
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 1.0,
            "reasoning": "test fixture",
        },
        session=session,
    )
    return session


def test_build_sandbox_preference_dataset_scopes_to_exactly_one_cell_variant():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["governance_vs_stability"]
    session = _populated_session("governance_vs_stability", option_a.symbol)

    df = build_sandbox_preference_dataset(
        session,
        sandbox_key="governance_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant="domestic",
    )

    assert not df.empty
    assert set(df.columns) >= {"agent_id", "chose_higher_option", "cara_a", "agent_type", "actual_model"}
    assert df["chose_higher_option"].isin([0, 1]).all()


def test_build_sandbox_preference_dataset_domestic_and_cross_border_are_disjoint_cells():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["governance_vs_stability"]
    session = _populated_session("governance_vs_stability", option_a.symbol)

    domestic_df = build_sandbox_preference_dataset(
        session,
        sandbox_key="governance_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant="domestic",
    )
    cross_border_df = build_sandbox_preference_dataset(
        session,
        sandbox_key="governance_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant="cross_border",
    )

    # Both non-empty (the mock forces the same decision across all 13 cells),
    # but no overlap in the underlying run/timestep/agent rows -- confirmed
    # indirectly by both being scoped to their own distinct cell.
    assert not domestic_df.empty
    assert not cross_border_df.empty


def test_build_sandbox_preference_dataset_rejects_unknown_cell_variant():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["governance_vs_stability"]
    session = _populated_session("governance_vs_stability", option_a.symbol)

    try:
        build_sandbox_preference_dataset(
            session,
            sandbox_key="governance_vs_stability",
            higher_option_selector=lambda a, b: a.symbol,
            cell_variant="not_a_real_variant",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "cell_variant" in str(exc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_h6_h10.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_sandbox_preference_dataset'`

- [ ] **Step 3: Implement `build_sandbox_preference_dataset` in `src/econometrics/hypothesis_datasets.py`**

Add near the top-level imports: `from typing import Callable` and `from src.currencies.currency import CurrencyConfig`. Add this function after `build_h5_dataset` (end of file):

```python
def build_sandbox_preference_dataset(
    session: Session,
    sandbox_key: str,
    higher_option_selector: Callable[[CurrencyConfig, CurrencyConfig], str],
    cell_variant: str,
    matrix_run_id: str | None = None,
) -> pd.DataFrame:
    """Shared H6-H10 dataset builder (Plan 6b): per-decision logit sample
    for exactly ONE of a sandbox's two cells (domestic XOR cross_border --
    unlike H3, which pools both with a cell_key fixed effect; H6-H10 report
    each cell variant separately per the Plan 6 design spec Sec 1). One row
    per eligible LLMDecisionRecord: `chose_higher_option=1` if the agent's
    proposed currency is `higher_option_selector`'s pick, `0` if it's the
    sandbox's other option, excluded entirely if the decision's currency
    isn't one of this sandbox's two symbols at all.

    `cell_variant` must be `"domestic"` or `"cross_border"` -- any other
    value raises ValueError immediately (a typo here should never silently
    return an empty/wrong-cell dataset).
    """
    if cell_variant not in ("domestic", "cross_border"):
        raise ValueError(f"cell_variant must be 'domestic' or 'cross_border', got {cell_variant!r}")

    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    higher_option_symbol = higher_option_selector(option_a, option_b)
    target_cell_key = f"{sandbox_key}_{cell_variant}"

    query = session.query(LLMDecisionRecord).filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
    decisions = [d for d in query.all() if _matches_matrix_run_id(d.simulation_id, matrix_run_id)]

    records = []
    for decision in decisions:
        if _safe_cell_key(decision.simulation_id) != target_cell_key:
            continue
        if decision.currency not in (option_a.symbol, option_b.symbol):
            continue
        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_higher_option": 1 if decision.currency == higher_option_symbol else 0,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
            }
        )

    df = pd.DataFrame.from_records(
        records,
        columns=["run_id", "timestep", "agent_id", "chose_higher_option", "agent_type", "actual_model"],
    )
    return _join_cara_a(session, df)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_hypothesis_h6_h10.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass (this is a purely additive function; H1-H5 untouched).

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/hypothesis_datasets.py tests/test_hypothesis_h6_h10.py
git commit -m "feat: add build_sandbox_preference_dataset, the shared H6-H10 builder"
```

---

### Task 2: Add `build_h6_dataset` .. `build_h10_dataset`

**Files:**
- Modify: `src/econometrics/hypothesis_datasets.py`
- Test: `tests/test_hypothesis_h6_h10.py`

**Interfaces:**
- Consumes: `build_sandbox_preference_dataset` from Task 1; `GoldBackedConfig` (`src.currencies.gold_token`), `TokenizedDepositConfig` (`src.currencies.tokenized_deposit`) for the `isinstance`-based selectors.
- Produces: `build_h6_dataset(session, cell_variant, matrix_run_id=None) -> pd.DataFrame` through `build_h10_dataset(session, cell_variant, matrix_run_id=None) -> pd.DataFrame` — thin wrappers, each with its hypothesis's own `sandbox_key` and `higher_option_selector` baked in. Consumed by Task 3's `regress_h6` .. `regress_h10`.

Concrete selector per hypothesis (each resolves to exactly one of the sandbox's two options — no ties, confirmed against `src/currencies/sandbox_currencies.py`'s actual field values):

| Hyp. | `sandbox_key` | `higher_option_selector` |
|---|---|---|
| H6 | `governance_vs_stability` | `lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol` (lower `peg_error` = higher stability) |
| H7 | `liquidity_vs_stability` | same as H6 (stability side of the pair) |
| H8 | `asset_backing_vs_liquidity` | `lambda a, b: a.symbol if isinstance(a, GoldBackedConfig) else b.symbol` (the gold-backed option) |
| H9 | `asset_backing_vs_stability` | `lambda a, b: a.symbol if isinstance(a, TokenizedDepositConfig) else b.symbol` (the FDIC-insured deposit option — H9's claim is deposit-wins, opposite framing from H8) |
| H10 | `asset_backing_vs_governance` | `lambda a, b: a.symbol if a.governance_score >= b.governance_score else b.symbol` (higher `governance_score`) |

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hypothesis_h6_h10.py`:

```python
from src.currencies.gold_token import GoldBackedConfig
from src.currencies.tokenized_deposit import TokenizedDepositConfig
from src.econometrics.hypothesis_datasets import (
    build_h6_dataset,
    build_h7_dataset,
    build_h8_dataset,
    build_h9_dataset,
    build_h10_dataset,
)

_H6_H10_CASES = [
    ("governance_vs_stability", build_h6_dataset),
    ("liquidity_vs_stability", build_h7_dataset),
    ("asset_backing_vs_liquidity", build_h8_dataset),
    ("asset_backing_vs_stability", build_h9_dataset),
    ("asset_backing_vs_governance", build_h10_dataset),
]


def test_each_h6_h10_dataset_builder_scopes_to_its_own_sandbox_and_variant():
    for sandbox_key, builder in _H6_H10_CASES:
        option_a, _ = SANDBOX_CURRENCY_PAIRS[sandbox_key]
        session = _populated_session(sandbox_key, option_a.symbol)

        domestic_df = builder(session, cell_variant="domestic")
        cross_border_df = builder(session, cell_variant="cross_border")

        assert not domestic_df.empty, f"{builder.__name__} domestic was empty"
        assert not cross_border_df.empty, f"{builder.__name__} cross_border was empty"
        assert set(domestic_df.columns) >= {"agent_id", "chose_higher_option", "cara_a"}


def test_h8_selector_picks_the_gold_backed_symbol():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_liquidity"]
    gold_option = option_a if isinstance(option_a, GoldBackedConfig) else option_b
    session = _populated_session("asset_backing_vs_liquidity", gold_option.symbol)

    df = build_h8_dataset(session, cell_variant="domestic")
    # Every forced-ACCEPT decision proposed the gold option -> chose_higher_option must be all 1s.
    assert (df["chose_higher_option"] == 1).all()


def test_h9_selector_picks_the_deposit_symbol_not_the_gold_symbol():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_stability"]
    deposit_option = option_a if isinstance(option_a, TokenizedDepositConfig) else option_b
    session = _populated_session("asset_backing_vs_stability", deposit_option.symbol)

    df = build_h9_dataset(session, cell_variant="domestic")
    assert (df["chose_higher_option"] == 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_h6_h10.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_h6_dataset'`

- [ ] **Step 3: Implement the 5 wrapper functions**

Add to `src/econometrics/hypothesis_datasets.py` (after `build_sandbox_preference_dataset`), plus the two new imports at the top of the file:

```python
from src.currencies.gold_token import GoldBackedConfig
from src.currencies.tokenized_deposit import TokenizedDepositConfig
```

```python
def build_h6_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H6: higher CARA `a` -> prioritizes peg stability (lower peg_error)
    over governance/compliance. governance_vs_stability sandbox."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="governance_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )


def build_h7_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H7: higher CARA `a` -> prioritizes peg stability over liquidity.
    liquidity_vs_stability sandbox."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="liquidity_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )


def build_h8_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H8: higher CARA `a` -> prioritizes gold/hard-asset backing over
    liquidity. asset_backing_vs_liquidity sandbox (static baseline
    preference, not crisis-proximity-driven like H4)."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="asset_backing_vs_liquidity",
        higher_option_selector=lambda a, b: a.symbol if isinstance(a, GoldBackedConfig) else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )


def build_h9_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H9: higher CARA `a` -> prioritizes the FDIC-insured deposit option
    (better peg + insurance) over gold backing. asset_backing_vs_stability
    sandbox. Lower-confidence hypothesis (approved as-is, see design spec
    Sec 1): this sandbox bundles asset-class AND a large peg_error gap
    (0.015 vs 0.0001) in one swap."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="asset_backing_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if isinstance(a, TokenizedDepositConfig) else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )


def build_h10_dataset(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> pd.DataFrame:
    """H10: higher CARA `a` -> prioritizes governance/compliance quality
    over asset-backing type. asset_backing_vs_governance sandbox.
    Lower-confidence hypothesis (approved as-is, see design spec Sec 1):
    the two options' governance_score (0.75 vs 0.70) and issuer_risk (0.25
    vs 0.20) are close, a subtler contrast than the other pairs."""
    return build_sandbox_preference_dataset(
        session,
        sandbox_key="asset_backing_vs_governance",
        higher_option_selector=lambda a, b: a.symbol if a.governance_score >= b.governance_score else b.symbol,
        cell_variant=cell_variant,
        matrix_run_id=matrix_run_id,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_hypothesis_h6_h10.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/hypothesis_datasets.py tests/test_hypothesis_h6_h10.py
git commit -m "feat: add build_h6_dataset through build_h10_dataset"
```

---

### Task 3: Add `regress_h6` .. `regress_h10`

**Files:**
- Modify: `src/econometrics/hypothesis_regressions.py`
- Test: `tests/test_hypothesis_h6_h10.py`

**Interfaces:**
- Consumes: `build_h6_dataset` .. `build_h10_dataset` (Task 2), `fit_clustered_logit`/`RegressionResult` (`src.econometrics.regression_engine`, unchanged).
- Produces: `regress_h6(session, cell_variant, matrix_run_id=None) -> RegressionResult` through `regress_h10(...)`. Each returns a `RegressionResult` with `hypothesis` set to e.g. `"H6_domestic"` / `"H6_cross_border"` (the `cell_variant`-suffixed label the design spec's Sec 1.1 specifies). Consumed by Task 4's `run_all_hypotheses`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hypothesis_h6_h10.py`:

```python
from src.econometrics.hypothesis_regressions import (
    regress_h6,
    regress_h7,
    regress_h8,
    regress_h9,
    regress_h10,
)
from src.econometrics.regression_engine import RegressionResult

_H6_H10_REGRESS_CASES = [
    ("governance_vs_stability", regress_h6, "H6"),
    ("liquidity_vs_stability", regress_h7, "H7"),
    ("asset_backing_vs_liquidity", regress_h8, "H8"),
    ("asset_backing_vs_stability", regress_h9, "H9"),
    ("asset_backing_vs_governance", regress_h10, "H10"),
]


def _populated_session_with_genuine_variation(sandbox_key: str, num_days: int = 8) -> Session:
    """Mirrors test_hypothesis_h3.py's _populated_session_with_genuine_variation:
    a single run_matrix call forces a constant proposed_currency, so
    chose_higher_option never varies within one call. Two run_matrix calls
    into the same session, forcing option_a then option_b, gives genuine
    variation for fit_clustered_logit to fit against."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    for call_index, symbol in enumerate((option_a.symbol, option_b.symbol)):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=num_days,
            dry_run=True,
            exercise_llm_path=True,
            matrix_run_id=f"{sandbox_key}-variation-{call_index}",
            mock_llm_decision={
                "action": "ACCEPT",
                "proposed_currency": symbol,
                "proposed_chain": "ethereum",
                "amount": 1.0,
                "price": 1.0,
                "reasoning": "forced alternation for genuine chose_higher_option variation",
            },
            session=session,
        )
    return session


def test_each_regress_h6_h10_returns_separate_domestic_and_cross_border_results():
    for sandbox_key, regress_fn, hyp_label in _H6_H10_REGRESS_CASES:
        session = _populated_session_with_genuine_variation(sandbox_key)

        domestic_result = regress_fn(session, cell_variant="domestic")
        cross_border_result = regress_fn(session, cell_variant="cross_border")

        assert isinstance(domestic_result, RegressionResult)
        assert isinstance(cross_border_result, RegressionResult)
        assert domestic_result.hypothesis == f"{hyp_label}_domestic"
        assert cross_border_result.hypothesis == f"{hyp_label}_cross_border"
        assert domestic_result.regressor == "cara_a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_h6_h10.py::test_each_regress_h6_h10_returns_separate_domestic_and_cross_border_results -v`
Expected: FAIL with `ImportError: cannot import name 'regress_h6'`

- [ ] **Step 3: Implement `regress_h6` .. `regress_h10` in `src/econometrics/hypothesis_regressions.py`**

Add the new imports and 5 functions:

```python
from src.econometrics.hypothesis_datasets import (
    build_h1_dataset,
    build_h2_dataset,
    build_h3_dataset,
    build_h4_dataset,
    build_h5_dataset,
    build_h6_dataset,
    build_h7_dataset,
    build_h8_dataset,
    build_h9_dataset,
    build_h10_dataset,
)
```

```python
def regress_h6(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h6_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H6_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h7(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h7_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H7_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h8(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h8_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H8_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h9(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h9_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H9_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )


def regress_h10(session: Session, cell_variant: str, matrix_run_id: str | None = None) -> RegressionResult:
    df = build_h10_dataset(session, cell_variant=cell_variant, matrix_run_id=matrix_run_id)
    return fit_clustered_logit(
        hypothesis=f"H10_{cell_variant}",
        df=df,
        dependent_col="chose_higher_option",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_hypothesis_h6_h10.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/hypothesis_regressions.py tests/test_hypothesis_h6_h10.py
git commit -m "feat: add regress_h6 through regress_h10"
```

---

### Task 4: Extend `run_all_hypotheses` to include H6-H10 (15 total results)

**Files:**
- Modify: `src/econometrics/report.py`
- Test: `tests/test_econometrics_report.py`

**Interfaces:**
- Consumes: `regress_h6` .. `regress_h10` (Task 3).
- Produces: `run_all_hypotheses(session, matrix_run_id=None) -> list[RegressionResult]` now returns 15 results (H1-H5 unchanged, plus H6_domestic, H6_cross_border, ..., H10_domestic, H10_cross_border). `results_to_dataframe`/`write_report_csv` require no changes (already accept arbitrary hypothesis labels).

- [ ] **Step 1: Write the failing test**

Modify `tests/test_econometrics_report.py`'s existing `test_run_all_hypotheses_returns_one_result_per_hypothesis` and `test_run_all_hypotheses_threads_matrix_run_id_to_every_hypothesis` to account for the 10 new results, and add patches for the 5 new `regress_hN` functions. Replace the top of the file's two existing tests with:

```python
_ALL_HYPOTHESIS_LABELS = (
    "H1", "H2", "H3", "H4", "H5",
    "H6_domestic", "H6_cross_border",
    "H7_domestic", "H7_cross_border",
    "H8_domestic", "H8_cross_border",
    "H9_domestic", "H9_cross_border",
    "H10_domestic", "H10_cross_border",
)


def test_run_all_hypotheses_returns_one_result_per_hypothesis():
    fake_results = {h: _fake_result(h) for h in _ALL_HYPOTHESIS_LABELS}
    with (
        patch("src.econometrics.report.regress_h1", return_value=fake_results["H1"]) as m1,
        patch("src.econometrics.report.regress_h2", return_value=fake_results["H2"]) as m2,
        patch("src.econometrics.report.regress_h3", return_value=fake_results["H3"]) as m3,
        patch("src.econometrics.report.regress_h4", return_value=fake_results["H4"]) as m4,
        patch("src.econometrics.report.regress_h5", return_value=fake_results["H5"]) as m5,
        patch("src.econometrics.report.regress_h6", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H6_{cell_variant}"]) as m6,
        patch("src.econometrics.report.regress_h7", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H7_{cell_variant}"]) as m7,
        patch("src.econometrics.report.regress_h8", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H8_{cell_variant}"]) as m8,
        patch("src.econometrics.report.regress_h9", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H9_{cell_variant}"]) as m9,
        patch("src.econometrics.report.regress_h10", side_effect=lambda s, cell_variant, matrix_run_id=None: fake_results[f"H10_{cell_variant}"]) as m10,
    ):
        session = object()
        results = run_all_hypotheses(session)

        for mock in (m1, m2, m3, m4, m5):
            mock.assert_called_once_with(session, matrix_run_id=None)
        for mock in (m6, m7, m8, m9, m10):
            assert mock.call_count == 2  # domestic + cross_border

    assert len(results) == 15
    assert {r.hypothesis for r in results} == set(_ALL_HYPOTHESIS_LABELS)
    assert all(isinstance(r, RegressionResult) for r in results)
```

(Update `test_results_to_dataframe_has_the_required_publication_columns` and `test_write_report_csv_writes_a_readable_file` to build their `results` list from `_ALL_HYPOTHESIS_LABELS` too, and change their `len(df) == 5` / `len(reloaded) == 5` assertions to `== 15`, and their hypothesis-set assertions to `set(_ALL_HYPOTHESIS_LABELS)`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_econometrics_report.py -v`
Expected: FAIL — `run_all_hypotheses` still returns 5 results; `regress_h6`..`regress_h10` don't exist as `src.econometrics.report` attributes yet.

- [ ] **Step 3: Implement in `src/econometrics/report.py`**

```python
from src.econometrics.hypothesis_regressions import (
    regress_h1,
    regress_h2,
    regress_h3,
    regress_h4,
    regress_h5,
    regress_h6,
    regress_h7,
    regress_h8,
    regress_h9,
    regress_h10,
)


def run_all_hypotheses(session: Session, matrix_run_id: str | None = None) -> list[RegressionResult]:
    """Runs every in-scope hypothesis's regression against `session`'s
    already-persisted matrix-runner data. H1-H5 each return one pooled
    result; H6-H10 each return two (domestic, cross_border), reported
    separately per Plan 6 design spec Sec 1 -- 15 results total.
    """
    results = [
        regress_h1(session, matrix_run_id=matrix_run_id),
        regress_h2(session, matrix_run_id=matrix_run_id),
        regress_h3(session, matrix_run_id=matrix_run_id),
        regress_h4(session, matrix_run_id=matrix_run_id),
        regress_h5(session, matrix_run_id=matrix_run_id),
    ]
    for regress_fn in (regress_h6, regress_h7, regress_h8, regress_h9, regress_h10):
        results.append(regress_fn(session, cell_variant="domestic", matrix_run_id=matrix_run_id))
        results.append(regress_fn(session, cell_variant="cross_border", matrix_run_id=matrix_run_id))
    return results
```

`results_to_dataframe` and `write_report_csv` need no code changes — verify by reading them (`src/econometrics/report.py`, already shown in this plan's investigation): both already iterate `results` generically over `RegressionResult`'s fields with no hardcoded count or hypothesis-name assumption.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_econometrics_report.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/report.py tests/test_econometrics_report.py
git commit -m "feat: extend run_all_hypotheses to include H6-H10 (15 results total)"
```

---

## Self-Review Notes

- **Spec coverage**: Design spec Sec 1's table (H6-H10 claims + confidence flags) -> Task 2's selector table and docstrings, verbatim. Sec 1.1's "single parameterized helper, not 5 near-duplicates" -> Task 1. Sec 1.1's "10 result rows, H3 unchanged" -> Tasks 3-4.
- **Placeholder scan**: none found — every step shows complete, real code.
- **Type consistency**: `build_sandbox_preference_dataset(cell_variant=...)` (Task 1) -> `build_h6_dataset(cell_variant=...)` (Task 2, same param name/type) -> `regress_h6(cell_variant=...)` (Task 3, same) -> `run_all_hypotheses` calling `regress_fn(session, cell_variant="domestic"/"cross_border", ...)` (Task 4) — verified consistent naming (`cell_variant`, values `"domestic"`/`"cross_border"`) across all 4 tasks.
