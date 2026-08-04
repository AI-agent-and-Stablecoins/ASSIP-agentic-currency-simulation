# Phase 3 Plan 5: Econometrics Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the publication-grade econometrics engine that regresses H1-H5 against the persisted matrix-runner data (β, SE, 95% CI, p-value, pseudo-R², adjusted pseudo-R² per hypothesis), per `docs/superpowers/specs/2026-08-02-phase3-plan5-econometrics-design.md`.

**Architecture:** A small additive extension to the already-merged Plan 4 persistence layer (Task 1: two new pieces of per-decision data needed for H2), a cell-identity parsing helper (Task 2), one shared statsmodels-based clustered-logit fitting function (Task 3) that every per-hypothesis dataset builder feeds into (Tasks 4-8), and a report assembler that runs all 5 and writes one output table (Task 9). Fully independent of any live `Environment`/API keys — this reads an already-populated database.

**Tech Stack:** Python 3.12, SQLAlchemy (existing `database/models.py`/`database/session.py`), pandas (already a dependency), new `statsmodels` dependency.

## Global Constraints

- Every new field/column is **nullable/optional with a default**, so no pre-existing row, test, or caller breaks (per the design spec's "zero behavior change for pre-existing callers" convention already used in Plans 2-4).
- H6 (privacy) is explicitly out of scope — do not build anything for it.
- No task in this plan constructs a live `Environment`, calls any LLM/Polygon client, or requires `dry_run=False` — this plan is 100% offline, reading only from an already-populated (or freshly test-populated) database.
- Every dataset-builder function takes a `Session` and returns a `pandas.DataFrame` — no ORM objects escape `src/econometrics/hypothesis_datasets.py`.
- Follow the existing repo convention: type hints on every function, Pydantic/dataclass for structured results, no bare dicts as a "shape" where a typed model would do.

---

### Task 1: Persist each decision's spread-optimal/gas-optimal choice + add `statsmodels`

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/simulation/timestep.py:57-98` (the `LLMDecisionRecord` Pydantic model), `src/simulation/timestep.py:220-317` (`_make_llm_decide_closure`), `src/simulation/timestep.py:500-623` (the buyer/good loop's closure construction)
- Modify: `database/models.py` (the `LLMDecisionRecord` ORM model, around line 109-144)
- Modify: `database/repository.py:48-75` (`LLMDecisionLogEntry`), `database/repository.py:360-425` (`_llm_decision_log_entry`)
- Test: `tests/test_simulation.py` (extend an existing `use_llm=True` test), `tests/test_full_persistence.py` (extend `test_llm_decision_log_entry_hashes_rendered_prompt_not_reasoning`-style test)

**Interfaces:**
- Consumes: `src.blockchain.routing_engine.CurrencyChainOption` (fields: `currency_symbol: str`, `chain_name: str`, `governance_score: float`, `liquidity_score: float`, `peg_error: float`, `gas_fee: float`, `finality_seconds: float`, `genius_compliant: bool`).
- Produces: `src.simulation.timestep.LLMDecisionRecord` gains 4 new optional fields: `spread_optimal_currency: str | None`, `spread_optimal_chain: str | None`, `gas_optimal_currency: str | None`, `gas_optimal_chain: str | None`. `database.models.LLMDecisionRecord` and `database.repository.LLMDecisionLogEntry` gain the same 4 fields (non-nullable `str` on the ORM/log-entry side, defaulting to `""` when unavailable, matching the existing convention for `currency`/`chain` on those same classes).

- [ ] **Step 1: Add the `statsmodels` dependency**

Edit `pyproject.toml`, adding a new optional-dependencies group after the existing `llm` group:

```toml
[project.optional-dependencies]
# Not required for Phase 1 core simulation -- only needed for the optional
# integrations in metrics/wandb_logger.py and scripts/calibrate_currency_configs.py.
observability = ["wandb>=0.17"]
market-data = ["requests>=2.31"]
# Phase 2: OpenRouter (src/llm/llm_router.py) and Polygon
# (src/llm/market_intelligence.py) both go over HTTP via httpx.
llm = ["httpx>=0.27"]
# Phase 3 Plan 5: the H1-H5 hypothesis regression engine (src/econometrics/).
econometrics = ["statsmodels>=0.14"]
```

Run: `pip install -e ".[econometrics]"` (or `pip install statsmodels>=0.14` directly in the active venv) so the rest of this plan's tests can import it.

- [ ] **Step 2: Write the failing test for the new `LLMDecisionRecord` (Pydantic) fields**

Add to `tests/test_simulation.py` (near any other `LLMDecisionRecord`-construction test):

```python
def test_llm_decision_record_has_optional_spread_and_gas_optimal_fields():
    from src.simulation.timestep import LLMDecisionRecord

    decision = LLMDecisionRecord(
        agent_id="a1",
        agent_type="consumer",
        risk_profile="medium",
        utility_type="cara",
        requested_model="vendor/model",
        actual_model="vendor/model",
        success=True,
    )
    assert decision.spread_optimal_currency is None
    assert decision.spread_optimal_chain is None
    assert decision.gas_optimal_currency is None
    assert decision.gas_optimal_chain is None

    decision_with_optima = LLMDecisionRecord(
        agent_id="a1",
        agent_type="consumer",
        risk_profile="medium",
        utility_type="cara",
        requested_model="vendor/model",
        actual_model="vendor/model",
        success=True,
        spread_optimal_currency="USDC",
        spread_optimal_chain="solana",
        gas_optimal_currency="USDT",
        gas_optimal_chain="base",
    )
    assert decision_with_optima.spread_optimal_currency == "USDC"
    assert decision_with_optima.gas_optimal_chain == "base"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_simulation.py::test_llm_decision_record_has_optional_spread_and_gas_optimal_fields -v`
Expected: FAIL with `TypeError: LLMDecisionRecord() got an unexpected keyword argument 'spread_optimal_currency'` (from the second construction).

- [ ] **Step 4: Add the 4 fields to `src.simulation.timestep.LLMDecisionRecord`**

In `src/simulation/timestep.py`, find the `LLMDecisionRecord` class (around line 57) and add 4 fields after `hallucination`:

```python
    hallucination: HallucinationResult | None = None
    spread_optimal_currency: str | None = None
    spread_optimal_chain: str | None = None
    gas_optimal_currency: str | None = None
    gas_optimal_chain: str | None = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_simulation.py::test_llm_decision_record_has_optional_spread_and_gas_optimal_fields -v`
Expected: PASS

- [ ] **Step 6: Write the failing test for computing spread/gas-optimal from a candidate list**

Add a new test file `tests/test_spread_gas_optimal.py`:

```python
from src.blockchain.routing_engine import CurrencyChainOption
from src.simulation.timestep import _spread_and_gas_optimal


def test_spread_and_gas_optimal_picks_highest_liquidity_and_lowest_gas():
    candidates = [
        CurrencyChainOption(
            currency_symbol="USDC", chain_name="ethereum", governance_score=0.9,
            liquidity_score=0.5, peg_error=0.0, gas_fee=5.0, finality_seconds=12.0,
            genius_compliant=True,
        ),
        CurrencyChainOption(
            currency_symbol="USDC", chain_name="solana", governance_score=0.9,
            liquidity_score=0.5, peg_error=0.0, gas_fee=0.01, finality_seconds=1.0,
            genius_compliant=True,
        ),
        CurrencyChainOption(
            currency_symbol="USDT", chain_name="ethereum", governance_score=0.6,
            liquidity_score=0.95, peg_error=0.01, gas_fee=5.0, finality_seconds=12.0,
            genius_compliant=False,
        ),
    ]

    spread_currency, spread_chain, gas_currency, gas_chain = _spread_and_gas_optimal(candidates)

    assert (spread_currency, spread_chain) == ("USDT", "ethereum")  # highest liquidity_score (0.95)
    assert (gas_currency, gas_chain) == ("USDC", "solana")  # lowest gas_fee (0.01)


def test_spread_and_gas_optimal_can_be_the_same_candidate():
    candidates = [
        CurrencyChainOption(
            currency_symbol="USDC", chain_name="solana", governance_score=0.9,
            liquidity_score=0.99, peg_error=0.0, gas_fee=0.01, finality_seconds=1.0,
            genius_compliant=True,
        ),
        CurrencyChainOption(
            currency_symbol="USDT", chain_name="ethereum", governance_score=0.6,
            liquidity_score=0.5, peg_error=0.01, gas_fee=5.0, finality_seconds=12.0,
            genius_compliant=False,
        ),
    ]

    spread_currency, spread_chain, gas_currency, gas_chain = _spread_and_gas_optimal(candidates)

    assert (spread_currency, spread_chain) == (gas_currency, gas_chain) == ("USDC", "solana")
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pytest tests/test_spread_gas_optimal.py -v`
Expected: FAIL with `ImportError: cannot import name '_spread_and_gas_optimal'`

- [ ] **Step 8: Implement `_spread_and_gas_optimal` and thread it through the closure**

In `src/simulation/timestep.py`, add this helper function right before `_make_llm_decide_closure` (around line 219):

```python
def _spread_and_gas_optimal(candidates: list[CurrencyChainOption]) -> tuple[str, str, str, str]:
    """Identifies which candidate this round was spread-optimal (highest
    liquidity_score -- the codebase's bid-ask-spread proxy, per Plan 2's
    design spec) vs. gas-optimal (lowest gas_fee). Used by Plan 5's H2
    tradeoff-sample design (see docs/superpowers/specs/
    2026-08-02-phase3-plan5-econometrics-design.md Sec 2) to identify
    whether a genuine spread-vs-gas tradeoff existed that round.
    """
    spread_optimal = max(candidates, key=lambda c: c.liquidity_score)
    gas_optimal = min(candidates, key=lambda c: c.gas_fee)
    return (
        spread_optimal.currency_symbol,
        spread_optimal.chain_name,
        gas_optimal.currency_symbol,
        gas_optimal.chain_name,
    )
```

Then update `_make_llm_decide_closure`'s signature (around line 220) to accept the 4 new values:

```python
def _make_llm_decide_closure(
    agent: BaseAgent,
    agent_class: str,
    context: AgentDecisionContext,
    model_id: str,
    client: httpx.Client,
    supported_currencies: set[str],
    supported_chains: set[str],
    listing_true_price: float,
    decision_log: list[LLMDecisionRecord],
    buyer_wallet_balances: dict[str, float],
    spread_optimal_currency: str,
    spread_optimal_chain: str,
    gas_optimal_currency: str,
    gas_optimal_chain: str,
) -> Callable[[NegotiationSession], NegotiationAction]:
```

And in its `decision_log.append(LLMDecisionRecord(...))` call (around line 282), add the 4 new fields:

```python
        decision_log.append(
            LLMDecisionRecord(
                agent_id=agent.agent_id,
                agent_type=agent.agent_class,
                risk_profile=agent.risk_profile,
                utility_type=agent.utility_type,
                requested_model=model_id,
                actual_model=model_id,
                success=action is not None,
                correction_attempts=telemetry.get("correction_attempts", 0),
                failure_reason=telemetry.get("failure_reason"),
                negotiation_id=session.negotiation_id,
                round=session.current_round,
                action=action.action.value if action is not None else None,
                currency_symbol=action.currency_symbol if action is not None else None,
                chain_name=action.chain_name if action is not None else None,
                amount=action.amount if action is not None else None,
                price=action.price if action is not None else None,
                reasoning=action.reasoning if action is not None else None,
                rendered_prompt=telemetry.get("rendered_prompt"),
                hallucination=hallucination,
                spread_optimal_currency=spread_optimal_currency,
                spread_optimal_chain=spread_optimal_chain,
                gas_optimal_currency=gas_optimal_currency,
                gas_optimal_chain=gas_optimal_chain,
            )
        )
```

Finally, in `run_timestep`'s buyer/good loop (around line 517-518, right after `if not candidates: continue`), compute the 4 values once and pass them to both closures:

```python
            if not candidates:
                continue

            spread_optimal_currency, spread_optimal_chain, gas_optimal_currency, gas_optimal_chain = (
                _spread_and_gas_optimal(candidates)
            )

            if use_llm:
```

And update both `_make_llm_decide_closure(...)` call sites (around line 600 and 612) to pass the 4 new positional values, e.g.:

```python
                buyer_decide = _make_llm_decide_closure(
                    buyer,
                    "buyer",
                    buyer_context,
                    buyer.assigned_model,
                    openrouter_client,
                    supported_currencies,
                    supported_chains,
                    listing.true_price,
                    result.llm_decisions,
                    buyer_wallet_balances=buyer_wallet_balances_usd,
                    spread_optimal_currency=spread_optimal_currency,
                    spread_optimal_chain=spread_optimal_chain,
                    gas_optimal_currency=gas_optimal_currency,
                    gas_optimal_chain=gas_optimal_chain,
                )
                seller_decide = _make_llm_decide_closure(
                    seller,
                    "seller",
                    seller_context,
                    seller.assigned_model,
                    openrouter_client,
                    supported_currencies,
                    supported_chains,
                    listing.true_price,
                    result.llm_decisions,
                    buyer_wallet_balances=buyer_wallet_balances_usd,
                    spread_optimal_currency=spread_optimal_currency,
                    spread_optimal_chain=spread_optimal_chain,
                    gas_optimal_currency=gas_optimal_currency,
                    gas_optimal_chain=gas_optimal_chain,
                )
```

- [ ] **Step 9: Run tests to verify they pass, then run the full suite**

Run: `pytest tests/test_spread_gas_optimal.py tests/test_simulation.py -v`
Expected: PASS

Run: `pytest -q`
Expected: all pre-existing tests still pass (the 4 new params are added to `_make_llm_decide_closure`'s signature, so double-check no OTHER call site of that function exists besides the two just updated — `grep -n "_make_llm_decide_closure(" src/simulation/timestep.py` should show exactly 3 lines: the `def`, and these two call sites).

- [ ] **Step 10: Write the failing test for the DB-level persistence (ORM model + `_llm_decision_log_entry`)**

Add to `tests/test_full_persistence.py`:

```python
def test_llm_decision_log_entry_carries_spread_and_gas_optimal_fields():
    decision = TimestepLLMDecisionRecord(
        agent_id="a1",
        agent_type="consumer",
        risk_profile="medium",
        utility_type="cara",
        requested_model="vendor/model",
        actual_model="vendor/model",
        success=True,
        action="ACCEPT",
        currency_symbol="USDC",
        chain_name="solana",
        amount=1.0,
        price=100.0,
        spread_optimal_currency="USDT",
        spread_optimal_chain="ethereum",
        gas_optimal_currency="USDC",
        gas_optimal_chain="solana",
    )

    entry = _llm_decision_log_entry(decision, "dec-1", "run-1", 0, agent=None, scenario_name="master_simulation")

    assert entry.spread_optimal_currency == "USDT"
    assert entry.spread_optimal_chain == "ethereum"
    assert entry.gas_optimal_currency == "USDC"
    assert entry.gas_optimal_chain == "solana"
```

- [ ] **Step 11: Run test to verify it fails**

Run: `pytest tests/test_full_persistence.py::test_llm_decision_log_entry_carries_spread_and_gas_optimal_fields -v`
Expected: FAIL with `TypeError: LLMDecisionLogEntry() got an unexpected keyword argument 'spread_optimal_currency'` (or similar, from `_llm_decision_log_entry` not yet returning these fields).

- [ ] **Step 12: Add the fields to `database/models.py`'s `LLMDecisionRecord`, `database/repository.py`'s `LLMDecisionLogEntry`, and `_llm_decision_log_entry`**

In `database/models.py`, add 4 nullable columns to the ORM `LLMDecisionRecord` class, right after `governance_prompt_enabled` and before `timestamp`:

```python
    governance_prompt_enabled: Mapped[bool] = mapped_column(Boolean)
    spread_optimal_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    spread_optimal_chain: Mapped[str | None] = mapped_column(String, nullable=True)
    gas_optimal_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    gas_optimal_chain: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
```

In `database/repository.py`, add the same 4 fields (non-nullable `str`, matching `currency`/`chain`'s existing convention) to `LLMDecisionLogEntry`, right after `domestic_or_cross_border`:

```python
    scenario: str
    domestic_or_cross_border: str
    governance_prompt_enabled: bool
    spread_optimal_currency: str
    spread_optimal_chain: str
    gas_optimal_currency: str
    gas_optimal_chain: str
```

Then update `_llm_decision_log_entry`'s return statement to populate them from `decision`:

```python
    return LLMDecisionLogEntry(
        decision_id=decision_id,
        simulation_id=run_id,
        timestep=timestep,
        agent_id=decision.agent_id,
        agent_type=decision.agent_type,
        requested_model=decision.requested_model,
        actual_model=decision.actual_model,
        fallback_used=False,
        fallback_reason=decision.failure_reason,
        model_attempts=model_attempts,
        prompt_version=prompt_version,
        rendered_prompt_hash=hash_rendered_prompt(decision.rendered_prompt or ""),
        system_prompt=decision.rendered_prompt or "",
        action=decision.action or "NONE",
        currency=decision.currency_symbol or "",
        chain=decision.chain_name or "",
        amount=decision.amount if decision.amount is not None else 0.0,
        price=decision.price if decision.price is not None else 0.0,
        reported_reasoning=decision.reasoning or (decision.failure_reason or ""),
        negotiation_id=decision.negotiation_id,
        round=decision.round if decision.round is not None else 0,
        risk_profile=decision.risk_profile,
        utility_type=decision.utility_type,
        utility_parameters=_llm_decision_utility_parameters(agent),
        scenario=scenario_name,
        domestic_or_cross_border="unknown",
        governance_prompt_enabled=False,
        spread_optimal_currency=decision.spread_optimal_currency or "",
        spread_optimal_chain=decision.spread_optimal_chain or "",
        gas_optimal_currency=decision.gas_optimal_currency or "",
        gas_optimal_chain=decision.gas_optimal_chain or "",
    )
```

- [ ] **Step 13: Run tests to verify they pass, then run the full suite**

Run: `pytest tests/test_full_persistence.py -v`
Expected: PASS

Run: `pytest -q`
Expected: all tests pass — the new columns are nullable/have defaults, so no pre-existing row construction (in any other test) breaks.

- [ ] **Step 14: Commit**

```bash
git add pyproject.toml src/simulation/timestep.py database/models.py database/repository.py tests/test_simulation.py tests/test_spread_gas_optimal.py tests/test_full_persistence.py
git commit -m "feat: persist each decision's spread-optimal/gas-optimal candidate, add statsmodels dependency"
```

---

### Task 2: Cell-identity parsing (`cell_key_from_run_id`)

**Files:**
- Create: `src/econometrics/__init__.py` (empty)
- Create: `src/econometrics/cell_identity.py`
- Test: `tests/test_cell_identity.py`

**Interfaces:**
- Consumes: `src.simulation.matrix_runner._build_cell_specs() -> list[_CellSpec]` (each with a `.key: str` attribute) — the single source of truth for the 13 valid cell keys.
- Produces: `cell_key_from_run_id(run_id: str) -> str`, raises `ValueError` for an unrecognized `run_id`. Every later task's dataset builders (Tasks 4-8) call this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cell_identity.py`:

```python
import pytest

from src.econometrics.cell_identity import cell_key_from_run_id


def test_cell_key_from_run_id_recovers_master():
    assert cell_key_from_run_id("matrix-abc123-master-seed0") == "master"


def test_cell_key_from_run_id_recovers_sandbox_domestic_and_cross_border():
    assert (
        cell_key_from_run_id("matrix-abc123-liquidity_vs_governance_domestic-seed3")
        == "liquidity_vs_governance_domestic"
    )
    assert (
        cell_key_from_run_id("matrix-abc123-liquidity_vs_governance_cross_border-seed3")
        == "liquidity_vs_governance_cross_border"
    )


def test_cell_key_from_run_id_handles_a_matrix_run_id_containing_hyphens():
    assert (
        cell_key_from_run_id("pilot-run-2026-08-02-asset_backing_vs_liquidity_cross_border-seed12")
        == "asset_backing_vs_liquidity_cross_border"
    )


def test_cell_key_from_run_id_raises_for_unrecognized_run_id():
    with pytest.raises(ValueError):
        cell_key_from_run_id("not-a-matrix-run-id-at-all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cell_identity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.econometrics'`

- [ ] **Step 3: Implement**

Create `src/econometrics/__init__.py` (empty file).

Create `src/econometrics/cell_identity.py`:

```python
"""Recovers which of the 13 matrix-runner cells a persisted
`LLMDecisionRecord` row came from.

`LLMDecisionRecord.scenario`/`.domestic_or_cross_border` cannot distinguish
all 13 cells (see docs/superpowers/specs/
2026-08-02-phase3-plan5-econometrics-design.md Sec 1: `domestic_or_cross_
border` is unconditionally "unknown" in production, and a sandbox's
domestic/cross-border cells share the same `scenario` value) -- only
`.simulation_id` (== the matrix runner's `run_id`,
"{matrix_run_id}-{cell_key}-seed{seed}") does.
"""

import re

from src.simulation.matrix_runner import _build_cell_specs

# Sorted longest-first so a shorter key can never accidentally match before
# a longer one that shares a suffix/prefix.
_VALID_CELL_KEYS = sorted((spec.key for spec in _build_cell_specs()), key=len, reverse=True)

_SEED_SUFFIX = re.compile(r"-seed\d+$")


def cell_key_from_run_id(run_id: str) -> str:
    """Extracts the matrix-runner cell key (e.g. "master",
    "liquidity_vs_governance_domestic") from a `run_id` of the form
    "{matrix_run_id}-{cell_key}-seed{seed}". `matrix_run_id` itself may
    contain hyphens (it's caller-supplied or `generate_id`-produced), so
    this matches against the KNOWN set of 13 valid cell keys rather than
    naively splitting on "-". Raises `ValueError` if no known cell key
    matches (e.g. a `run_id` from outside `run_matrix`).
    """
    seed_match = _SEED_SUFFIX.search(run_id)
    prefix = run_id[: seed_match.start()] if seed_match else run_id
    for key in _VALID_CELL_KEYS:
        if prefix.endswith(f"-{key}"):
            return key
    raise ValueError(f"run_id {run_id!r} does not match any known matrix-runner cell key")
```

- [ ] **Step 4: Run test to verify it passes, then run full suite**

Run: `pytest tests/test_cell_identity.py -v`
Expected: PASS

Run: `pytest -q`
Expected: all tests pass (this task only adds new files).

- [ ] **Step 5: Commit**

```bash
git add src/econometrics/__init__.py src/econometrics/cell_identity.py tests/test_cell_identity.py
git commit -m "feat: add cell_key_from_run_id, the only reliable way to recover which of the 13 matrix cells a decision came from"
```

---

### Task 3: Shared regression engine (`fit_clustered_logit`)

**Files:**
- Create: `src/econometrics/regression_engine.py`
- Test: `tests/test_regression_engine.py`

**Interfaces:**
- Consumes: `pandas.DataFrame`, `statsmodels.api` (installed in Task 1).
- Produces: `RegressionResult` (frozen dataclass: `hypothesis: str`, `regressor: str`, `beta: float`, `se: float`, `ci_lower: float`, `ci_upper: float`, `p_value: float`, `pseudo_r2: float`, `adjusted_pseudo_r2: float`, `n_obs: int`) and `fit_clustered_logit(hypothesis: str, df: pd.DataFrame, dependent_col: str, regressor_col: str, cluster_col: str, fixed_effect_cols: list[str]) -> RegressionResult`. Every hypothesis's regression function (Tasks 4-8) calls this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_regression_engine.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.econometrics.regression_engine import RegressionResult, fit_clustered_logit


def _synthetic_dataset(n_agents: int = 50, decisions_per_agent: int = 20, seed: int = 0) -> pd.DataFrame:
    """Builds a dataset where higher `regressor` genuinely makes `chose_x`
    more likely (a real logistic relationship, not noise), with several
    repeated decisions per agent -- exercising both the regression fit
    itself and the agent-clustering machinery."""
    rng = np.random.default_rng(seed)
    rows = []
    for agent_idx in range(n_agents):
        agent_regressor = rng.uniform(-2.0, 2.0)
        agent_type = "consumer" if agent_idx % 2 == 0 else "bank"
        model = "vendor/model-a" if agent_idx % 3 == 0 else "vendor/model-b"
        for _ in range(decisions_per_agent):
            probability = 1.0 / (1.0 + np.exp(-(2.0 * agent_regressor)))
            chose_x = 1 if rng.uniform(0.0, 1.0) < probability else 0
            rows.append(
                {
                    "agent_id": f"agent-{agent_idx}",
                    "chose_x": chose_x,
                    "regressor": agent_regressor,
                    "agent_type": agent_type,
                    "actual_model": model,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_fit_clustered_logit_recovers_a_positive_relationship():
    df = _synthetic_dataset()
    result = fit_clustered_logit(
        hypothesis="H_TEST",
        df=df,
        dependent_col="chose_x",
        regressor_col="regressor",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )

    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H_TEST"
    assert result.regressor == "regressor"
    assert result.beta > 0  # the synthetic data has a genuine positive relationship
    assert result.p_value < 0.05  # should be clearly statistically significant given the sample size
    assert result.ci_lower < result.beta < result.ci_upper
    assert 0.0 <= result.pseudo_r2 <= 1.0
    assert result.n_obs == len(df)


def test_fit_clustered_logit_works_with_no_fixed_effects():
    df = _synthetic_dataset(n_agents=30, decisions_per_agent=10, seed=1)
    result = fit_clustered_logit(
        hypothesis="H_TEST2",
        df=df,
        dependent_col="chose_x",
        regressor_col="regressor",
        cluster_col="agent_id",
        fixed_effect_cols=[],
    )
    assert result.n_obs == len(df)


def test_fit_clustered_logit_raises_on_empty_dataframe():
    df = pd.DataFrame(columns=["agent_id", "chose_x", "regressor", "agent_type", "actual_model"])
    with pytest.raises(ValueError):
        fit_clustered_logit(
            hypothesis="H_EMPTY",
            df=df,
            dependent_col="chose_x",
            regressor_col="regressor",
            cluster_col="agent_id",
            fixed_effect_cols=["agent_type"],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regression_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.econometrics.regression_engine'`

- [ ] **Step 3: Implement**

Create `src/econometrics/regression_engine.py`:

```python
"""Shared logistic-regression fitting for every H1-H5 hypothesis, per
docs/superpowers/specs/2026-08-02-phase3-plan5-econometrics-design.md Sec 3:
per-decision logit, agent-clustered standard errors, McFadden pseudo-R^2/
adjusted pseudo-R^2 in place of OLS's R^2/adjusted R^2 (undefined for a
binary outcome).
"""

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class RegressionResult:
    hypothesis: str
    regressor: str
    beta: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    pseudo_r2: float
    adjusted_pseudo_r2: float
    n_obs: int


def fit_clustered_logit(
    hypothesis: str,
    df: pd.DataFrame,
    dependent_col: str,
    regressor_col: str,
    cluster_col: str,
    fixed_effect_cols: list[str],
) -> RegressionResult:
    """Fits `dependent_col ~ regressor_col + <fixed_effect_cols dummies>`
    via logistic regression with standard errors clustered by
    `cluster_col` (agent-level, per the design spec's Sec 0 decision).
    Returns `regressor_col`'s own coefficient/SE/CI/p-value plus the whole
    model's McFadden pseudo-R^2/adjusted pseudo-R^2 and sample size.
    Raises `ValueError` if `df` is empty (a hypothesis's dataset builder
    found no eligible decisions at all -- a real problem to surface
    loudly, not silently return a meaningless fit for).
    """
    if df.empty:
        raise ValueError(
            f"fit_clustered_logit({hypothesis!r}): received an empty DataFrame -- "
            "the dataset builder found no eligible decisions for this hypothesis."
        )

    y = df[dependent_col].astype(float)
    x_numeric = df[[regressor_col]].astype(float)
    if fixed_effect_cols:
        x_dummies = pd.get_dummies(df[fixed_effect_cols], drop_first=True, dtype=float)
        x = pd.concat([x_numeric, x_dummies], axis=1)
    else:
        x = x_numeric
    x = sm.add_constant(x)

    model = sm.Logit(y, x)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]}, disp=0)

    ci = result.conf_int().loc[regressor_col]
    adjusted_pseudo_r2 = 1.0 - (result.llf - result.df_model) / result.llnull

    return RegressionResult(
        hypothesis=hypothesis,
        regressor=regressor_col,
        beta=float(result.params[regressor_col]),
        se=float(result.bse[regressor_col]),
        ci_lower=float(ci.iloc[0]),
        ci_upper=float(ci.iloc[1]),
        p_value=float(result.pvalues[regressor_col]),
        pseudo_r2=float(result.prsquared),
        adjusted_pseudo_r2=float(adjusted_pseudo_r2),
        n_obs=int(result.nobs),
    )
```

- [ ] **Step 4: Run test to verify it passes, then run full suite**

Run: `pytest tests/test_regression_engine.py -v`
Expected: PASS

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/econometrics/regression_engine.py tests/test_regression_engine.py
git commit -m "feat: add fit_clustered_logit, the shared regression engine every H1-H5 hypothesis uses"
```

---

### Task 4: Shared CARA-`a` join helper + H1 dataset/regression

**Files:**
- Create: `src/econometrics/hypothesis_datasets.py`
- Create: `src/econometrics/hypothesis_regressions.py`
- Test: `tests/test_hypothesis_h1.py`

**Interfaces:**
- Consumes: `database.models.LLMDecisionRecord`, `database.models.AgentStateRecord`, `src.currencies.currency.load_currency_universe`, `src.economy.fx_tax.currency_zone_of`, `src.econometrics.cell_identity.cell_key_from_run_id`, `sqlalchemy.orm.Session`.
- Produces: `src.econometrics.hypothesis_datasets._join_cara_a(session: Session, df: pd.DataFrame) -> pd.DataFrame` (shared by H1/H2/H3), `build_h1_dataset(session: Session) -> pd.DataFrame` (columns: `agent_id`, `chose_usd_zone`, `cara_a`, `agent_type`, `actual_model`), `src.econometrics.hypothesis_regressions.regress_h1(session: Session) -> RegressionResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hypothesis_h1.py`:

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.econometrics.hypothesis_datasets import build_h1_dataset
from src.econometrics.hypothesis_regressions import regress_h1
from src.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 5, seeds: list[int] | None = None) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=seeds or [0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
        keep_daily_results=False,
    )
    return session


def test_build_h1_dataset_only_includes_master_cell_decisions():
    session = _populated_session()
    df = build_h1_dataset(session)

    assert not df.empty
    assert set(df.columns) >= {"agent_id", "chose_usd_zone", "cara_a", "agent_type", "actual_model"}
    assert df["chose_usd_zone"].isin([0, 1]).all()


def test_build_h1_dataset_excludes_gold_backed_decisions():
    session = _populated_session()
    df = build_h1_dataset(session)
    # PAXG/XAUT (gold-backed, peg=None per CurrencyConfig -- see
    # src.economy.fx_tax.currency_zone_of) must never appear as a
    # dependent-variable observation -- H1 is a USD-vs-EUR contrast only.
    assert df["chose_usd_zone"].isin([0, 1]).all()


def test_regress_h1_returns_a_regression_result():
    session = _populated_session(num_days=10)
    result = regress_h1(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H1"
    assert result.regressor == "cara_a"
    assert result.n_obs > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_h1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.econometrics.hypothesis_datasets'`

- [ ] **Step 3: Implement `hypothesis_datasets.py` (shared join helper + H1)**

Create `src/econometrics/hypothesis_datasets.py`:

```python
"""Per-hypothesis dataset builders: each function returns a `pandas.
DataFrame`, one row per eligible `LLMDecisionRecord`, ready for `src.
econometrics.regression_engine.fit_clustered_logit`. See
docs/superpowers/specs/2026-08-02-phase3-plan5-econometrics-design.md
Sec 1 for the exact per-hypothesis data-source/dependent-variable/
regressor design this implements.
"""

import pandas as pd
from sqlalchemy.orm import Session

from database.models import AgentStateRecord, LLMDecisionRecord
from src.currencies.currency import load_currency_universe
from src.econometrics.cell_identity import cell_key_from_run_id
from src.economy.fx_tax import currency_zone_of

_DECIDED_ACTIONS = ("ACCEPT", "OFFER")


def _join_cara_a(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Joins each row's agent's CARA `a` AT THAT DECISION'S timestep from
    `AgentStateRecord` (matched on run_id/timestep/agent_id) -- the correct
    source per the design spec Sec 1 (NOT `LLMDecisionRecord.utility_
    parameters`, which omits risk-neutral agents' `a=0.0` entirely).
    `df` must already have `run_id`/`timestep`/`agent_id` columns. Rows
    with no matching `AgentStateRecord` (shouldn't happen in practice --
    every persisted day writes one per agent -- but defensively dropped
    rather than silently coerced) are excluded.
    """
    if df.empty:
        return df.assign(cara_a=pd.Series(dtype=float))

    run_ids = df["run_id"].unique().tolist()
    states = (
        session.query(
            AgentStateRecord.run_id,
            AgentStateRecord.timestep,
            AgentStateRecord.agent_id,
            AgentStateRecord.cara_coefficient,
        )
        .filter(AgentStateRecord.run_id.in_(run_ids))
        .all()
    )
    states_df = pd.DataFrame(states, columns=["run_id", "timestep", "agent_id", "cara_a"])
    merged = df.merge(states_df, on=["run_id", "timestep", "agent_id"], how="left")
    return merged.dropna(subset=["cara_a"]).reset_index(drop=True)


def build_h1_dataset(session: Session) -> pd.DataFrame:
    """H1: higher CARA `a` -> stronger preference for USD-zone stablecoins
    over EUR-zone. Master cell only (the only cell with real currency-zone
    variation). Gold-backed/zone-neutral decisions (currency_zone_of
    returns None) are excluded -- H1 is a USD-vs-EUR contrast only."""
    currencies = load_currency_universe()

    decisions = (
        session.query(LLMDecisionRecord)
        .filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
        .all()
    )

    records = []
    for decision in decisions:
        if cell_key_from_run_id(decision.simulation_id) != "master":
            continue
        currency = currencies.get(decision.currency)
        if currency is None:
            continue
        zone = currency_zone_of(currency)
        if zone is None:
            continue
        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_usd_zone": 1 if zone == "USD" else 0,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
            }
        )

    df = pd.DataFrame.from_records(
        records, columns=["run_id", "timestep", "agent_id", "chose_usd_zone", "agent_type", "actual_model"]
    )
    return _join_cara_a(session, df)
```

- [ ] **Step 4: Implement `hypothesis_regressions.py` (H1 only for now)**

Create `src/econometrics/hypothesis_regressions.py`:

```python
"""One function per hypothesis: builds that hypothesis's dataset (`src.
econometrics.hypothesis_datasets`) and fits it (`src.econometrics
.regression_engine.fit_clustered_logit`), per docs/superpowers/specs/
2026-08-02-phase3-plan5-econometrics-design.md.
"""

from sqlalchemy.orm import Session

from src.econometrics.hypothesis_datasets import build_h1_dataset
from src.econometrics.regression_engine import RegressionResult, fit_clustered_logit


def regress_h1(session: Session) -> RegressionResult:
    df = build_h1_dataset(session)
    return fit_clustered_logit(
        hypothesis="H1",
        df=df,
        dependent_col="chose_usd_zone",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )
```

- [ ] **Step 5: Run test to verify it passes, then run full suite**

Run: `pytest tests/test_hypothesis_h1.py -v`
Expected: PASS (may take ~5-15s: `_populated_session` runs a real, tiny `run_matrix(dry_run=True, exercise_llm_path=True)` call across all 13 cells)

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/hypothesis_datasets.py src/econometrics/hypothesis_regressions.py tests/test_hypothesis_h1.py
git commit -m "feat: add H1 dataset builder and regression (CARA a -> USD-vs-EUR stablecoin preference)"
```

---

### Task 5: H2 dataset/regression (spread-vs-gas tradeoff sample)

**Files:**
- Modify: `src/econometrics/hypothesis_datasets.py` (add `build_h2_dataset`)
- Modify: `src/econometrics/hypothesis_regressions.py` (add `regress_h2`)
- Test: `tests/test_hypothesis_h2.py`

**Interfaces:**
- Consumes: `LLMDecisionRecord.spread_optimal_currency`/`.spread_optimal_chain`/`.gas_optimal_currency`/`.gas_optimal_chain` (Task 1), `LLMDecisionRecord.currency`/`.chain` (the chosen option).
- Produces: `build_h2_dataset(session: Session) -> pd.DataFrame` (columns: `agent_id`, `chose_spread_optimal`, `cara_a`, `agent_type`, `actual_model`), `regress_h2(session: Session) -> RegressionResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hypothesis_h2.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.econometrics.hypothesis_datasets import build_h2_dataset
from src.econometrics.hypothesis_regressions import regress_h2
from src.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 10) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
    )
    return session


def test_build_h2_dataset_only_includes_genuine_tradeoff_decisions():
    session = _populated_session()
    df = build_h2_dataset(session)

    assert set(df.columns) >= {"agent_id", "chose_spread_optimal", "cara_a", "agent_type", "actual_model"}
    assert df["chose_spread_optimal"].isin([0, 1]).all()


def test_regress_h2_returns_a_regression_result():
    session = _populated_session(num_days=15)
    result = regress_h2(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H2"
    assert result.regressor == "cara_a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_h2.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_h2_dataset'`

- [ ] **Step 3: Implement `build_h2_dataset`**

Add to `src/econometrics/hypothesis_datasets.py`:

```python
def build_h2_dataset(session: Session) -> pd.DataFrame:
    """H2: higher CARA `a` -> prioritizes low spread (liquidity_score, the
    codebase's spread proxy) over low gas fees. Master cell only. Keeps
    only decisions where the round's spread-optimal and gas-optimal
    candidates DIFFERED (a genuine tradeoff existed) AND the agent's
    actual choice matches one of those two candidates -- per the design
    spec Sec 2's resolved tradeoff-sample design.
    """
    decisions = (
        session.query(LLMDecisionRecord)
        .filter(
            LLMDecisionRecord.action.in_(_DECIDED_ACTIONS),
            LLMDecisionRecord.spread_optimal_currency.isnot(None),
            LLMDecisionRecord.spread_optimal_currency != "",
        )
        .all()
    )

    records = []
    for decision in decisions:
        if cell_key_from_run_id(decision.simulation_id) != "master":
            continue
        spread_optimal = (decision.spread_optimal_currency, decision.spread_optimal_chain)
        gas_optimal = (decision.gas_optimal_currency, decision.gas_optimal_chain)
        if spread_optimal == gas_optimal:
            continue  # no genuine tradeoff this round
        chosen = (decision.currency, decision.chain)
        if chosen not in (spread_optimal, gas_optimal):
            continue  # chose neither optimal option -- ambiguous, excluded
        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_spread_optimal": 1 if chosen == spread_optimal else 0,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
            }
        )

    df = pd.DataFrame.from_records(
        records, columns=["run_id", "timestep", "agent_id", "chose_spread_optimal", "agent_type", "actual_model"]
    )
    return _join_cara_a(session, df)
```

- [ ] **Step 4: Implement `regress_h2`**

Add to `src/econometrics/hypothesis_regressions.py` (update the top import line to include `build_h2_dataset`):

```python
from src.econometrics.hypothesis_datasets import build_h1_dataset, build_h2_dataset
```

```python
def regress_h2(session: Session) -> RegressionResult:
    df = build_h2_dataset(session)
    return fit_clustered_logit(
        hypothesis="H2",
        df=df,
        dependent_col="chose_spread_optimal",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )
```

- [ ] **Step 5: Run test to verify it passes, then run full suite**

Run: `pytest tests/test_hypothesis_h2.py -v`
Expected: PASS. **If this fails with a `ValueError` from `fit_clustered_logit` (empty DataFrame)**, the tiny test fixture's `num_days` may not produce enough genuine spread-vs-gas tradeoffs — increase `num_days` in the test (more simulated days -> more decisions -> more chance of a genuine tradeoff) rather than changing the dataset-builder logic.

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/hypothesis_datasets.py src/econometrics/hypothesis_regressions.py tests/test_hypothesis_h2.py
git commit -m "feat: add H2 dataset builder and regression (CARA a -> spread-vs-gas-fee tradeoff)"
```

---

### Task 6: H3 dataset/regression (liquidity_vs_governance sandbox, pooled domestic+cross-border)

**Files:**
- Modify: `src/econometrics/hypothesis_datasets.py` (add `build_h3_dataset`)
- Modify: `src/econometrics/hypothesis_regressions.py` (add `regress_h3`)
- Test: `tests/test_hypothesis_h3.py`

**Interfaces:**
- Consumes: `src.currencies.sandbox_currencies.SANDBOX_CURRENCY_PAIRS` (dict: sandbox name -> `(option_a: CurrencyConfig, option_b: CurrencyConfig)`), `cell_key_from_run_id`.
- Produces: `build_h3_dataset(session: Session) -> pd.DataFrame` (columns: `agent_id`, `chose_higher_governance`, `cara_a`, `agent_type`, `actual_model`, `cell_key` -- `cell_key` is the fixed effect distinguishing domestic from cross-border), `regress_h3(session: Session) -> RegressionResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hypothesis_h3.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.econometrics.hypothesis_datasets import build_h3_dataset
from src.econometrics.hypothesis_regressions import regress_h3
from src.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 10) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
    )
    return session


def test_build_h3_dataset_only_includes_the_liquidity_vs_governance_sandbox():
    session = _populated_session()
    df = build_h3_dataset(session)

    assert not df.empty
    assert set(df.columns) >= {
        "agent_id", "chose_higher_governance", "cara_a", "agent_type", "actual_model", "cell_key",
    }
    assert set(df["cell_key"].unique()) <= {
        "liquidity_vs_governance_domestic", "liquidity_vs_governance_cross_border",
    }
    assert df["chose_higher_governance"].isin([0, 1]).all()


def test_regress_h3_returns_a_regression_result():
    session = _populated_session(num_days=15)
    result = regress_h3(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H3"
    assert result.regressor == "cara_a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_h3.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_h3_dataset'`

- [ ] **Step 3: Implement `build_h3_dataset`**

Add to `src/econometrics/hypothesis_datasets.py` (add the import at the top: `from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS`):

```python
_H3_SANDBOX_KEY = "liquidity_vs_governance"
_H3_CELLS = {f"{_H3_SANDBOX_KEY}_domestic", f"{_H3_SANDBOX_KEY}_cross_border"}


def build_h3_dataset(session: Session) -> pd.DataFrame:
    """H3: higher CARA `a` -> prioritizes GENIUS Act compliance/governance
    over liquidity. The `liquidity_vs_governance` sandbox (domestic +
    cross-border pooled, with `cell_key` as a fixed effect distinguishing
    the two -- see design spec Sec 1)."""
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[_H3_SANDBOX_KEY]
    higher_governance_symbol = (
        option_a.symbol if option_a.governance_score >= option_b.governance_score else option_b.symbol
    )

    decisions = (
        session.query(LLMDecisionRecord)
        .filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
        .all()
    )

    records = []
    for decision in decisions:
        cell_key = cell_key_from_run_id(decision.simulation_id)
        if cell_key not in _H3_CELLS:
            continue
        if decision.currency not in (option_a.symbol, option_b.symbol):
            continue
        records.append(
            {
                "run_id": decision.simulation_id,
                "timestep": decision.timestep,
                "agent_id": decision.agent_id,
                "chose_higher_governance": 1 if decision.currency == higher_governance_symbol else 0,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
                "cell_key": cell_key,
            }
        )

    df = pd.DataFrame.from_records(
        records,
        columns=["run_id", "timestep", "agent_id", "chose_higher_governance", "agent_type", "actual_model", "cell_key"],
    )
    return _join_cara_a(session, df)
```

- [ ] **Step 4: Implement `regress_h3`**

Add to `src/econometrics/hypothesis_regressions.py` (update the import line: `from src.econometrics.hypothesis_datasets import build_h1_dataset, build_h2_dataset, build_h3_dataset`):

```python
def regress_h3(session: Session) -> RegressionResult:
    df = build_h3_dataset(session)
    return fit_clustered_logit(
        hypothesis="H3",
        df=df,
        dependent_col="chose_higher_governance",
        regressor_col="cara_a",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model", "cell_key"],
    )
```

- [ ] **Step 5: Run test to verify it passes, then run full suite**

Run: `pytest tests/test_hypothesis_h3.py -v`
Expected: PASS

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/hypothesis_datasets.py src/econometrics/hypothesis_regressions.py tests/test_hypothesis_h3.py
git commit -m "feat: add H3 dataset builder and regression (CARA a -> governance-vs-liquidity preference)"
```

---

### Task 7: H4 dataset/regression (crisis proximity -> gold preference)

**Files:**
- Modify: `src/econometrics/hypothesis_datasets.py` (add `build_h4_dataset`)
- Modify: `src/econometrics/hypothesis_regressions.py` (add `regress_h4`)
- Test: `tests/test_hypothesis_h4.py`

**Interfaces:**
- Consumes: `database.models.InterventionLogRecord` (fields: `run_id`, `timestep`, `shock_type`, `target_currency`), `src.economy.shocks.ShockType.CRISIS_WARNING`/`.DEPEG_EVENT` (values `"crisis_warning"`/`"depeg_event"`), `SANDBOX_CURRENCY_PAIRS`.
- Produces: `build_h4_dataset(session: Session) -> pd.DataFrame` (columns: `agent_id`, `chose_gold`, `proximity_days`, `agent_type`, `actual_model`, `cell_key`), `regress_h4(session: Session) -> RegressionResult`. Note: H4's regressor is `proximity_days`, NOT CARA `a` -- this hypothesis does not call `_join_cara_a`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hypothesis_h4.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.econometrics.hypothesis_datasets import build_h4_dataset
from src.econometrics.hypothesis_regressions import regress_h4
from src.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 30) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
    )
    return session


def test_build_h4_dataset_only_includes_gold_backed_sandboxes():
    session = _populated_session()
    df = build_h4_dataset(session)

    assert not df.empty
    assert set(df.columns) >= {
        "agent_id", "chose_gold", "proximity_days", "agent_type", "actual_model", "cell_key",
    }
    assert set(df["cell_key"].unique()) <= {
        "asset_backing_vs_liquidity_domestic", "asset_backing_vs_liquidity_cross_border",
        "asset_backing_vs_stability_domestic", "asset_backing_vs_stability_cross_border",
    }
    assert df["chose_gold"].isin([0, 1]).all()


def test_regress_h4_returns_a_regression_result():
    session = _populated_session(num_days=40)
    result = regress_h4(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H4"
    assert result.regressor == "proximity_days"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_h4.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_h4_dataset'`

- [ ] **Step 3: Implement `build_h4_dataset`**

Add to `src/econometrics/hypothesis_datasets.py` (add imports at the top: `from database.models import InterventionLogRecord`):

```python
_H4_SANDBOX_KEYS = ("asset_backing_vs_liquidity", "asset_backing_vs_stability")
_H4_CELLS = {f"{key}_{suffix}" for key in _H4_SANDBOX_KEYS for suffix in ("domestic", "cross_border")}
_H4_PROXIMITY_SHOCK_TYPES = ("crisis_warning", "depeg_event")


def _signed_proximity(timestep: int, event_days: list[int]) -> int:
    """Signed distance (in days) from `timestep` to the nearest crisis_
    warning/depeg_event day for this run: negative = approaching (before
    the event), positive = past (after it)."""
    nearest = min(event_days, key=lambda day: abs(day - timestep))
    return timestep - nearest


def build_h4_dataset(session: Session) -> pd.DataFrame:
    """H4: closer crisis/depeg proximity -> stronger shift to gold-backed
    tokens. The two sandboxes with a gold option (asset_backing_vs_
    liquidity, asset_backing_vs_stability), domestic + cross-border
    pooled with `cell_key` as a fixed effect. `proximity_days` is signed
    (negative = approaching, positive = past the nearest crisis_warning/
    depeg_event) -- see design spec Sec 0's continuous-proximity decision.
    """
    gold_symbols = {
        cfg.symbol
        for sandbox_key in _H4_SANDBOX_KEYS
        for cfg in SANDBOX_CURRENCY_PAIRS[sandbox_key]
        if cfg.peg == "XAU"
    }

    decisions = (
        session.query(LLMDecisionRecord)
        .filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
        .all()
    )

    relevant_run_ids = {
        decision.simulation_id
        for decision in decisions
        if cell_key_from_run_id(decision.simulation_id) in _H4_CELLS
    }
    if not relevant_run_ids:
        return pd.DataFrame(columns=["agent_id", "chose_gold", "proximity_days", "agent_type", "actual_model", "cell_key"])

    intervention_rows = (
        session.query(InterventionLogRecord.run_id, InterventionLogRecord.timestep)
        .filter(
            InterventionLogRecord.run_id.in_(relevant_run_ids),
            InterventionLogRecord.shock_type.in_(_H4_PROXIMITY_SHOCK_TYPES),
        )
        .all()
    )
    event_days_by_run: dict[str, list[int]] = {}
    for run_id, timestep in intervention_rows:
        event_days_by_run.setdefault(run_id, []).append(timestep)

    records = []
    for decision in decisions:
        cell_key = cell_key_from_run_id(decision.simulation_id)
        if cell_key not in _H4_CELLS:
            continue
        event_days = event_days_by_run.get(decision.simulation_id)
        if not event_days:
            continue  # this cell/seed's data has no crisis/depeg event at all -- no proximity to measure
        records.append(
            {
                "agent_id": decision.agent_id,
                "chose_gold": 1 if decision.currency in gold_symbols else 0,
                "proximity_days": _signed_proximity(decision.timestep, event_days),
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
                "cell_key": cell_key,
            }
        )

    return pd.DataFrame.from_records(
        records, columns=["agent_id", "chose_gold", "proximity_days", "agent_type", "actual_model", "cell_key"]
    )
```

- [ ] **Step 4: Implement `regress_h4`**

Add to `src/econometrics/hypothesis_regressions.py` (update the import line: `from src.econometrics.hypothesis_datasets import (build_h1_dataset, build_h2_dataset, build_h3_dataset, build_h4_dataset)`):

```python
def regress_h4(session: Session) -> RegressionResult:
    df = build_h4_dataset(session)
    return fit_clustered_logit(
        hypothesis="H4",
        df=df,
        dependent_col="chose_gold",
        regressor_col="proximity_days",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model", "cell_key"],
    )
```

- [ ] **Step 5: Run test to verify it passes, then run full suite**

Run: `pytest tests/test_hypothesis_h4.py -v`
Expected: PASS. **If `test_build_h4_dataset_only_includes_gold_backed_sandboxes` fails with an empty DataFrame**, it means the tiny dry-run test fixture's synthetic sandbox scenario (built by `build_sandbox_scenario`) didn't fire a `crisis_warning`/`depeg_event` shock within `num_days` — check `src/economy/sandbox_scenarios.py`'s `_CRISIS_WARNING_DAY`/`_DEPEG_GAP_DAYS` constants and raise this test's `num_days` past that day number (e.g. `_CRISIS_WARNING_DAY + _DEPEG_GAP_DAYS + 5` or more) rather than changing the dataset-builder logic.

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/hypothesis_datasets.py src/econometrics/hypothesis_regressions.py tests/test_hypothesis_h4.py
git commit -m "feat: add H4 dataset builder and regression (crisis/depeg proximity -> gold-backed preference)"
```

---

### Task 8: H5 dataset/regression (EUR/USD volatility -> USD preference, cross-zone pairs)

**Files:**
- Modify: `src/econometrics/hypothesis_datasets.py` (add `build_h5_dataset`)
- Modify: `src/econometrics/hypothesis_regressions.py` (add `regress_h5`)
- Test: `tests/test_hypothesis_h5.py`

**Interfaces:**
- Consumes: `database.models.TimestepLogRecord` (fields: `run_id`, `timestep`, `eur_usd_exchange_rate`), `database.models.AgentRecord` (field: `currency_zone`), `src.currencies.currency.load_currency_universe`, `src.economy.fx_tax.currency_zone_of`.
- Produces: `build_h5_dataset(session: Session) -> pd.DataFrame` (columns: `agent_id`, `chose_usd_zone`, `eur_usd_volatility`, `agent_type`, `actual_model`), `regress_h5(session: Session) -> RegressionResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hypothesis_h5.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.econometrics.hypothesis_datasets import build_h5_dataset
from src.econometrics.hypothesis_regressions import regress_h5
from src.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 30) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
    )
    return session


def test_build_h5_dataset_only_includes_zone_mismatched_master_decisions():
    session = _populated_session()
    df = build_h5_dataset(session)

    assert set(df.columns) >= {
        "agent_id", "chose_usd_zone", "eur_usd_volatility", "agent_type", "actual_model",
    }
    assert df["chose_usd_zone"].isin([0, 1]).all()


def test_regress_h5_returns_a_regression_result():
    session = _populated_session(num_days=40)
    result = regress_h5(session)
    assert isinstance(result, RegressionResult)
    assert result.hypothesis == "H5"
    assert result.regressor == "eur_usd_volatility"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hypothesis_h5.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_h5_dataset'`

- [ ] **Step 3: Implement `build_h5_dataset`**

Add to `src/econometrics/hypothesis_datasets.py` (add imports at the top: `from database.models import AgentRecord, TimestepLogRecord`):

```python
_H5_VOLATILITY_WINDOW_DAYS = 30  # trailing window for realized EUR/USD volatility -- see design spec Sec 1


def _rolling_volatility(rates_by_day: dict[int, float], day: int, window: int) -> float | None:
    """Sample standard deviation of `eur_usd_exchange_rate` over the
    `window` days up to and including `day`. Returns None if fewer than 2
    days of history exist yet (std of a single point is undefined)."""
    window_days = [d for d in rates_by_day if day - window < d <= day]
    if len(window_days) < 2:
        return None
    values = pd.Series([rates_by_day[d] for d in window_days])
    return float(values.std(ddof=1))


def build_h5_dataset(session: Session) -> pd.DataFrame:
    """H5: higher EUR/USD volatility -> stronger preference for USD-zone
    stablecoins in cross-border settlement. Master cell only, filtered to
    decisions by an agent whose OWN currency_zone differs from at least
    one plausible counterparty's (master's pairing is zone-agnostic, so
    cross-zone pairs occur naturally in a 50/50 USD/EUR population) --
    approximated here as: the deciding agent's currency_zone is set (not
    None, i.e. not a legacy count-based agent), since `LLMDecisionRecord`
    does not persist the counterparty's zone directly (see design spec
    Sec 1 -- same underlying gap as H1's zone lookup)."""
    currencies = load_currency_universe()

    decisions = (
        session.query(LLMDecisionRecord)
        .filter(LLMDecisionRecord.action.in_(_DECIDED_ACTIONS))
        .all()
    )
    master_decisions = [d for d in decisions if cell_key_from_run_id(d.simulation_id) == "master"]
    if not master_decisions:
        return pd.DataFrame(columns=["agent_id", "chose_usd_zone", "eur_usd_volatility", "agent_type", "actual_model"])

    run_ids = {d.simulation_id for d in master_decisions}
    agent_ids = {d.agent_id for d in master_decisions}
    agent_zones = dict(
        session.query(AgentRecord.id, AgentRecord.currency_zone).filter(AgentRecord.id.in_(agent_ids)).all()
    )

    timestep_rows = (
        session.query(TimestepLogRecord.run_id, TimestepLogRecord.timestep, TimestepLogRecord.eur_usd_exchange_rate)
        .filter(TimestepLogRecord.run_id.in_(run_ids))
        .all()
    )
    rates_by_run: dict[str, dict[int, float]] = {}
    for run_id, timestep, rate in timestep_rows:
        rates_by_run.setdefault(run_id, {})[timestep] = rate

    records = []
    for decision in master_decisions:
        if agent_zones.get(decision.agent_id) is None:
            continue  # legacy count-based agent, no zone -- excluded
        currency = currencies.get(decision.currency)
        if currency is None:
            continue
        zone = currency_zone_of(currency)
        if zone is None:
            continue
        volatility = _rolling_volatility(
            rates_by_run.get(decision.simulation_id, {}), decision.timestep, _H5_VOLATILITY_WINDOW_DAYS
        )
        if volatility is None:
            continue
        records.append(
            {
                "agent_id": decision.agent_id,
                "chose_usd_zone": 1 if zone == "USD" else 0,
                "eur_usd_volatility": volatility,
                "agent_type": decision.agent_type,
                "actual_model": decision.actual_model,
            }
        )

    return pd.DataFrame.from_records(
        records, columns=["agent_id", "chose_usd_zone", "eur_usd_volatility", "agent_type", "actual_model"]
    )
```

- [ ] **Step 4: Implement `regress_h5`**

Add to `src/econometrics/hypothesis_regressions.py` (update the import line to include `build_h5_dataset`):

```python
def regress_h5(session: Session) -> RegressionResult:
    df = build_h5_dataset(session)
    return fit_clustered_logit(
        hypothesis="H5",
        df=df,
        dependent_col="chose_usd_zone",
        regressor_col="eur_usd_volatility",
        cluster_col="agent_id",
        fixed_effect_cols=["agent_type", "actual_model"],
    )
```

- [ ] **Step 5: Run test to verify it passes, then run full suite**

Run: `pytest tests/test_hypothesis_h5.py -v`
Expected: PASS. **If `test_regress_h5_returns_a_regression_result` fails with an empty-DataFrame `ValueError`**, the 30-day rolling window (`_H5_VOLATILITY_WINDOW_DAYS`) may exceed the tiny test fixture's `num_days` for every decision to have 2+ days of prior history — raise the test's `num_days` (e.g. to 60) rather than changing the window constant.

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/econometrics/hypothesis_datasets.py src/econometrics/hypothesis_regressions.py tests/test_hypothesis_h5.py
git commit -m "feat: add H5 dataset builder and regression (EUR/USD volatility -> USD-zone preference)"
```

---

### Task 9: Report assembly (all 5 hypotheses, one output table)

**Files:**
- Create: `src/econometrics/report.py`
- Test: `tests/test_econometrics_report.py`

**Interfaces:**
- Consumes: `regress_h1`/`regress_h2`/`regress_h3`/`regress_h4`/`regress_h5` (Tasks 4-8), `src.econometrics.regression_engine.RegressionResult`.
- Produces: `run_all_hypotheses(session: Session) -> list[RegressionResult]`, `results_to_dataframe(results: list[RegressionResult]) -> pd.DataFrame`, `write_report_csv(results: list[RegressionResult], path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_econometrics_report.py`:

```python
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.econometrics.regression_engine import RegressionResult
from src.econometrics.report import results_to_dataframe, run_all_hypotheses, write_report_csv
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(num_days: int = 40) -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
    )
    return session


def test_run_all_hypotheses_returns_one_result_per_hypothesis():
    session = _populated_session()
    results = run_all_hypotheses(session)

    assert len(results) == 5
    assert {r.hypothesis for r in results} == {"H1", "H2", "H3", "H4", "H5"}
    assert all(isinstance(r, RegressionResult) for r in results)


def test_results_to_dataframe_has_the_required_publication_columns():
    session = _populated_session()
    results = run_all_hypotheses(session)
    df = results_to_dataframe(results)

    assert set(df.columns) >= {
        "hypothesis", "regressor", "beta", "se", "ci_lower", "ci_upper",
        "p_value", "pseudo_r2", "adjusted_pseudo_r2", "n_obs",
    }
    assert len(df) == 5


def test_write_report_csv_writes_a_readable_file(tmp_path):
    session = _populated_session()
    results = run_all_hypotheses(session)
    out_path = tmp_path / "hypothesis_report.csv"

    write_report_csv(results, out_path)

    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == 5
    assert set(reloaded["hypothesis"]) == {"H1", "H2", "H3", "H4", "H5"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_econometrics_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.econometrics.report'`

- [ ] **Step 3: Implement**

Create `src/econometrics/report.py`:

```python
"""Assembles all 5 in-scope hypotheses' regression results (H6 is
deferred, per docs/superpowers/specs/2026-07-29-phase3-full-scale-
simulation-design.md Sec 7) into one output table.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.econometrics.hypothesis_regressions import regress_h1, regress_h2, regress_h3, regress_h4, regress_h5
from src.econometrics.regression_engine import RegressionResult


def run_all_hypotheses(session: Session) -> list[RegressionResult]:
    """Runs every in-scope hypothesis's regression against `session`'s
    already-persisted matrix-runner data. Each hypothesis's own dataset
    builder independently filters to its own relevant cell(s) -- a
    hypothesis whose sample turns out empty raises `ValueError` from
    `fit_clustered_logit` rather than silently omitting itself, so a
    misconfigured run surfaces loudly instead of shipping a report
    missing a hypothesis with no explanation.
    """
    return [
        regress_h1(session),
        regress_h2(session),
        regress_h3(session),
        regress_h4(session),
        regress_h5(session),
    ]


def results_to_dataframe(results: list[RegressionResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis": r.hypothesis,
                "regressor": r.regressor,
                "beta": r.beta,
                "se": r.se,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "p_value": r.p_value,
                "pseudo_r2": r.pseudo_r2,
                "adjusted_pseudo_r2": r.adjusted_pseudo_r2,
                "n_obs": r.n_obs,
            }
            for r in results
        ]
    )


def write_report_csv(results: list[RegressionResult], path: Path) -> None:
    results_to_dataframe(results).to_csv(path, index=False)
```

- [ ] **Step 4: Run test to verify it passes, then run the full suite one final time for this plan**

Run: `pytest tests/test_econometrics_report.py -v`
Expected: PASS

Run: `pytest -q`
Expected: all tests pass, including every test added across Tasks 1-9.

- [ ] **Step 5: Commit**

```bash
git add src/econometrics/report.py tests/test_econometrics_report.py
git commit -m "feat: add run_all_hypotheses report assembly, producing one output table for all 5 in-scope hypotheses"
```

---

## What comes after this plan

1. **Streamlit dashboard** (`dashboard/app.py`) — deferred per master spec §7; a later viewer over `src.econometrics.report`'s output, not part of this plan.
2. **Full-scale run launch** — still the same explicit, separate go/no-go checkpoint with the user described in Plan 4's plan; this plan's Task 1 (persisting spread/gas-optimal fields) must land and be reviewed/merged BEFORE that launch, or H2 will have no usable data once the real run starts.
3. **H6 privacy sandbox** — still explicitly out of scope (master spec §7), no privacy-rail currency/chain config exists.
