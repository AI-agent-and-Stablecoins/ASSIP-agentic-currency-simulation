# Phase 3 Plan 4: Matrix Runner / Experiment Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement everything in `docs/superpowers/specs/2026-07-31-phase3-plan4-matrix-runner-design.md`: real LLM-driven decisions + full LLM-vs-LLM negotiation in the day loop, live Polygon price context, `CurrencyHistory`/`MacroHistory` auto-population, the cross-border FX tax, loss-driven CARA adaptation, synthetic sandbox currencies, full per-timestep persistence, run provenance, the 365-day master scenario, and the 13-cell x 5-seed matrix runner.

**Architecture:** Additive throughout — every existing entry point (`Environment.build`, `SimulationRunner`, `run_timestep`'s deterministic path, `persist_timestep`) keeps working unchanged for existing callers/tests. New capability is reached via new optional parameters (`use_llm`, `dry_run`) defaulting to today's behavior, or new sibling functions/classmethods, never by mutating an existing signature's meaning.

**Tech Stack:** Pydantic >=2.6, SQLAlchemy >=2.0, httpx >=0.27 (now required for this plan's new modules — install via `pip install -e ".[llm]"` before starting; existing core-only callers remain unaffected since they never import the new LLM-wired modules). No new dependencies beyond what `pyproject.toml`'s `llm` extra already declares.

## Global Constraints

- Python >=3.12, Pydantic >=2.6, SQLAlchemy >=2.0 — no new dependencies without checking with the user first.
- No hardcoded economic constants: `fx_tax_rate` lives in `configs/economy/fx_params.yaml`; `eta_risk`/`a_max` live in `configs/economy/risk_adaptation_params.yaml`. Never hardcoded in `src/`.
- Every existing test must keep passing unmodified. New optional parameters (`use_llm: bool = False` on `run_timestep`, `dry_run: bool = True` on the matrix runner) must default to preserving current behavior exactly.
- `dry_run=True` is the only mode any test in this plan may use — every new test that exercises LLM-calling or Polygon-calling code paths must construct its own `httpx.MockTransport` client (following the existing per-test convention — see Task 1's shared test helper, which centralizes this instead of hand-rolling it per file for the first time in this codebase).
- Per Plan 3's per-agent-fixed-model design: LLM calls in this plan use exactly `agent.assigned_model`, never a fallback chain across different models — a fallback would silently substitute a different agent's model identity.
- The 55 CARA-eligible agents are consumer/bank/investor (per Plan 3's resolved ambiguity, `cara_coefficient is not None`) — only these adapt `a` over time; merchant/institution (`multi_attribute`, `cara_coefficient is None`) never adapt.
- There are 6 factor-isolation sandboxes (not 7) — Liquidity-vs-Governance, Governance-vs-Stability, Liquidity-vs-Stability, Asset-Backing-vs-Liquidity, Asset-Backing-vs-Stability, Asset-Backing-vs-Governance — each run once domestically and once cross-border, plus the master simulation = 13 total experiment cells.
- Do not add scope beyond the design spec and this plan without checking with the user first.
- Task order matters: Tasks 1-2 (test helper, provenance) have no dependents and can be built first; Task 3 (price index/wealth helpers) is needed by Task 6 (adaptation); Task 4 (Decision->Transaction adapter) and Task 5 (LLM day-loop wiring) must land before Task 8 (persistence wiring, which persists LLM decisions) and Task 12 (matrix runner, which drives the whole loop).

---

## File Structure

- **Create:** `tests/llm_test_helpers.py` (shared `MockTransport` client factory)
- **Create:** `src/simulation/provenance.py`
- **Create:** `src/agents/wealth.py` (real purchasing power / price index helpers)
- **Modify:** `src/simulation/environment.py` (`price_index`, `event_log` fields, `build_from_population` classmethod)
- **Create:** `src/llm/decision_to_transaction.py` (Decision/NegotiationAction -> Transaction adapter)
- **Modify:** `src/simulation/timestep.py` (`use_llm` parameter, LLM-driven decision + negotiation path, FX tax, event_log recording)
- **Create:** `configs/economy/fx_params.yaml`, `configs/economy/risk_adaptation_params.yaml`
- **Modify:** `src/transactions/transaction.py` / `src/transactions/settlement.py` (fx_tax_paid debit)
- **Create:** `src/economy/risk_adaptation.py` (loss-driven CARA adaptation)
- **Create:** `src/economy/history_builder.py` (`build_currency_history`, `build_macro_history`)
- **Modify:** `src/llm/agent_reasoning.py` (live-price/history wiring helpers if needed — task decides exact placement)
- **Create:** `src/currencies/sandbox_currencies.py` (6 synthetic currency pairs)
- **Modify:** `database/repository.py` (new `persist_full_timestep` function or extended `persist_timestep`)
- **Create:** `configs/scenarios/master_simulation.yaml`
- **Create:** `src/simulation/matrix_runner.py`
- **Test:** new test files per task, listed within each task

---

### Task 1: Shared LLM/Polygon mock-client test helper

**Files:**
- Create: `tests/llm_test_helpers.py`
- Test: `tests/test_llm_test_helpers.py`

**Context:** every existing LLM test hand-rolls its own `httpx.MockTransport(handler)`. This plan adds many new LLM/Polygon-calling tests; centralize the pattern once here rather than duplicating it across a dozen new test files.

**Interfaces:**
- Produces: `mock_openrouter_client(model_responses: dict[str, dict]) -> httpx.Client` — `model_responses` maps a model_id to the JSON body to return for that model's chat-completion call (matching whatever request shape `llm_router.py`'s `_post_chat_completion` actually sends — read that function first to match its exact request path/body format). `mock_polygon_client(ticker_prices: dict[str, float]) -> httpx.Client` — returns a live-price-shaped response per ticker matching `market_intelligence.fetch_live_price`'s expected response shape (read that function first).

- [ ] **Step 1: Read the exact request/response shapes these helpers must match**

Read `src/llm/llm_router.py`'s `_post_chat_completion`/`call_model` and `src/llm/market_intelligence.py`'s `fetch_live_price`/`build_polygon_client` in full. Also read 2-3 existing test files that already hand-roll a `MockTransport` (`tests/test_llm_router.py`, `tests/test_market_intelligence.py`) to confirm the exact JSON shapes those tests already use successfully — match those exactly, don't invent a new shape.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_llm_test_helpers.py` with tests that use `mock_openrouter_client`/`mock_polygon_client` to make one real call through `llm_router.call_model`/`market_intelligence.fetch_live_price` respectively and confirm the response parses correctly — i.e., these are tests that prove the helpers actually work with the real calling code, not just that they construct an `httpx.Client` object.

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_llm_test_helpers.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 4: Implement the two factory functions**

Implement `tests/llm_test_helpers.py` matching the exact request/response shapes confirmed in Step 1.

- [ ] **Step 5: Run test to verify it passes, then run the full suite**

Run: `pytest tests/test_llm_test_helpers.py -v` then `pytest -q`
Expected: PASS; full suite unaffected (this is a new test-only file).

- [ ] **Step 6: Commit**

```bash
git add tests/llm_test_helpers.py tests/test_llm_test_helpers.py
git commit -m "test: add shared MockTransport client factory for LLM/Polygon tests"
```

---

### Task 2: Provenance helpers

**Files:**
- Create: `src/simulation/provenance.py`
- Test: new `tests/test_provenance.py`

**Interfaces:**
- Produces: `compute_git_commit_hash() -> str`, `compute_config_hash(paths: list[Path]) -> str` (SHA-256 over concatenated file bytes, sorted by path for determinism), `model_roster_summary_for(agents: list[BaseAgent]) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
import subprocess

import pytest

from src.simulation.provenance import compute_config_hash, compute_git_commit_hash, model_roster_summary_for


def test_compute_git_commit_hash_matches_git_rev_parse():
    result = compute_git_commit_hash()
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert result == expected


def test_compute_config_hash_is_deterministic_and_order_independent(tmp_path):
    file_a = tmp_path / "a.yaml"
    file_b = tmp_path / "b.yaml"
    file_a.write_text("key: value\n")
    file_b.write_text("other: 1\n")

    hash_1 = compute_config_hash([file_a, file_b])
    hash_2 = compute_config_hash([file_b, file_a])  # reversed order

    assert hash_1 == hash_2  # sorted internally, order of the input list shouldn't matter
    assert len(hash_1) == 64  # hex-encoded SHA-256


def test_compute_config_hash_changes_when_file_content_changes(tmp_path):
    file_a = tmp_path / "a.yaml"
    file_a.write_text("key: value\n")
    hash_before = compute_config_hash([file_a])

    file_a.write_text("key: different\n")
    hash_after = compute_config_hash([file_a])

    assert hash_before != hash_after


def test_model_roster_summary_for_describes_agent_count_and_model_diversity():
    from src.agents.population import generate_agent_population

    population = generate_agent_population(seed=0, model_candidates=[f"vendor/model-{i}" for i in range(10)])

    summary = model_roster_summary_for(population)

    assert "100" in summary
    assert "10" in summary  # 10 distinct models used
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provenance.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

```python
import hashlib
import subprocess
from pathlib import Path

from src.agents.base_agent import BaseAgent


def compute_git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def compute_config_hash(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def model_roster_summary_for(agents: list[BaseAgent]) -> str:
    distinct_models = {a.assigned_model for a in agents if a.assigned_model is not None}
    return f"{len(agents)} agents across {len(distinct_models)} OpenRouter models"
```

- [ ] **Step 4: Run test to verify it passes, run full suite**

Run: `pytest tests/test_provenance.py -v` then `pytest -q`

- [ ] **Step 5: Commit**

```bash
git add src/simulation/provenance.py tests/test_provenance.py
git commit -m "feat: add run-provenance helpers (git commit hash, config hash, model roster summary)"
```

---

### Task 3: Price index + real purchasing power helpers

**Files:**
- Create: `src/agents/wealth.py`
- Modify: `src/simulation/environment.py` (add `price_index: float = 1.0` field, initialized in `__init__`)
- Test: new `tests/test_wealth.py`, extend `tests/test_simulation.py`

**Interfaces:**
- Produces: `Environment.price_index: float` (starts at `1.0`), `real_purchasing_power(wallet: Wallet, rates: ExchangeRateTable, price_index: float) -> float` (`wallet.total_value_usd(rates) / price_index`), `advance_price_index(price_index: float, inflation_rate: float) -> float` (`price_index * (1 + inflation_rate)`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wealth.py
from src.agents.wallet import Wallet
from src.agents.wealth import advance_price_index, real_purchasing_power
from src.market.exchange_rates import ExchangeRateTable  # confirm exact import path first


def test_real_purchasing_power_divides_nominal_value_by_price_index():
    wallet = Wallet(balances={"USDC": 1000.0})
    rates = ExchangeRateTable({}, {})  # confirm exact constructor signature first; USDC->USD should be ~1:1

    result = real_purchasing_power(wallet, rates, price_index=1.0)

    assert result == pytest.approx(wallet.total_value_usd(rates))


def test_real_purchasing_power_shrinks_as_price_index_rises():
    wallet = Wallet(balances={"USDC": 1000.0})
    rates = ExchangeRateTable({}, {})

    at_baseline = real_purchasing_power(wallet, rates, price_index=1.0)
    after_inflation = real_purchasing_power(wallet, rates, price_index=1.1)

    assert after_inflation < at_baseline


def test_advance_price_index_compounds_daily_inflation():
    index = advance_price_index(1.0, inflation_rate=0.02)
    assert index == pytest.approx(1.02)

    index_2 = advance_price_index(index, inflation_rate=0.02)
    assert index_2 == pytest.approx(1.02 * 1.02)
```

(Check `src/market/exchange_rates.py` — or wherever `ExchangeRateTable` actually lives, confirm via grep first — for its real constructor signature before finalizing these tests; adjust the fixture construction to match, following whatever pattern `tests/test_simulation.py` or similar already uses to build one.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wealth.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `src/agents/wealth.py`**

```python
from src.agents.wallet import Wallet
from src.market.exchange_rates import ExchangeRateTable  # adjust import to actual path


def real_purchasing_power(wallet: Wallet, rates: ExchangeRateTable, price_index: float) -> float:
    return wallet.total_value_usd(rates) / price_index


def advance_price_index(price_index: float, inflation_rate: float) -> float:
    return price_index * (1 + inflation_rate)
```

- [ ] **Step 4: Add `price_index` to Environment**

In `src/simulation/environment.py`'s `__init__`, add `self.price_index: float = 1.0` alongside `self.trust_ledger`/`self.event_log` (Task's own addition — check whether Task in this same plan responsible for `event_log` has landed first per task order; if not yet, add just `price_index` here and let a later task add `event_log`).

Add a test to `tests/test_simulation.py`:
```python
def test_environment_starts_with_price_index_of_one():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    assert env.price_index == 1.0
```

- [ ] **Step 5: Run tests, then full suite**

Run: `pytest tests/test_wealth.py tests/test_simulation.py -v` then `pytest -q`

- [ ] **Step 6: Commit**

```bash
git add src/agents/wealth.py src/simulation/environment.py tests/test_wealth.py tests/test_simulation.py
git commit -m "feat: add price-index tracker and real-purchasing-power helper"
```

---

### Task 4: Decision/NegotiationAction -> Transaction adapter

**Files:**
- Create: `src/llm/decision_to_transaction.py`
- Test: new `tests/test_decision_to_transaction.py`

**Context:** `adapt_decision` (existing) stops at `NegotiationAction`. Nothing turns an accepted `NegotiationSession`'s final offer into a settleable `Transaction` matching a `CurrencyChainOption` from `generate_candidates`. This task builds that glue, plus the tightened anti-hallucination check (validate against the exact candidate list offered, not the full currency/chain universe).

**Interfaces:**
- Produces: `build_transaction_from_negotiation(session: NegotiationSession, candidates: list[CurrencyChainOption], buyer_id: str, seller_id: str, good_name: str, day: int) -> Transaction | None` (returns `None` if the session didn't end in ACCEPT, or raises/returns `None` if the accepted currency/chain doesn't match any candidate — task decides exact error-vs-None contract after reading `NegotiationSession`'s exact final-state shape).

- [ ] **Step 1: Read `NegotiationSession`/`LLMOffer` in full**

Read `src/negotiation/llm_negotiation_engine.py` and `src/negotiation/llm_offer.py` in full to confirm the exact fields available on a completed session (status, current_offer, conversation_history) before writing the adapter — do not guess field names.

- [ ] **Step 2: Write the failing tests**

Construct a `NegotiationSession` directly (or via `run_llm_negotiation` with fake decision callables, matching the existing test pattern in `tests/test_llm_negotiation_engine.py`) that ends in `ACCEPT` with a currency/chain matching one of a small `candidates` list, and assert `build_transaction_from_negotiation` returns a `Transaction` with matching fields (`buyer_id`, `seller_id`, `currency_symbol`, `chain_name`, `gas_fee` copied from the matching candidate, `paid_value` from the accepted price, `timestep=day`). Add a second test where the session's accepted currency does NOT match any candidate (simulating a hallucinated choice that slipped past `adapt_decision`'s looser check) and assert the function returns `None` (or raises — match whatever `timestep.py`'s Task 5 wiring will need; keep this a deliberate, tested contract either way). Add a third test where the session ends in `REJECT`/`WALK_AWAY`/hits `max_rounds` without an ACCEPT and assert `None`.

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_decision_to_transaction.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Implement**

Implement `build_transaction_from_negotiation` per the confirmed `NegotiationSession` shape from Step 1, looking up the matching `CurrencyChainOption` from `candidates` by `(currency_symbol, chain_name)` to source `gas_fee`, returning `None` for any non-ACCEPT terminal state or any accepted currency/chain absent from `candidates`.

- [ ] **Step 5: Run tests, then full suite**

Run: `pytest tests/test_decision_to_transaction.py -v` then `pytest -q`

- [ ] **Step 6: Commit**

```bash
git add src/llm/decision_to_transaction.py tests/test_decision_to_transaction.py
git commit -m "feat: add Decision/NegotiationSession to Transaction adapter with candidate-matched validation"
```

---

### Task 5: Wire LLM-driven decisions + full LLM-vs-LLM negotiation into run_timestep

**Files:**
- Modify: `src/simulation/timestep.py`
- Test: extend `tests/test_simulation.py`

**Context:** the largest task in this plan. Adds `use_llm: bool = False` to `run_timestep` (and `SimulationRunner`/wherever else `run_timestep` is invoked, so the parameter threads through), replacing the deterministic `buyer.choose_currency_and_chain` + `negotiate()` calls with a per-agent `decide()`-equivalent call (thin wrapper per design spec Sec 1.2, bypassing `decide()`'s shared-roster/policy resolution since Phase 3 uses one fixed model per agent) followed by `run_llm_negotiation`.

**Interfaces:**
- Produces: `run_timestep(..., use_llm: bool = False, openrouter_client: httpx.Client | None = None, max_negotiation_rounds: int = 10) -> TimestepResult` — when `use_llm=True`, `openrouter_client` is required (raise `ValueError` if `None`); a new module-level helper `decide_single_model(agent_class: str, context: AgentDecisionContext, model_id: str, client: httpx.Client, supported_currencies: set[str], supported_chains: set[str]) -> NegotiationAction | None` (render -> `call_model(single model_id)` -> `adapt_decision` -> on `DecisionValidationError`, one correction re-prompt, then `None` on total failure — no fallback chain, no deterministic fallback, since Phase 3's per-agent model is fixed and a hard failure should surface, not silently substitute).

- [ ] **Step 1: Read the exact current `run_timestep` body**

Re-read `src/simulation/timestep.py` in full (already done during Plan 4 research — confirm nothing has changed) to identify exactly where `buyer.choose_currency_and_chain(candidates)` and the `negotiate(...)` call happen (lines ~110-125 as of Plan 4's research).

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_simulation.py`, using `tests/llm_test_helpers.py`'s `mock_openrouter_client` (Task 1):

```python
def test_run_timestep_with_use_llm_true_requires_a_client():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    with pytest.raises(ValueError):
        run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=None)


def test_run_timestep_with_use_llm_true_produces_llm_driven_transactions():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    for agent in env.agents.values():
        agent.assigned_model = "test-vendor/test-model"
    rng = random.Random(0)
    client = mock_openrouter_client({"test-vendor/test-model": <ACCEPT-shaped decision response>})

    result = run_timestep(env, day=0, rng=rng, use_llm=True, openrouter_client=client)

    # At least confirms the LLM path executes without error and can produce transactions --
    # exact assertions depend on the mock response's decision fields; match whatever
    # tests/llm_test_helpers.py's response-construction convention ends up being (Task 1).
    assert isinstance(result.transactions, list)


def test_run_timestep_with_use_llm_false_is_unchanged_from_before():
    # Regression guard: confirm the existing deterministic-path tests
    # (test_simulation_runs_end_to_end_without_errors, etc.) still pass
    # unmodified -- this test just documents the default explicitly.
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)  # use_llm defaults to False

    assert isinstance(result.transactions, list)
```

(These are illustrative — read `tests/test_simulation.py`'s existing conventions and `tests/llm_test_helpers.py`'s actual interface once Task 1 lands, and write real, concrete assertions rather than the placeholder shown here.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_simulation.py -v`
Expected: FAIL — `use_llm` parameter doesn't exist yet.

- [ ] **Step 4: Implement `decide_single_model` and wire it into `run_timestep`**

Add `decide_single_model` (module-level helper in `timestep.py`, or a new small module if it grows large — task's own judgment) implementing: render prompt via `render_prompt(agent_class, context, schema_json)` -> `call_model(prompt, model_id, client)` -> `adapt_decision(decision, {c.currency_symbol for c in candidates}, {c.chain_name for c in candidates}, wallet.balances)` -> on `DecisionValidationError`, one re-prompt with a correction message appended, then `None`.

In `run_timestep`, replace the `chosen = buyer.choose_currency_and_chain(candidates)` + `negotiate(...)` block with (when `use_llm=True`): build each side's `AgentDecisionContext` (buyer and seller both need one, using their respective `assigned_model`), wrap `decide_single_model` results into two closures matching `run_llm_negotiation`'s `buyer_decide`/`seller_decide: Callable[[NegotiationSession], NegotiationAction]` signature, call `run_llm_negotiation(buyer.agent_id, seller.agent_id, buyer_decide, seller_decide, max_rounds=max_negotiation_rounds)`, then `build_transaction_from_negotiation` (Task 4) to get a `Transaction` or `None`. Every LLM decision (both success and total-failure) gets a `detect_hallucination` call and is recorded for Task 8's persistence wiring to pick up (task decides exact `TimestepResult` field additions needed to carry LLM-decision records forward to the persistence layer — likely a new `llm_decisions: list[...]` field on `TimestepResult`, matching the existing `memory_events`/`fired_shocks` pattern).

Keep the existing deterministic path completely intact for `use_llm=False` (the default) — this is an `if use_llm: ... else: ...` branch around the decision+negotiation section only; everything else (shock application, marketplace listing, memory recording) is shared and unconditional.

- [ ] **Step 5: Run tests, then full suite**

Run: `pytest tests/test_simulation.py -v` then `pytest -q`
Expected: all pass, including every pre-existing deterministic-path test (which must be completely unaffected since `use_llm` defaults to `False`).

- [ ] **Step 6: Commit**

```bash
git add src/simulation/timestep.py tests/test_simulation.py
git commit -m "feat: wire LLM-driven decisions and full LLM-vs-LLM negotiation into run_timestep (use_llm flag)"
```

---

### Task 6: Cross-border FX conversion tax

**Files:**
- Create: `configs/economy/fx_params.yaml`
- Modify: `src/simulation/timestep.py` (or new `src/transactions/fx_tax.py` — task's judgment on placement)
- Modify: `src/transactions/settlement.py` (debit `fx_tax_paid` from buyer's wallet)
- Test: new `tests/test_fx_tax.py`, extend `tests/test_transactions.py`

**Interfaces:**
- Produces: `configs/economy/fx_params.yaml` (`fx_tax_rate: 0.0002`), `FxParams` Pydantic model + `load_fx_params()` (matching `trust_params.yaml`'s existing loader pattern from Plan 2), `currency_zone_of(currency: CurrencyConfig) -> str | None` (`"USD"`/`"EUR"`/`None` for gold-backed, derived from `currency.peg`), `compute_fx_tax(paid_value: float, currency: CurrencyConfig, buyer_zone: str | None, fx_tax_rate: float) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fx_tax.py
import pytest

from src.currencies.currency import load_currency_universe
from src.economy.fx_tax import FxParams, compute_fx_tax, currency_zone_of, load_fx_params


def test_load_fx_params_reads_the_real_config():
    params = load_fx_params()
    assert params.fx_tax_rate == 0.0002


def test_currency_zone_of_maps_usd_pegged_currencies():
    currencies = load_currency_universe()
    assert currency_zone_of(currencies["USDC"]) == "USD"


def test_currency_zone_of_maps_eur_pegged_currencies():
    currencies = load_currency_universe()
    assert currency_zone_of(currencies["EURC"]) == "EUR"


def test_currency_zone_of_is_none_for_gold_backed():
    currencies = load_currency_universe()
    assert currency_zone_of(currencies["PAXG"]) is None
    assert currency_zone_of(currencies["XAUT"]) is None


def test_compute_fx_tax_applies_when_buyer_zone_differs_from_currency_zone():
    currencies = load_currency_universe()
    tax = compute_fx_tax(paid_value=1000.0, currency=currencies["EURC"], buyer_zone="USD", fx_tax_rate=0.0002)
    assert tax == pytest.approx(0.2)


def test_compute_fx_tax_is_zero_when_zones_match():
    currencies = load_currency_universe()
    tax = compute_fx_tax(paid_value=1000.0, currency=currencies["USDC"], buyer_zone="USD", fx_tax_rate=0.0002)
    assert tax == 0.0


def test_compute_fx_tax_is_zero_for_gold_backed_regardless_of_buyer_zone():
    currencies = load_currency_universe()
    tax = compute_fx_tax(paid_value=1000.0, currency=currencies["PAXG"], buyer_zone="EUR", fx_tax_rate=0.0002)
    assert tax == 0.0


def test_compute_fx_tax_is_zero_when_buyer_zone_is_none():
    # An agent with no currency_zone assigned (e.g. legacy single-profile construction) never pays FX tax.
    currencies = load_currency_universe()
    tax = compute_fx_tax(paid_value=1000.0, currency=currencies["EURC"], buyer_zone=None, fx_tax_rate=0.0002)
    assert tax == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fx_tax.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create the config and implement `src/economy/fx_tax.py`**

Create `configs/economy/fx_params.yaml`:
```yaml
fx_tax_rate: 0.0002
```

Implement (following Plan 2's `trust.py`/`trust_params.yaml` loader pattern exactly — read that file first to match `load_yaml_as`/`CONFIG_ROOT` usage):
```python
from pydantic import BaseModel

from src.currencies.currency import CurrencyConfig
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

FX_PARAMS_PATH = CONFIG_ROOT / "economy" / "fx_params.yaml"


class FxParams(BaseModel):
    fx_tax_rate: float


def load_fx_params(path=FX_PARAMS_PATH) -> FxParams:
    return load_yaml_as(path, FxParams)


def currency_zone_of(currency: CurrencyConfig) -> str | None:
    if currency.peg == "USD":
        return "USD"
    if currency.peg == "EUR":
        return "EUR"
    return None


def compute_fx_tax(paid_value: float, currency: CurrencyConfig, buyer_zone: str | None, fx_tax_rate: float) -> float:
    currency_zone = currency_zone_of(currency)
    if currency_zone is None or buyer_zone is None or currency_zone == buyer_zone:
        return 0.0
    return paid_value * fx_tax_rate
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fx_tax.py -v`

- [ ] **Step 5: Wire fx_tax_paid into the Transaction and debit it at settlement**

In `src/simulation/timestep.py`, after computing `agreed_price` (in both the deterministic and LLM-driven paths from Task 5), compute `fx_tax_paid = compute_fx_tax(agreed_price, env.currencies[chosen.currency_symbol], buyer.currency_zone, fx_params.fx_tax_rate)` and pass it into the `Transaction(...)` construction.

In `src/transactions/settlement.py`'s `settle(...)`, read `tx.fx_tax_paid` and debit it from the buyer's wallet balance for `tx.currency_symbol`, **in addition to** the existing price debit (confirm exact wallet-debit mechanics by reading `settle()`'s current 16-line body first — do not guess whether it's `wallet.withdraw(...)` or direct balance mutation).

Add a test to `tests/test_transactions.py`:
```python
def test_settle_debits_fx_tax_paid_in_addition_to_paid_value():
    # Construct a Transaction with a nonzero fx_tax_paid and confirm the buyer's
    # wallet balance drops by paid_value + fx_tax_paid, not just paid_value.
    ...
```
(Match whatever `Wallet`/`settle()` construction pattern the rest of `tests/test_transactions.py` already uses.)

- [ ] **Step 6: Run tests, then full suite**

Run: `pytest tests/test_fx_tax.py tests/test_transactions.py -v` then `pytest -q`

- [ ] **Step 7: Commit**

```bash
git add configs/economy/fx_params.yaml src/economy/fx_tax.py src/simulation/timestep.py src/transactions/settlement.py tests/test_fx_tax.py tests/test_transactions.py
git commit -m "feat: add cross-border FX conversion tax (0.02%, zone-mismatch triggered)"
```

---

### Task 7: Loss-driven CARA-coefficient adaptation

**Files:**
- Create: `configs/economy/risk_adaptation_params.yaml`
- Create: `src/economy/risk_adaptation.py`
- Test: new `tests/test_risk_adaptation.py`

**Interfaces:**
- Produces: `RiskAdaptationParams` (`eta_risk: float`, `a_max: float`) + `load_risk_adaptation_params()`, `adapt_cara_coefficient(agent: BaseAgent, w_real_before: float, w_real_after: float, params: RiskAdaptationParams) -> None` (mutates `agent.cara_coefficient`, `agent.risk_aversion`, `agent.utility_fn` in place; no-op if `agent.cara_coefficient is None`).

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.economy.risk_adaptation import RiskAdaptationParams, adapt_cara_coefficient, load_risk_adaptation_params
from src.utility.cara import CARAUtility
from src.utility.risk_neutral import RiskNeutralUtility


def _params() -> RiskAdaptationParams:
    return RiskAdaptationParams(eta_risk=1.0, a_max=5.0)


def test_load_risk_adaptation_params_reads_the_real_config():
    params = load_risk_adaptation_params()
    assert params.eta_risk == 1.0
    assert params.a_max == 5.0


def test_adapt_cara_coefficient_is_a_noop_for_non_cara_eligible_agents():
    profile = load_agent_profiles()["merchant"]
    agent = build_agent(profile)  # cara_coefficient is None
    original_utility_fn = agent.utility_fn

    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=800.0, params=_params())

    assert agent.cara_coefficient is None
    assert agent.utility_fn is original_utility_fn


def test_adapt_cara_coefficient_increases_a_after_a_realized_loss():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, cara_override=("cara", 1.0))

    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=800.0, params=_params())
    # Loss_t = 200, W_real_t = 800, eta_risk=1.0 -> a_next = 1.0 + 1.0 * 200/800 = 1.25

    assert agent.cara_coefficient == pytest.approx(1.25)
    assert agent.risk_aversion == pytest.approx(1.25)
    assert isinstance(agent.utility_fn, CARAUtility)


def test_adapt_cara_coefficient_never_decreases_on_a_gain():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, cara_override=("cara", 1.0))

    adapt_cara_coefficient(agent, w_real_before=800.0, w_real_after=1000.0, params=_params())

    assert agent.cara_coefficient == pytest.approx(1.0)  # unchanged, gains don't reduce a


def test_adapt_cara_coefficient_clamps_at_a_max():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, cara_override=("cara", 4.9))

    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=1.0, params=_params())
    # Loss_t = 999, W_real_t = 1.0 -> raw a_next would be huge; must clamp at a_max=5.0

    assert agent.cara_coefficient == pytest.approx(5.0)


def test_adapt_cara_coefficient_switches_to_cara_when_a_crosses_above_zero():
    profile = load_agent_profiles()["bank"]
    agent = build_agent(profile, cara_override=("risk_neutral", None))  # nominal a starts at 0.0

    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=500.0, params=_params())
    # Loss_t = 500, W_real_t = 500, eta_risk=1.0 -> a_next = 0.0 + 1.0*500/500 = 1.0 (crosses above 0)

    assert agent.cara_coefficient == pytest.approx(1.0)
    assert agent.utility_type == "cara"
    assert isinstance(agent.utility_fn, CARAUtility)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_risk_adaptation.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create the config and implement**

Create `configs/economy/risk_adaptation_params.yaml`:
```yaml
eta_risk: 1.0
a_max: 5.0
```

Implement `src/economy/risk_adaptation.py` (following the same `load_yaml_as`/`CONFIG_ROOT` pattern as Task 6):

```python
from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.utility.utility_factory import build_utility_function
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

RISK_ADAPTATION_PARAMS_PATH = CONFIG_ROOT / "economy" / "risk_adaptation_params.yaml"


class RiskAdaptationParams(BaseModel):
    eta_risk: float
    a_max: float


def load_risk_adaptation_params(path=RISK_ADAPTATION_PARAMS_PATH) -> RiskAdaptationParams:
    return load_yaml_as(path, RiskAdaptationParams)


def adapt_cara_coefficient(
    agent: BaseAgent, w_real_before: float, w_real_after: float, params: RiskAdaptationParams
) -> None:
    if agent.cara_coefficient is None:
        return

    loss = max(0.0, w_real_before - w_real_after)
    a_next = min(params.a_max, agent.cara_coefficient + params.eta_risk * loss / w_real_after)

    agent.cara_coefficient = a_next
    if a_next == 0.0:
        agent.utility_type = "risk_neutral"
        agent.risk_aversion = None
    else:
        agent.utility_type = "cara"
        agent.risk_aversion = a_next
    agent.utility_fn = build_utility_function(agent.utility_type, agent.risk_aversion, agent.multi_attribute_weights, agent.eis)
```

(Verify `w_real_after` is never `0.0` before dividing — if an agent's real wealth can legitimately hit exactly zero, add a guard; check whether this is reachable given wallet balances in practice, and note the finding in the task report either way.)

- [ ] **Step 4: Run tests, then full suite**

Run: `pytest tests/test_risk_adaptation.py -v` then `pytest -q`

- [ ] **Step 5: Commit**

```bash
git add configs/economy/risk_adaptation_params.yaml src/economy/risk_adaptation.py tests/test_risk_adaptation.py
git commit -m "feat: add loss-driven CARA-coefficient adaptation"
```

---

### Task 8: CurrencyHistory/MacroHistory auto-population + EventLog wiring

**Files:**
- Modify: `src/simulation/environment.py` (`self.event_log = EventLog()`)
- Modify: `src/simulation/timestep.py` (record due_shocks into `env.event_log`)
- Create: `src/economy/history_builder.py`
- Test: extend `tests/test_simulation.py`, new `tests/test_history_builder.py`

**Interfaces:**
- Produces: `Environment.event_log: EventLog`, `build_currency_history(ledger: TrustLedger, event_log: EventLog, symbol: str, day: int) -> CurrencyHistory`, `build_macro_history(env: Environment, day: int) -> MacroHistory` (exact field computations per Plan 2's design spec §3.4 — `trust_now`/`trust_30d_ago`/`trust_min_90d` read `TrustLedger.history()`; `depeg_events_90d`/`last_event_days_ago`/`recent_events` read `EventLog.all_events()` filtered to the last 90 days and to `target_currency == symbol`).

- [ ] **Step 1: Re-read Plan 2's design spec §3.4 for the exact field semantics**

Read `docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md` §3.4 (or the master spec's equivalent section) for `trend` ("declining"/"stable"/"recovering")'s exact definition before implementing — this plan implements, not (re)designs, those semantics.

- [ ] **Step 2: Add `event_log` to Environment and wire recording**

In `src/simulation/environment.py`, add `self.event_log = EventLog()` in `__init__`. In `src/simulation/timestep.py`, add `env.event_log.record(shock)` for each `due_shock`, directly after the existing `due_shocks = env.event_queue.pop_due(day)` loop.

Add a test to `tests/test_simulation.py`:
```python
def test_run_timestep_records_fired_shocks_into_the_event_log():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    env.event_queue.schedule(ShockEvent(day=0, type=ShockType.INFLATION, magnitude=0.02))  # confirm actual scheduling mechanism per Plan 2's Task 5
    rng = random.Random(0)

    run_timestep(env, day=0, rng=rng)

    assert len(env.event_log.all_events()) == 1
```

- [ ] **Step 3: Write the failing tests for build_currency_history/build_macro_history**

Create `tests/test_history_builder.py` exercising a `TrustLedger` + `EventLog` with a known shock history (a `depeg_event` a few days ago) and asserting the returned `CurrencyHistory`'s `depeg_events_90d`, `last_event_days_ago`, `trend`, and `recent_events` match expectations; similarly for `build_macro_history`'s `days_since_last_shock`/`last_shock_type`.

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_simulation.py tests/test_history_builder.py -v`
Expected: FAIL — `event_log` attribute / module don't exist yet.

- [ ] **Step 5: Implement `src/economy/history_builder.py`**

Implement per Step 1's confirmed field semantics.

- [ ] **Step 6: Run tests, then full suite**

Run: `pytest tests/test_simulation.py tests/test_history_builder.py -v` then `pytest -q`

- [ ] **Step 7: Commit**

```bash
git add src/simulation/environment.py src/simulation/timestep.py src/economy/history_builder.py tests/test_simulation.py tests/test_history_builder.py
git commit -m "feat: wire EventLog recording and add CurrencyHistory/MacroHistory builders"
```

---

### Task 9: Live Polygon price wiring

**Files:**
- Modify: `src/simulation/timestep.py` (fetch live prices once per day, pass into decision contexts)
- Test: extend `tests/test_simulation.py`

**Interfaces:**
- Consumes: `market_intelligence.fetch_live_price`/`build_polygon_client` (existing), `tests/llm_test_helpers.py`'s `mock_polygon_client` (Task 1).
- Produces: `run_timestep(..., polygon_client: httpx.Client | None = None)` — when `use_llm=True` and `polygon_client` is provided, fetches one `LivePriceSnapshot` per tradable currency's reference ticker once at the start of the day (not per-agent), passed into every `build_decision_context` call that day via `live_price_snapshots`. When `polygon_client` is `None` (default), `live_price_snapshots` stays empty (existing behavior — `build_decision_context` already defaults this to `{}`).

- [ ] **Step 1: Confirm the ticker-per-currency mapping**

Check whether `CurrencyConfig` or another existing config already declares a ticker symbol per currency (grep for `ticker` in `src/currencies/`, `configs/currencies/`); if none exists, this task must decide/add one (e.g. `peg`-based: USD-pegged currencies use a stablecoin index ticker, EUR-pegged use EURUSD, gold-backed use XAU/USD — confirm via `market_intelligence.py`'s existing test fixtures for what ticker format `fetch_live_price` actually expects before inventing a mapping).

- [ ] **Step 2: Write the failing test**

Add to `tests/test_simulation.py`, using `mock_polygon_client` (Task 1):
```python
def test_run_timestep_with_polygon_client_populates_live_price_snapshots():
    # Confirm build_decision_context (called internally by the use_llm=True path
    # from Task 5) receives a non-empty live_price_snapshots dict when a
    # polygon_client is supplied. Exact assertion mechanism depends on how
    # Task 5's implementation exposes this for testing (e.g. a spy/capture
    # on build_decision_context, or checking a side effect) -- read Task 5's
    # actual landed implementation first rather than assuming.
    ...
```

- [ ] **Step 3: Run test to verify it fails, implement, verify it passes**

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`

- [ ] **Step 5: Commit**

```bash
git add src/simulation/timestep.py tests/test_simulation.py
git commit -m "feat: wire live Polygon price fetching into the LLM decision context"
```

---

### Task 10: Synthetic sandbox currency configs

**Files:**
- Create: `src/currencies/sandbox_currencies.py`
- Modify: `src/simulation/environment.py` (`Environment.build_from_population` classmethod, per design spec Sec 6.1)
- Test: new `tests/test_sandbox_currencies.py`, extend `tests/test_simulation.py`

**Interfaces:**
- Produces: `SANDBOX_CURRENCY_PAIRS: dict[str, tuple[CurrencyConfig, CurrencyConfig]]` keyed by sandbox name (`"liquidity_vs_governance"`, `"governance_vs_stability"`, `"liquidity_vs_stability"`, `"asset_backing_vs_liquidity"`, `"asset_backing_vs_stability"`, `"asset_backing_vs_governance"`), `Environment.build_from_population(scenario_name, agents, currencies=None, goods=None) -> Environment`.

- [ ] **Step 1: Confirm CurrencyConfig's exact required fields**

Read `src/currencies/currency.py`'s `CurrencyConfig` Pydantic model in full to confirm every required/optional field (symbol, asset_class, peg, governance_score, liquidity_score, peg_error, issuer_risk, genius_compliant, plus any asset-class-specific optional fields like `redemption_mechanism`/`gold_reserve_oz`/`custodian`/`issuing_bank`/`fdic_insured` seen in the real YAML files) before constructing the 12 synthetic instances — do not guess field names or omit a required one.

- [ ] **Step 2: Write the failing tests**

```python
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS


def test_all_six_sandboxes_present():
    assert set(SANDBOX_CURRENCY_PAIRS.keys()) == {
        "liquidity_vs_governance", "governance_vs_stability", "liquidity_vs_stability",
        "asset_backing_vs_liquidity", "asset_backing_vs_stability", "asset_backing_vs_governance",
    }


def test_liquidity_vs_governance_isolates_exactly_those_two_dimensions():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_governance"]
    assert option_a.liquidity_score != option_b.liquidity_score
    assert option_a.governance_score != option_b.governance_score
    assert option_a.peg_error == option_b.peg_error  # held constant
    assert option_a.peg == option_b.peg == "USD"


# (Similar isolation tests for the other 5 sandboxes, per the design spec Sec 6.2 table --
# implementer constructs the full test suite from that table, one test per sandbox
# confirming the intended dimension(s) differ and the others are held constant.)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_sandbox_currencies.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Implement the 12 synthetic CurrencyConfig instances**

Implement per the design spec Sec 6.2 table's exact numbers, using Step 1's confirmed schema.

- [ ] **Step 5: Add Environment.build_from_population**

In `src/simulation/environment.py`:
```python
@classmethod
def build_from_population(
    cls, scenario_name: str, agents: list[BaseAgent],
    currencies: dict[str, CurrencyConfig] | None = None, goods: list[Good] | None = None,
) -> "Environment":
    resolved_currencies = currencies if currencies is not None else load_currency_universe()
    chains = load_chain_universe()
    scenario = load_scenario(scenario_name)
    return cls(currencies=resolved_currencies, chains=chains, scenario=scenario, agents=agents, goods=goods)
```

Add a test to `tests/test_simulation.py`:
```python
def test_build_from_population_uses_full_universe_when_currencies_is_none():
    from src.agents.population import generate_agent_population
    population = generate_agent_population(seed=0, model_candidates=["vendor/model"])

    env = Environment.build_from_population("baseline", population)

    assert len(env.currencies) == 9  # full real universe


def test_build_from_population_uses_supplied_currencies_when_given():
    from src.agents.population import generate_agent_population
    from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
    population = generate_agent_population(seed=0, model_candidates=["vendor/model"])
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["liquidity_vs_governance"]

    env = Environment.build_from_population(
        "baseline", population, currencies={option_a.symbol: option_a, option_b.symbol: option_b}
    )

    assert len(env.currencies) == 2
```

- [ ] **Step 6: Run tests, then full suite**

Run: `pytest tests/test_sandbox_currencies.py tests/test_simulation.py -v` then `pytest -q`

- [ ] **Step 7: Commit**

```bash
git add src/currencies/sandbox_currencies.py src/simulation/environment.py tests/test_sandbox_currencies.py tests/test_simulation.py
git commit -m "feat: add synthetic sandbox currency pairs and Environment.build_from_population"
```

---

### Task 11: Full per-day persistence wiring

**Files:**
- Modify: `database/repository.py` (`persist_full_timestep` function)
- Test: new `tests/test_full_persistence.py`

**Context:** ties Tasks 2 (provenance), 3 (real purchasing power), 5 (LLM decisions), 7 (adapted cara_coefficient), 8 (event_log/memory_events) together into one persistence call per day.

**Interfaces:**
- Produces: `persist_full_timestep(session: Session, env: Environment, result: TimestepResult, run_id: str) -> None` — writes one `TimestepLogEntry`, one `AgentStateLogEntry` per agent (using Task 3's `real_purchasing_power`, Task 7's post-adaptation `cara_coefficient`, Sec 7's `utility_score` resolution), one `InterventionLogEntry` per `result.fired_shocks`, one `AgentMemoryLogEntry` per `result.memory_events` tuple, plus whatever Task 5 added to `TimestepResult` for LLM decisions (one `LLMDecisionLogEntry`/`HallucinationLogEntry` pair per decision) — and still calls the existing `AgentRepository.upsert_agent`/`TransactionRepository.record`/`NegotiationRepository.record` exactly as `persist_timestep` already does (extend that function rather than duplicating its existing behavior, unless doing so is awkward — task's judgment, documented in its report either way).

- [ ] **Step 1: Write the failing test**

Create `tests/test_full_persistence.py` with a small `Environment` (2-4 agents, `baseline` scenario, 1-2 days via `run_timestep`), calling `persist_full_timestep` and asserting each of the 4-6 new tables gained the expected row counts (one `TimestepLogRecord` for the day, `N` `AgentStateRecord`s for `N` agents, etc.) — following the existing `sqlite:///:memory:` + `Base.metadata.create_all` pattern used throughout `tests/test_*_persistence.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_full_persistence.py -v`
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Implement `persist_full_timestep`**

Implement per the Interfaces description, reading `database/repository.py`'s existing `persist_timestep` first to extend consistently.

- [ ] **Step 4: Run tests, then full suite**

Run: `pytest tests/test_full_persistence.py -v` then `pytest -q`

- [ ] **Step 5: Commit**

```bash
git add database/repository.py tests/test_full_persistence.py
git commit -m "feat: add persist_full_timestep wiring all per-day/per-agent/per-decision tables"
```

---

### Task 12: Master 365-day scenario + H4 proximity sweep

**Files:**
- Create: `configs/scenarios/master_simulation.yaml`
- Test: new `tests/test_master_scenario.py`

**Interfaces:**
- Produces: `configs/scenarios/master_simulation.yaml` — `duration_days: 365`, shock schedule per design spec Sec 9 (0/5/10/20-day `crisis_warning`->`depeg_event` gaps, each pair targeting a different currency; at least one instance of every other shock type; non-confounding day-spacing).

- [ ] **Step 1: Draft the shock schedule**

Author the full `shocks:` list. Reuse `configs/scenarios/baseline.yaml`'s `initial_state` shape. Space every shock at least 15 days apart from any other shock's day (except the deliberate `crisis_warning`/`depeg_event` pairs, which are intentionally close per their gap value) so effects don't confound. Use distinct `target_currency` values across different shocks where applicable so no single currency gets hit twice in a way that would confound a single shock's isolated effect.

- [ ] **Step 2: Write the failing test**

```python
from src.economy.shocks import ShockType, load_scenario


def test_master_simulation_scenario_loads_and_spans_365_days():
    scenario = load_scenario("master_simulation")
    assert scenario.duration_days == 365


def test_master_simulation_has_h4_proximity_sweep_at_four_gap_values():
    scenario = load_scenario("master_simulation")
    warnings = [s for s in scenario.shocks if s.type == ShockType.CRISIS_WARNING]
    depegs = [s for s in scenario.shocks if s.type == ShockType.DEPEG_EVENT]
    # For each of the 4 gap values, confirm a crisis_warning/depeg_event pair
    # targeting the same currency exists at that exact day-gap.
    gaps_found = set()
    for warning in warnings:
        matching_depeg = next(
            (d for d in depegs if d.target_currency == warning.target_currency and d.day > warning.day), None
        )
        if matching_depeg is not None:
            gaps_found.add(matching_depeg.day - warning.day)
    assert gaps_found == {0, 5, 10, 20}


def test_master_simulation_includes_every_shock_type_at_least_once():
    scenario = load_scenario("master_simulation")
    types_present = {s.type for s in scenario.shocks}
    assert types_present == set(ShockType)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_master_scenario.py -v`
Expected: FAIL — scenario file doesn't exist.

- [ ] **Step 4: Author the YAML, run test to verify it passes**

Run: `pytest tests/test_master_scenario.py -v`

- [ ] **Step 5: Commit**

```bash
git add configs/scenarios/master_simulation.yaml tests/test_master_scenario.py
git commit -m "feat: add 365-day master simulation scenario with H4 proximity sweep"
```

---

### Task 13: The matrix runner

**Files:**
- Create: `src/simulation/matrix_runner.py`
- Test: new `tests/test_matrix_runner.py`

**Context:** the final task, tying every prior task together. Must default to `dry_run=True` and refuse real API calls unless the caller explicitly opts in with real clients — this is the code-level half of the "explicit confirmation before billed spend" gate; it does not itself decide to launch anything.

**Interfaces:**
- Produces: `run_matrix(model_candidates: list[str], seeds: list[int], num_days: int, dry_run: bool = True, openrouter_client: httpx.Client | None = None, polygon_client: httpx.Client | None = None) -> list[SimulationResult]` per design spec Sec 10. Cross-border pairing mechanism for the 6 cross-border cells: **read `src/market/marketplace.py`'s listing/matching logic in full first** (not yet done by any prior task in this plan) to confirm exactly how buyer-seller matching happens in `run_timestep` today (`marketplace.find_counterparties`), then add the minimal change needed to force zone-mismatched pairing only for cross-border cells (e.g. an optional `require_cross_zone: bool = False` parameter threaded through `find_counterparties`/`generate_candidates`'s caller in `timestep.py`) — document the exact mechanism chosen in the task report, since the design spec deliberately left this as an implementation-time detail.

- [ ] **Step 1: Read `src/market/marketplace.py` in full**

Confirm exact listing/matching mechanics before deciding how to force cross-border pairing.

- [ ] **Step 2: Write the failing tests**

```python
def test_run_matrix_with_dry_run_true_does_not_require_real_clients():
    results = run_matrix(model_candidates=["vendor/model"], seeds=[0], num_days=2, dry_run=True)
    assert len(results) == 13  # 1 master + 6 domestic + 6 cross-border, 1 seed


def test_run_matrix_refuses_dry_run_false_without_real_clients():
    with pytest.raises(ValueError):
        run_matrix(model_candidates=["vendor/model"], seeds=[0], num_days=2, dry_run=False)


def test_run_matrix_produces_13_cells_per_seed():
    results = run_matrix(model_candidates=["vendor/model"], seeds=[0, 1], num_days=1, dry_run=True)
    assert len(results) == 26  # 13 cells x 2 seeds
```

(Illustrative — implementer writes the real, concrete test suite once every prior task's actual interface is confirmed; this is the integration point where any earlier task's interface mismatch would surface, so read every dependency's actual landed code, not this plan's proposed-code snippets, before implementing.)

- [ ] **Step 3: Run test to verify it fails, implement, verify it passes**

- [ ] **Step 4: Run the full suite one final time for this plan**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass — confirms every table/mechanism added across Tasks 1-13 coexists cleanly.

- [ ] **Step 5: Commit**

```bash
git add src/simulation/matrix_runner.py tests/test_matrix_runner.py
git commit -m "feat: add the 13-cell x 5-seed matrix runner with dry_run safety gate"
```

---

## What comes after this plan

1. **Full-scale run launch** — an explicit, separate go/no-go checkpoint with the user before `run_matrix(..., dry_run=False)` is ever called with real clients. Nothing in this plan launches that run automatically.
2. **Econometrics engine (Plan 5)** — H1-H5 regression outputs (β, SE, 95% CI, p-value, R², adjusted R²), reading from the complete raw dataset this plan persists.
