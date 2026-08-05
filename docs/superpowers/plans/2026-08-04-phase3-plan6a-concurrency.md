# Phase 3 Plan 6a: Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 13-cell x 3-seed x 365-day production matrix run feasible in wall-clock time by parallelizing (a) LLM negotiation calls across agents within a simulated day and (b) independent cells/seeds across OS processes, plus add cost/token visibility that doesn't exist today.

**Architecture:** Within a day, each buyer's full per-day work (all its goods, processed sequentially per buyer since one buyer's own wallet balance carries across its own goods) runs as one unit of work; different buyers' units run concurrently on a thread pool (the negotiation calls are I/O-bound network requests, not CPU work). A single `threading.Lock` serializes the handful of shared-state mutations (wallet settlement, ledger recording, result-list appends) so nothing outside that narrow critical section is ever contended. Across cells/seeds, `run_matrix` gains a cell-subsetting parameter so a new orchestrator can partition the 13 cells x 3 seeds across separate OS processes, each running its own subset against one shared SQLite database (WAL mode + busy-timeout so concurrent-process writes don't fail).

**Tech Stack:** `concurrent.futures.ThreadPoolExecutor` (within-day), `concurrent.futures.ProcessPoolExecutor` (cross-cell/seed), stdlib `threading.Lock`, SQLite `PRAGMA journal_mode=WAL` / `PRAGMA busy_timeout`.

## Global Constraints

- Default behavior for every existing caller/test must be byte-for-byte unchanged: new concurrency is opt-in via new parameters that default to today's fully-sequential behavior.
- The rule-based (`use_llm=False`) day-loop path is untouched — it has no network calls, so parallelizing it has no value and is out of scope.
- No change to any hypothesis's economic meaning: this plan only changes *how fast* decisions are computed, never *what* gets computed or in what economically-meaningful order.
- Follow the existing project convention: `from __future__ import annotations` + function-local `httpx`-dependent imports in `src/simulation/timestep.py` (see that file's module docstring) — do not add a module-level `httpx` import there.

---

### Task 1: Extract the per-buyer LLM-path body into its own function

**Files:**
- Modify: `src/simulation/timestep.py:536-813` (the `for buyer in active_buyers:` loop)
- Test: `tests/test_timestep_persistence.py`

**Interfaces:**
- Produces: `_process_buyer_llm_day(buyer, env, day, fx_params, live_price_snapshots, currency_history, macro_history, openrouter_client, max_negotiation_rounds, agreement_tolerance, concession_rate, result, lock) -> None` — processes every good for one buyer under the LLM path, mutating `result` and wallets under `lock` exactly where settlement/recording happens. Consumed by Task 2's sequential and parallel call sites.

This task is a pure refactor with no behavior change: it moves the existing `if use_llm:` branch body (currently inlined inside the doubly-nested `for buyer / for good` loop) into a standalone function, and adds a `lock` parameter around the small shared-state critical section. Running this task alone (before Task 2 changes anything about *how* it's called) must leave every existing test passing unchanged.

- [ ] **Step 1: Write a test that pins current sequential behavior as a regression baseline**

Add to `tests/test_timestep_persistence.py` (uses the same `dry_run=True, exercise_llm_path=True` pattern `tests/test_matrix_runner.py` already establishes for exercising the LLM path without real network calls):

```python
from src.simulation.matrix_runner import run_matrix
from database.session import new_session
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database.models import Base


def test_llm_path_produces_identical_transaction_count_before_and_after_refactor():
    """Regression baseline for Plan 6a Task 1: extracting the per-buyer LLM
    body into _process_buyer_llm_day must not change how many transactions
    a fixed-seed run produces."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    results, failures = run_matrix(
        model_candidates=["vendor/fake-model"],
        seeds=[0],
        num_days=3,
        dry_run=True,
        exercise_llm_path=True,
        session=session,
        keep_daily_results=True,
    )
    assert failures == []
    master_result = next(r for r in results if r.cell_key == "master")
    # Fixed seed + fixed mock decision -> deterministic transaction count.
    # Record the count this test observes BEFORE Task 1's refactor; Step 2
    # runs this test to capture that number, then Step 4 re-runs it after
    # the refactor to confirm it is unchanged.
    assert master_result.total_transactions >= 0  # placeholder assertion, see Step 2
```

- [ ] **Step 2: Run the test before refactoring to record the real transaction count**

Run: `pytest tests/test_timestep_persistence.py::test_llm_path_produces_identical_transaction_count_before_and_after_refactor -v -s`

Note the actual `master_result.total_transactions` value printed (add a temporary `print(master_result.total_transactions)` if needed, then remove it), and replace the placeholder assertion in Step 1 with `assert master_result.total_transactions == <that exact number>`.

- [ ] **Step 3: Extract `_process_buyer_llm_day` in `src/simulation/timestep.py`**

Add `import threading` to the top-level imports (alongside the existing `import json`, `import random`). Add this new function right after `_make_llm_decide_closure` (after line 348):

```python
def _process_buyer_llm_day(
    buyer: BuyerAgent,
    env: Environment,
    day: int,
    fx_params,
    live_price_snapshots: dict,
    currency_history: dict,
    macro_history,
    openrouter_client: httpx.Client,
    max_negotiation_rounds: int,
    agreement_tolerance: float,
    concession_rate: float,
    result: TimestepResult,
    lock: threading.Lock,
) -> None:
    """Runs one buyer's full per-day LLM-path work (every good in
    env.goods, in order -- a buyer's own wallet balance carries from one
    good to the next within its own loop, so goods for ONE buyer must stay
    sequential; see Plan 6a's design spec Sec 2.1). Safe to call for
    different buyers concurrently from separate threads: `lock` serializes
    the only shared-state mutations (result.llm_decisions/llm_negotiations/
    transactions appends, wallet settlement, env.ledger.record, memory
    updates) -- everything else here (listing lookup, candidate
    generation, LLM negotiation calls) touches only this buyer's own
    thread-exclusive state or read-only environment state (see Plan 6a's
    design spec Sec 2 for the full safety analysis this relies on).

    Known accepted limitation (documented, not a bug): a seller shared by
    two concurrently-running buyers may have its wallet read (via
    seller.build_llm_context(), for that seller's own LLM prompt) without
    holding `lock`, so that read can very rarely reflect a wallet state
    from just before or just after another thread's concurrent settlement
    of a different transaction with the same seller. This affects only
    what a seller's OWN prompt displays as its current balance, not any
    economic invariant (settlement itself is always lock-protected and
    exact) -- locking every context-read would serialize away most of this
    task's concurrency benefit for a cosmetic, momentary display staleness.
    """
    from src.llm.agent_reasoning import TransactionContext, build_decision_context
    from src.llm.market_intelligence import load_currency_profile

    for good in env.goods:
        listings = env.marketplace.find_counterparties(good.name, exclude_agent_id=buyer.agent_id)
        if not listings:
            continue
        listing = listings[0]
        seller = env.agents[listing.seller_id]

        candidates = generate_candidates(
            buyer.wallet.balances,
            env.currencies,
            env.chains,
            env.liquidity_pools,
            trust_ledger=env.trust_ledger,
        )
        if not candidates:
            continue

        spread_optimal_currency, spread_optimal_chain, gas_optimal_currency, gas_optimal_chain = (
            _spread_and_gas_optimal(candidates)
        )

        supported_currencies = {c.currency_symbol for c in candidates}
        supported_chains = {c.chain_name for c in candidates}
        currency_profiles = {
            symbol: profile
            for symbol in supported_currencies
            if (profile := load_currency_profile(symbol)) is not None
        }
        counterparty_cross_zone = (
            buyer.currency_zone is not None
            and seller.currency_zone is not None
            and buyer.currency_zone != seller.currency_zone
        )
        transaction_context = TransactionContext(
            is_cross_border=counterparty_cross_zone,
            origin_currency=buyer.currency_zone if counterparty_cross_zone else None,
            destination_currency=seller.currency_zone if counterparty_cross_zone else None,
        )

        buyer_context = build_decision_context(
            buyer.build_llm_context(),
            candidates,
            currency_profiles,
            env.macro_state,
            env.macro_state,
            transaction_context,
            live_price_snapshots=live_price_snapshots,
            currency_history=currency_history,
            macro_history=macro_history,
        )
        seller_context = build_decision_context(
            seller.build_llm_context(),
            candidates,
            currency_profiles,
            env.macro_state,
            env.macro_state,
            transaction_context,
            live_price_snapshots=live_price_snapshots,
            currency_history=currency_history,
            macro_history=macro_history,
        )

        buyer_wallet_balances_usd = {
            symbol: env.exchange_rates.convert(balance, symbol, "USD")
            for symbol, balance in buyer_context.agent.wallet_balances.items()
        }

        buyer_decide = _make_llm_decide_closure(
            buyer,
            "buyer",
            buyer_context,
            buyer.assigned_model,
            openrouter_client,
            supported_currencies,
            supported_chains,
            listing.true_price,
            _LockedList(result.llm_decisions, lock),
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
            _LockedList(result.llm_decisions, lock),
            buyer_wallet_balances=buyer_wallet_balances_usd,
            spread_optimal_currency=spread_optimal_currency,
            spread_optimal_chain=spread_optimal_chain,
            gas_optimal_currency=gas_optimal_currency,
            gas_optimal_chain=gas_optimal_chain,
        )

        session = run_llm_negotiation(
            buyer.agent_id,
            seller.agent_id,
            buyer_decide,
            seller_decide,
            max_rounds=max_negotiation_rounds,
        )
        with lock:
            result.llm_negotiations.append(session)

        tx = build_transaction_from_negotiation(
            session, candidates, buyer.agent_id, seller.agent_id, good.name, day
        )
        if tx is None:
            continue

        tx.expected_value = listing.true_price
        tx.paid_value = env.exchange_rates.convert(tx.paid_value, "USD", tx.currency_symbol)
        tx.fx_tax_paid = compute_fx_tax(
            tx.paid_value, env.currencies[tx.currency_symbol], buyer.currency_zone, fx_params.fx_tax_rate
        )

        with lock:
            validation = validate_transaction(tx, buyer.wallet, env.currencies)
            if not validation.is_valid:
                tx.status = TransactionStatus.FAILED
                result.transactions.append(tx)
                continue

            settle(tx, buyer.wallet, seller.wallet)
            env.ledger.record(tx)
            result.transactions.append(tx)

            success = tx.status == TransactionStatus.SETTLED
            buyer.update_memory(tx.currency_symbol, success)
            seller.update_memory(tx.currency_symbol, success)
```

Add this small helper class right above `_process_buyer_llm_day` (a plain list-like wrapper so `_make_llm_decide_closure`'s existing `decision_log.append(...)` call site needs no change at all — it just calls `.append()` on whatever it's given):

```python
class _LockedList:
    """Wraps a list so `.append()` is lock-protected -- lets
    `_make_llm_decide_closure` keep its existing `decision_log.append(...)`
    call site unchanged while making concurrent appends from multiple
    buyer-threads safe."""

    def __init__(self, target: list, lock: threading.Lock):
        self._target = target
        self._lock = lock

    def append(self, item) -> None:
        with self._lock:
            self._target.append(item)
```

`_make_llm_decide_closure`'s signature and body do not change at all — it already just calls `decision_log.append(...)`, so passing it a `_LockedList` instead of the raw `result.llm_decisions` list is the only change needed to make its appends thread-safe.

Now replace the body of `run_timestep`'s buyer loop (lines 536-813) with:

```python
    lock = threading.Lock()

    if use_llm:
        for buyer in active_buyers:
            _process_buyer_llm_day(
                buyer,
                env,
                day,
                fx_params,
                live_price_snapshots,
                currency_history,
                macro_history,
                openrouter_client,
                max_negotiation_rounds,
                agreement_tolerance,
                concession_rate,
                result,
                lock,
            )
    else:
        for buyer in active_buyers:
            for good in env.goods:
                listings = env.marketplace.find_counterparties(good.name, exclude_agent_id=buyer.agent_id)
                if not listings:
                    continue
                listing = listings[0]
                seller = env.agents[listing.seller_id]

                candidates = generate_candidates(
                    buyer.wallet.balances,
                    env.currencies,
                    env.chains,
                    env.liquidity_pools,
                    trust_ledger=env.trust_ledger,
                )
                if not candidates:
                    continue

                chosen = buyer.choose_currency_and_chain(candidates)

                buyer_open = buyer.opening_offer_price(listing.true_price)
                seller_open = seller.asking_price(listing.true_price)
                agreed_price, log = negotiate(
                    buyer_opening_price=buyer_open,
                    seller_opening_price=seller_open,
                    currency_symbol=chosen.currency_symbol,
                    chain_name=chosen.chain_name,
                    true_price=listing.true_price,
                    supported_currencies=set(env.currencies.keys()),
                    max_rounds=max_negotiation_rounds,
                    agreement_tolerance=agreement_tolerance,
                    concession_rate=concession_rate,
                )
                result.negotiations.append(log)
                if agreed_price is None:
                    continue

                native_paid_value = env.exchange_rates.convert(agreed_price, "USD", chosen.currency_symbol)
                fx_tax_paid = compute_fx_tax(
                    native_paid_value, env.currencies[chosen.currency_symbol], buyer.currency_zone, fx_params.fx_tax_rate
                )

                tx = Transaction(
                    buyer_id=buyer.agent_id,
                    seller_id=seller.agent_id,
                    good_name=good.name,
                    currency_symbol=chosen.currency_symbol,
                    chain_name=chosen.chain_name,
                    gas_fee=chosen.gas_fee,
                    expected_value=listing.true_price,
                    paid_value=native_paid_value,
                    timestep=day,
                    fx_tax_paid=fx_tax_paid,
                )

                validation = validate_transaction(tx, buyer.wallet, env.currencies)
                if not validation.is_valid:
                    tx.status = TransactionStatus.FAILED
                    result.transactions.append(tx)
                    continue

                settle(tx, buyer.wallet, seller.wallet)
                env.ledger.record(tx)
                result.transactions.append(tx)

                success = tx.status == TransactionStatus.SETTLED
                buyer.update_memory(chosen.currency_symbol, success)
                seller.update_memory(chosen.currency_symbol, success)
```

(The `else` branch above is copied verbatim from the pre-refactor code -- it is completely unchanged, just re-indented one level shallower now that it's no longer nested under a per-good `if use_llm:` check.)

- [ ] **Step 4: Run the regression baseline test to confirm identical behavior**

Run: `pytest tests/test_timestep_persistence.py::test_llm_path_produces_identical_transaction_count_before_and_after_refactor -v`
Expected: PASS, with `master_result.total_transactions` equal to the exact number recorded in Step 2.

- [ ] **Step 5: Run the full existing test suite to confirm no other regression**

Run: `pytest tests/ -x -q`
Expected: All tests pass (same pass count as on `main` before this task).

- [ ] **Step 6: Commit**

```bash
git add src/simulation/timestep.py tests/test_timestep_persistence.py
git commit -m "refactor: extract per-buyer LLM day-loop body into _process_buyer_llm_day"
```

---

### Task 2: Add opt-in thread-pool concurrency to `run_timestep`

**Files:**
- Modify: `src/simulation/timestep.py` (the `run_timestep` signature and the `if use_llm:` dispatch added in Task 1)
- Test: `tests/test_timestep_persistence.py`

**Interfaces:**
- Consumes: `_process_buyer_llm_day` from Task 1.
- Produces: `run_timestep(..., max_workers: int = 1)` — new keyword parameter. `max_workers=1` (default) preserves Task 1's sequential loop exactly. `max_workers > 1` dispatches each buyer's `_process_buyer_llm_day` call to a `ThreadPoolExecutor`. Consumed by Task 3 (`matrix_runner.run_matrix`).

- [ ] **Step 1: Write a test asserting concurrent execution produces the same aggregate results as sequential**

Add to `tests/test_timestep_persistence.py`:

```python
def test_max_workers_greater_than_one_produces_same_transaction_count_as_sequential():
    """Concurrency must not change WHAT happens, only how fast -- same
    fixed seed/mock decision, sequential vs max_workers=4, must agree on
    total transaction count (order of dict/list entries may differ, but
    counts must not)."""
    def _run(max_workers):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = Session(engine)
        results, failures = run_matrix(
            model_candidates=["vendor/fake-model"],
            seeds=[0],
            num_days=3,
            dry_run=True,
            exercise_llm_path=True,
            session=session,
            keep_daily_results=True,
            llm_max_workers=max_workers,
        )
        assert failures == []
        return next(r for r in results if r.cell_key == "master").total_transactions

    assert _run(max_workers=1) == _run(max_workers=8)
```

(This test references `run_matrix(llm_max_workers=...)`, which Task 3 adds — writing it now, failing until Task 3 lands, documents the end-to-end contract this task exists to support. `run_timestep`'s own direct parameter is `max_workers`, threaded through as `llm_max_workers` at the `run_matrix` layer in Task 3 to avoid ambiguity with any future non-LLM worker count.)

- [ ] **Step 2: Run test to verify it fails (until Task 3 exists)**

Run: `pytest tests/test_timestep_persistence.py::test_max_workers_greater_than_one_produces_same_transaction_count_as_sequential -v`
Expected: FAIL with `TypeError: run_matrix() got an unexpected keyword argument 'llm_max_workers'`

- [ ] **Step 3: Add `max_workers` to `run_timestep` and dispatch via `ThreadPoolExecutor`**

Add to `timestep.py`'s top-level imports: `from concurrent.futures import ThreadPoolExecutor`.

Change `run_timestep`'s signature (currently ending `polygon_client: httpx.Client | None = None,`) to add one more parameter:

```python
def run_timestep(
    env: Environment,
    day: int,
    rng: random.Random,
    max_negotiation_rounds: int = 10,
    agreement_tolerance: float = 0.01,
    concession_rate: float = 0.3,
    use_llm: bool = False,
    openrouter_client: httpx.Client | None = None,
    polygon_client: httpx.Client | None = None,
    max_workers: int = 1,
) -> TimestepResult:
```

Add one line to the docstring right after the existing `polygon_client` paragraph:

```
    `max_workers` (default 1, meaning fully sequential -- identical to this
    function's original behavior) only applies when `use_llm=True`: values
    above 1 run different buyers' `_process_buyer_llm_day` calls
    concurrently on a `ThreadPoolExecutor`, since those calls are I/O-bound
    (OpenRouter network requests) rather than CPU-bound. A single
    `threading.Lock` (constructed fresh per `run_timestep` call) serializes
    the narrow shared-state critical sections inside `_process_buyer_llm_day`
    (settlement, ledger recording, result-list appends) -- see that
    function's docstring for the full safety analysis. `max_workers` has no
    effect when `use_llm=False`: the rule-based path has no network calls,
    so there is nothing to gain from parallelizing it.
```

Replace the `if use_llm:` dispatch block from Task 1 (`for buyer in active_buyers: _process_buyer_llm_day(...)`) with:

```python
    if use_llm:
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _process_buyer_llm_day,
                        buyer,
                        env,
                        day,
                        fx_params,
                        live_price_snapshots,
                        currency_history,
                        macro_history,
                        openrouter_client,
                        max_negotiation_rounds,
                        agreement_tolerance,
                        concession_rate,
                        result,
                        lock,
                    )
                    for buyer in active_buyers
                ]
                for future in futures:
                    future.result()  # re-raises any worker exception on the main thread
        else:
            for buyer in active_buyers:
                _process_buyer_llm_day(
                    buyer,
                    env,
                    day,
                    fx_params,
                    live_price_snapshots,
                    currency_history,
                    macro_history,
                    openrouter_client,
                    max_negotiation_rounds,
                    agreement_tolerance,
                    concession_rate,
                    result,
                    lock,
                )
```

- [ ] **Step 4: Run the new test — still expected to fail until Task 3**

Run: `pytest tests/test_timestep_persistence.py::test_max_workers_greater_than_one_produces_same_transaction_count_as_sequential -v`
Expected: FAIL (same `run_matrix(llm_max_workers=...)` TypeError as Step 2 — Task 3 hasn't run yet).

- [ ] **Step 5: Run the Task 1 regression test to confirm `max_workers=1` default still matches**

Run: `pytest tests/test_timestep_persistence.py::test_llm_path_produces_identical_transaction_count_before_and_after_refactor -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/simulation/timestep.py tests/test_timestep_persistence.py
git commit -m "feat: add opt-in thread-pool concurrency to run_timestep's LLM path"
```

---

### Task 3: Thread `llm_max_workers` through `matrix_runner.run_matrix`

**Files:**
- Modify: `src/simulation/matrix_runner.py:511-525` (signature), `:773-780` (the `run_timestep` call site)
- Test: `tests/test_matrix_runner.py`

**Interfaces:**
- Consumes: `run_timestep(..., max_workers=...)` from Task 2.
- Produces: `run_matrix(..., llm_max_workers: int = 1)`.

- [ ] **Step 1: Run Task 2's Step 1 test — now the real target to make pass**

Run: `pytest tests/test_timestep_persistence.py::test_max_workers_greater_than_one_produces_same_transaction_count_as_sequential -v`
Expected: still FAIL (confirms the test is still correctly pointing at the not-yet-added parameter).

- [ ] **Step 2: Add `llm_max_workers` to `run_matrix`'s signature and thread it to `run_timestep`**

In `src/simulation/matrix_runner.py`, change the `run_matrix` signature (currently ending `checkpoint_dir: Path | None = None,`) to add:

```python
def run_matrix(
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    dry_run: bool = True,
    openrouter_client: httpx.Client | None = None,
    polygon_client: httpx.Client | None = None,
    session: Session | None = None,
    matrix_run_id: str | None = None,
    keep_daily_results: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
    exercise_llm_path: bool = False,
    mock_llm_decision: dict | None = None,
    checkpoint_dir: Path | None = None,
    llm_max_workers: int = 1,
) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]]]:
```

Add one paragraph to the docstring (after the `checkpoint_dir` paragraph):

```
    `llm_max_workers` (default 1, no behavior change) is passed straight
    through to every `run_timestep` call as `max_workers` -- see that
    function's docstring. Values above 1 parallelize LLM negotiation calls
    across buyers within each simulated day; this is the mechanism Plan 6a
    adds to make a 365-day x 3-seed x 13-cell real run feasible in
    wall-clock time.
```

In the day-loop's `run_timestep` call (currently at lines 773-780), add the new argument:

```python
                    result = run_timestep(
                        env,
                        day=day,
                        rng=rng,
                        use_llm=use_llm,
                        openrouter_client=cell_openrouter_client,
                        polygon_client=polygon_client,
                        max_workers=llm_max_workers,
                    )
```

- [ ] **Step 3: Run the concurrency-parity test**

Run: `pytest tests/test_timestep_persistence.py::test_max_workers_greater_than_one_produces_same_transaction_count_as_sequential -v`
Expected: PASS.

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass, same count as before this task (every existing `run_matrix` caller omits `llm_max_workers`, so it defaults to 1 — zero behavior change).

- [ ] **Step 5: Commit**

```bash
git add src/simulation/matrix_runner.py tests/test_timestep_persistence.py
git commit -m "feat: thread llm_max_workers through run_matrix to run_timestep"
```

---

### Task 4: Add cell/seed subsetting to `run_matrix`

**Files:**
- Modify: `src/simulation/matrix_runner.py:511-525` (signature), `:657` (the cell loop), `_build_cell_specs` usage
- Test: `tests/test_matrix_runner.py`

**Interfaces:**
- Produces: `run_matrix(..., cell_keys: list[str] | None = None)` — when given, restricts the run to only cells whose `key` is in `cell_keys` (matching `_build_cell_specs()`'s `.key` values, e.g. `"master"`, `"liquidity_vs_governance_domestic"`). `None` (default) runs all 13, unchanged. Consumed by Task 5's cross-process orchestrator, which partitions the 13 cell keys across worker processes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_matrix_runner.py`:

```python
def test_cell_keys_restricts_which_cells_run():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    results, failures = run_matrix(
        model_candidates=["vendor/fake-model"],
        seeds=[0],
        num_days=2,
        dry_run=True,
        session=session,
        cell_keys=["master", "liquidity_vs_governance_domestic"],
    )
    assert failures == []
    assert {r.cell_key for r in results} == {"master", "liquidity_vs_governance_domestic"}
```

(Adjust the import block at the top of the file if `create_engine`/`Base`/`Session` aren't already imported there — check the existing file's imports first and reuse them rather than re-importing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matrix_runner.py::test_cell_keys_restricts_which_cells_run -v`
Expected: FAIL with `TypeError: run_matrix() got an unexpected keyword argument 'cell_keys'`

- [ ] **Step 3: Add `cell_keys` parameter and filter `_build_cell_specs()`'s output**

Add `cell_keys: list[str] | None = None,` to `run_matrix`'s signature (right after `llm_max_workers`), and a docstring paragraph:

```
    `cell_keys`, if given, restricts this call to only the cells whose
    `_CellSpec.key` is in the list (e.g. `["master",
    "liquidity_vs_governance_domestic"]`) -- every other cell is skipped
    entirely, as if it didn't exist in `_build_cell_specs()`'s output.
    `None` (the default) runs all 13 cells, unchanged. This exists so a
    caller can partition the full matrix across separate processes/
    machines (see Plan 6a's cross-process orchestrator), each restricted
    to a disjoint subset of cell_keys against the same shared database.
```

Change the cell loop (currently `for spec in _build_cell_specs():`) to:

```python
    all_specs = _build_cell_specs()
    specs_to_run = all_specs if cell_keys is None else [s for s in all_specs if s.key in cell_keys]
    if cell_keys is not None:
        unknown = set(cell_keys) - {s.key for s in all_specs}
        if unknown:
            raise ValueError(f"cell_keys contains unknown cell key(s): {sorted(unknown)}")

    for spec in specs_to_run:
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_matrix_runner.py::test_cell_keys_restricts_which_cells_run -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/simulation/matrix_runner.py tests/test_matrix_runner.py
git commit -m "feat: add cell_keys subsetting parameter to run_matrix"
```

---

### Task 5: Enable SQLite WAL mode + busy-timeout for concurrent-process writes

**Files:**
- Modify: `database/session.py`
- Test: `tests/test_database_session.py` (new file)

**Interfaces:**
- Produces: `get_engine()`'s SQLite connections now have `journal_mode=WAL` and a `busy_timeout` set. No signature changes — purely a connection-configuration change.

- [ ] **Step 1: Write the failing test**

Create `tests/test_database_session.py`:

```python
from sqlalchemy import text

from database.session import get_engine


def test_sqlite_engine_uses_wal_journal_mode():
    engine = get_engine()
    with engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert mode.lower() == "wal"


def test_sqlite_engine_has_a_nonzero_busy_timeout():
    engine = get_engine()
    with engine.connect() as conn:
        timeout_ms = conn.execute(text("PRAGMA busy_timeout")).scalar()
    assert timeout_ms > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_database_session.py -v`
Expected: FAIL — `journal_mode` defaults to `"delete"`, not `"wal"`; `busy_timeout` defaults to `0`.

- [ ] **Step 3: Configure WAL mode + busy-timeout via SQLAlchemy's connect event**

Modify `database/session.py`:

```python
"""SQLAlchemy engine/session factory.

Reads DATABASE_URL from .env (python-dotenv), defaulting to a local SQLite
file so simulations run with zero setup. Swapping to Postgres later means
changing .env, not code.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.utils.constants import DEFAULT_DATABASE_URL, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

_engine = create_engine(DATABASE_URL, echo=False)


@event.listens_for(_engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record) -> None:
    """WAL mode lets concurrent OS processes write to this SQLite file
    without one writer blocking every reader (Plan 6a: separate
    run_matrix(cell_keys=...) processes share one database). busy_timeout
    makes a writer that DOES contend retry for up to 30s instead of
    immediately raising "database is locked" -- both are no-ops for a
    non-SQLite DATABASE_URL (this listener only fires for the sqlite3
    DB-API module, which is what dbapi_connection is when DATABASE_URL
    points at a .db file)."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_engine():
    return _engine


def create_all_tables() -> None:
    from database.models import Base

    Base.metadata.create_all(_engine)


def new_session() -> Session:
    return SessionLocal()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_database_session.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass — WAL mode is a storage-format detail transparent to every existing query/write.

- [ ] **Step 6: Commit**

```bash
git add database/session.py tests/test_database_session.py
git commit -m "feat: enable SQLite WAL mode + busy-timeout for concurrent-process writes"
```

---

### Task 6: Cross-process orchestrator partitioning cells/seeds across worker processes

**Files:**
- Create: `src/simulation/distributed_matrix_runner.py`
- Test: `tests/test_distributed_matrix_runner.py`

**Interfaces:**
- Consumes: `run_matrix(..., cell_keys=..., llm_max_workers=...)` from Tasks 3-4; `_build_cell_specs` from `src.simulation.matrix_runner` (for the full list of 13 cell keys to partition).
- Produces: `run_matrix_distributed(model_candidates, seeds, num_days, dry_run=True, openrouter_client_factory=None, polygon_client_factory=None, matrix_run_id=None, num_processes=4, llm_max_workers=1, checkpoint_dir=None) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]]]` — partitions the 13 cell keys into `num_processes` groups, runs each group in its own `run_matrix(cell_keys=<group>, session=<this process's own session>)` call inside a separate process, and merges all `(results, failures)` pairs.

`ProcessPoolExecutor` requires its worker function and arguments to be picklable; `httpx.Client` is not picklable, so a real client cannot be passed directly into a worker process — instead this function accepts *factory* callables (`openrouter_client_factory: Callable[[], httpx.Client] | None`), each worker calls the factory itself to build its own client after the process starts. `dry_run=True` (the default) needs no factories at all, matching `run_matrix`'s own default.

- [ ] **Step 1: Write the failing test**

Create `tests/test_distributed_matrix_runner.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.simulation.distributed_matrix_runner import run_matrix_distributed


def test_run_matrix_distributed_runs_all_13_cells_across_processes(tmp_path):
    db_path = tmp_path / "distributed_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    results, failures = run_matrix_distributed(
        model_candidates=["vendor/fake-model"],
        seeds=[0],
        num_days=2,
        dry_run=True,
        num_processes=2,
        matrix_run_id="distributed-test",
        database_url=f"sqlite:///{db_path}",
    )

    assert failures == []
    assert len(results) == 13
    cell_keys_seen = {r.cell_key for r in results}
    assert len(cell_keys_seen) == 13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributed_matrix_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.simulation.distributed_matrix_runner'`

- [ ] **Step 3: Create `src/simulation/distributed_matrix_runner.py`**

```python
"""Cross-process orchestrator partitioning the 13-cell matrix across
worker processes (Plan 6a Sec 2.2). Each worker process runs its own
disjoint subset of cell keys via `run_matrix(cell_keys=...)`, opening its
OWN database session against the SAME SQLite file (WAL mode, enabled in
`database/session.py`, is what makes concurrent-process writes to that
one shared file safe). httpx.Client objects are not picklable, so a real
client cannot cross the process boundary directly -- callers needing
dry_run=False pass factory callables instead, and each worker calls the
factory itself after the process starts.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.simulation.matrix_runner import MatrixCellResult, _build_cell_specs, run_matrix


def _partition(items: list, num_groups: int) -> list[list]:
    """Splits `items` into `num_groups` roughly-equal contiguous chunks
    (never more groups than items -- a group that would be empty is
    dropped, since spawning a process with zero work is pure overhead)."""
    num_groups = min(num_groups, len(items)) or 1
    chunk_size = -(-len(items) // num_groups)  # ceiling division
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_cell_group(
    cell_keys: list[str],
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    dry_run: bool,
    database_url: str,
    matrix_run_id: str,
    llm_max_workers: int,
    checkpoint_dir: Path | None,
    openrouter_client_factory: Callable[[], "httpx.Client"] | None,
    polygon_client_factory: Callable[[], "httpx.Client"] | None,
) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]]]:
    """Runs in a separate process: builds its OWN engine/session (engines
    aren't picklable/shareable across processes either) and its own real
    clients from the factories, if given, then calls run_matrix restricted
    to this group's cell_keys."""
    engine = create_engine(database_url)
    session = Session(engine)
    openrouter_client = openrouter_client_factory() if openrouter_client_factory is not None else None
    polygon_client = polygon_client_factory() if polygon_client_factory is not None else None

    return run_matrix(
        model_candidates=model_candidates,
        seeds=seeds,
        num_days=num_days,
        dry_run=dry_run,
        openrouter_client=openrouter_client,
        polygon_client=polygon_client,
        session=session,
        matrix_run_id=matrix_run_id,
        cell_keys=cell_keys,
        llm_max_workers=llm_max_workers,
        checkpoint_dir=checkpoint_dir,
    )


def run_matrix_distributed(
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    database_url: str,
    dry_run: bool = True,
    openrouter_client_factory: Callable[[], "httpx.Client"] | None = None,
    polygon_client_factory: Callable[[], "httpx.Client"] | None = None,
    matrix_run_id: str | None = None,
    num_processes: int = 4,
    llm_max_workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]]]:
    """Partitions the 13 matrix cells into `num_processes` groups and runs
    each group in its own OS process via `run_matrix(cell_keys=...)`,
    against the same `database_url` (must be a file-based SQLite URL, or
    another DB that supports concurrent-process writes). Each worker
    process opens its OWN SQLAlchemy engine/session against `database_url`
    -- engines, like httpx.Client, are not picklable across the process
    boundary. `database/session.py`'s WAL-mode connect-event listener is
    registered against that module's specific `_engine` instance, not
    `database_url` globally, so it has no effect on engines built inside
    worker processes; `_run_cell_group` below registers the same pragmas
    on its own locally-built engine instead.

    `matrix_run_id`, if `None`, is generated once here (not per-group) so
    every cell across every process shares one consistent prefix -- see
    `run_matrix`'s own `matrix_run_id` docstring for why a stable shared
    prefix matters for resumability.
    """
    from src.utils.helpers import generate_id

    if matrix_run_id is None:
        matrix_run_id = generate_id("matrix")

    all_cell_keys = [spec.key for spec in _build_cell_specs()]
    groups = _partition(all_cell_keys, num_processes)

    with ProcessPoolExecutor(max_workers=len(groups)) as executor:
        futures = [
            executor.submit(
                _run_cell_group,
                group,
                model_candidates,
                seeds,
                num_days,
                dry_run,
                database_url,
                matrix_run_id,
                llm_max_workers,
                checkpoint_dir,
                openrouter_client_factory,
                polygon_client_factory,
            )
            for group in groups
        ]
        all_results: list[MatrixCellResult] = []
        all_failures: list[tuple[str, int, Exception]] = []
        for future in futures:
            group_results, group_failures = future.result()
            all_results.extend(group_results)
            all_failures.extend(group_failures)

    return all_results, all_failures
```

Also add the same `@event.listens_for` pragma-setting to `_run_cell_group`, right after `engine = create_engine(database_url)`:

```python
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, connection_record) -> None:
        if not database_url.startswith("sqlite"):
            return
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_distributed_matrix_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/simulation/distributed_matrix_runner.py tests/test_distributed_matrix_runner.py
git commit -m "feat: add run_matrix_distributed, partitioning the 13-cell matrix across processes"
```

---

### Task 7: Add cost/token logging to the LLM router

**Files:**
- Modify: `src/llm/llm_router.py`
- Test: `tests/test_llm_router.py`

**Interfaces:**
- Produces: `call_model` now returns usage alongside the parsed `Decision`. `LLMCallResult` gains a `usage: LLMUsage | None` field. New `LLMUsage` model: `prompt_tokens: int, completion_tokens: int, total_tokens: int`. A new module-level `_TOTAL_USAGE` accumulator plus `get_cumulative_usage() -> LLMUsage` and `reset_cumulative_usage() -> None` functions give any caller (e.g. a long matrix run's progress callback) live visibility into total tokens spent so far, without requiring every caller to thread usage data through themselves.

Check `tests/test_llm_router.py` (or wherever `call_model`/`_parse_decision`'s existing tests live — locate via `grep -rl "_parse_decision\|def call_model" tests/`) for the exact mock-response fixture shape before writing new tests, so the new usage-parsing test reuses that same fixture pattern rather than inventing a different one.

- [ ] **Step 1: Write the failing test**

Add (matching the existing file's mock-`httpx.Response`-building convention — inspect one existing `call_model` test there first):

```python
def test_call_model_captures_token_usage_and_accumulates_across_calls():
    from src.llm.llm_router import call_model, get_cumulative_usage, reset_cumulative_usage

    reset_cumulative_usage()
    # Build a mock client/transport whose /chat/completions response includes
    # an OpenRouter-style "usage" block, using the same mock-transport
    # helper the existing call_model tests in this file already use.
    client = _mock_client_with_usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    call_model("some prompt", "vendor/fake-model", client)

    usage = get_cumulative_usage()
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150

    call_model("some prompt", "vendor/fake-model", client)
    usage_after_second_call = get_cumulative_usage()
    assert usage_after_second_call.total_tokens == 300
```

(`_mock_client_with_usage` is a new small test helper this step also adds, in the same style as whatever existing helper builds a mock `/chat/completions` response in this test file — copy that helper's structure and add an `"usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}` key to its canned response body.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_router.py::test_call_model_captures_token_usage_and_accumulates_across_calls -v`
Expected: FAIL with `ImportError: cannot import name 'get_cumulative_usage'`

- [ ] **Step 3: Add `LLMUsage`, usage parsing, and the cumulative accumulator**

In `src/llm/llm_router.py`, add near the top (after the existing `ModelEntry`/`ReliabilityChain` model definitions):

```python
class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


_cumulative_usage = LLMUsage()


def get_cumulative_usage() -> LLMUsage:
    return _cumulative_usage.model_copy()


def reset_cumulative_usage() -> None:
    global _cumulative_usage
    _cumulative_usage = LLMUsage()


def _parse_usage(response: httpx.Response) -> LLMUsage:
    body = response.json()
    usage_block = body.get("usage") or {}
    return LLMUsage(
        prompt_tokens=usage_block.get("prompt_tokens", 0),
        completion_tokens=usage_block.get("completion_tokens", 0),
        total_tokens=usage_block.get("total_tokens", 0),
    )
```

Modify `call_model` to record usage on every successful parse. Change the two `return _parse_decision(...)` call sites (the main path at line 199 and the repair path at line 213) to capture and accumulate usage before returning:

```python
        try:
            decision = _parse_decision(response)
            global _cumulative_usage
            _cumulative_usage = _cumulative_usage + _parse_usage(response)
            return decision
        except (KeyError, IndexError, ValueError) as exc:
```

and, in the repair branch:

```python
            try:
                repair_response = _post_chat_completion(client, model_id, repair_messages)
                repaired_decision = _parse_decision(repair_response)
                global _cumulative_usage
                _cumulative_usage = _cumulative_usage + _parse_usage(repair_response)
                return repaired_decision
            except (KeyError, IndexError, ValueError) as repair_exc:
```

(`global _cumulative_usage` must appear before the name is reassigned within `call_model`'s scope — add it once near the top of the function body, not inside each nested `try`, if Python's `global` scoping rules make the duplicate declaration awkward: place a single `global _cumulative_usage` statement at the very top of `call_model`'s body instead of repeating it at each call site.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_llm_router.py::test_call_model_captures_token_usage_and_accumulates_across_calls -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass — `usage` parsing degrades to zeros when a mock response has no `"usage"` key (every pre-existing mock fixture), so no existing test's fixtures need updating.

- [ ] **Step 6: Wire cumulative usage into `run_matrix`'s existing `progress_callback`**

Modify `src/simulation/matrix_runner.py`'s day-loop `progress_callback` call site (Task 3's location) to also report usage, by changing the callback contract. Add a new, separate callback rather than changing `progress_callback`'s existing 3-arg signature (which existing callers may already rely on):

In `run_matrix`'s signature, add `usage_callback: Callable[[str, int, int, "LLMUsage"], None] | None = None,` (after `progress_callback`), with a docstring paragraph:

```
    `usage_callback`, if given, is called once per simulated day (same
    timing as `progress_callback`) as `usage_callback(cell_key, seed, day,
    cumulative_usage)`, where `cumulative_usage` is
    `src.llm.llm_router.get_cumulative_usage()`'s snapshot at that point --
    the running token total across every LLM call made by this run_matrix
    invocation so far, letting a caller driving a long real run log/display
    spend visibility without polling anything itself. `None` (the default)
    is a no-op, same as `progress_callback`.
```

Add the import `from src.llm.llm_router import get_cumulative_usage` to `matrix_runner.py`'s imports, and add this line right after the existing `if progress_callback is not None:` block in the day loop:

```python
                    if usage_callback is not None:
                        usage_callback(spec.key, seed, day, get_cumulative_usage())
```

- [ ] **Step 7: Write a test for `usage_callback`**

Add to `tests/test_matrix_runner.py`:

```python
def test_usage_callback_is_called_once_per_day_with_cumulative_usage():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    calls = []

    run_matrix(
        model_candidates=["vendor/fake-model"],
        seeds=[0],
        num_days=2,
        dry_run=True,
        session=session,
        usage_callback=lambda cell_key, seed, day, usage: calls.append((cell_key, seed, day)),
    )

    # dry_run=True with no exercise_llm_path never calls the LLM router at
    # all (rule-based path), so usage_callback still fires once per day
    # per cell/seed -- it just always reports zero cumulative usage in
    # that mode. This test only asserts the callback fires with the right
    # cadence, not that usage is nonzero (that's exercised in Task 7 Step
    # 1's call_model-level test above).
    assert len(calls) == 13 * 1 * 2  # 13 cells x 1 seed x 2 days
```

- [ ] **Step 8: Run the test**

Run: `pytest tests/test_matrix_runner.py::test_usage_callback_is_called_once_per_day_with_cumulative_usage -v`
Expected: PASS.

- [ ] **Step 9: Run the full test suite and commit**

Run: `pytest tests/ -x -q`
Expected: All tests pass.

```bash
git add src/llm/llm_router.py src/simulation/matrix_runner.py tests/test_llm_router.py tests/test_matrix_runner.py
git commit -m "feat: add cumulative token-usage tracking and a usage_callback to run_matrix"
```

---

## Self-Review Notes

- **Spec coverage**: Sec 2.1 (within-day thread pool) -> Tasks 1-3; Sec 2.2 (cross-process cells/seeds + SQLite WAL) -> Tasks 4-6; Sec 3 (cost/token logging) -> Task 7. All three concurrency-spec sections have a task.
- **Placeholder scan**: Task 6's first draft docstring paragraph was garbled placeholder-style prose — replaced with a real explanation in Step 3.
- **Type consistency**: `run_timestep(max_workers=...)` (Task 2) -> `run_matrix(llm_max_workers=...)` (Task 3, deliberately different name to avoid confusion with a possible future non-LLM worker count) -> `run_matrix_distributed(llm_max_workers=...)` (Task 6, passed straight through to each worker's own `run_matrix` call) — verified consistent across all three tasks' code blocks above.
