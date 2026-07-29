# Phase 3 Plan 1: Merge Phase 2 + Foundation Persistence Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the completed-but-unmerged Phase 2 LLM negotiation layer into `main`, then extend the SQLAlchemy persistence layer with the raw-data tables and fields the rest of Phase 3 needs (full prompt text instead of just a hash, wired-up hallucination telemetry, run provenance, shock/intervention logs, per-agent-per-day state snapshots, episodic memory logs) so every later Phase 3 subsystem has a stable, already-tested place to write data.

**Architecture:** Everything happens on `main` after the merge. New tables/fields are added to `database/models.py` (SQLAlchemy 2.0 Declarative `Base` subclasses) with matching DAO classes in `database/repository.py`, following the codebase's existing convention: a plain-Pydantic `...LogEntry` input model + a `...Repository` class exposing a single `record()` method. Calling code never constructs an ORM row directly.

**Tech Stack:** SQLAlchemy >=2.0 (Declarative `Mapped`/`mapped_column` style), Pydantic >=2.6, pytest, SQLite in-memory for tests (`sqlite:///:memory:`, matching `tests/test_llm_persistence.py`'s existing `_session()` helper).

## Global Constraints

- Python >=3.12, Pydantic >=2.6, SQLAlchemy >=2.0, PyYAML >=6.0, python-dotenv >=1.0, httpx >=0.27 (from `pyproject.toml`) — do not add new dependencies without checking with the user first.
- This is the final data-collection phase per the approved spec (`docs/superpowers/specs/2026-07-29-phase3-full-scale-simulation-design.md`). Do not modify, add to, or reinterpret scope beyond what that spec and this plan describe without checking with the user first.
- Follow the existing repository pattern exactly: an SQLAlchemy `Base` subclass in `database/models.py`; a Pydantic `...LogEntry` model + a `...Repository` class with one `record()` method in `database/repository.py`. Never construct or query ORM records directly from simulation/experiment code.
- No hardcoded economic constants (project-wide rule) — this plan only touches schema/persistence plumbing, not economic logic, so no economic constant should appear in any new code here.
- Tests use `sqlite:///:memory:` + `Base.metadata.create_all(engine)` + `Session(engine)`, matching the existing `_session()` pattern in `tests/test_llm_persistence.py`.
- Never use `git push --force`, `git reset --hard`, or skip hooks (`--no-verify`) during the merge in Task 1.

---

## Pre-flight: merge conflict check (already done, informational only)

A dry-run `git merge-tree` between `main` and `worktree-phase2-llm-negotiation` (merge-base `9e95709`) shows **zero conflicts** — `main` has only added `docs/superpowers/specs/2026-07-29-phase3-full-scale-simulation-design.md` since the branches diverged, and the worktree branch adds the full Phase 2 implementation cleanly on top. Task 1 below expects a clean merge.

## File Structure

- **Modify:** `database/models.py` — add `SimulationRunRecord`, `InterventionLogRecord`, `TimestepLogRecord`, `AgentStateRecord`, `AgentMemoryLogRecord`; extend `LLMDecisionRecord` (+`system_prompt`), `HallucinationRecord` (+`decision_id`, +`direction`, +`is_hallucination`, `transaction_id` becomes nullable), `TransactionRecord` (+`fx_tax_paid`).
- **Modify:** `database/repository.py` — add `SimulationRunLogEntry`/`SimulationRunRepository`, `InterventionLogEntry`/`InterventionLogRepository`, `TimestepLogEntry`/`TimestepLogRepository`, `AgentStateLogEntry`/`AgentStateRepository`, `AgentMemoryLogEntry`/`AgentMemoryLogRepository`, `HallucinationLogEntry`/`HallucinationRepository`; extend `LLMDecisionLogEntry` (+`system_prompt`).
- **Modify:** `src/transactions/transaction.py` — add `fx_tax_paid: float = 0.0` to `Transaction`.
- **Modify:** `experiments/experiment_007_governance_prompting.py` — pass `system_prompt=prompt` into `LLMDecisionLogEntry`; persist a `HallucinationLogEntry` via a new `HallucinationRepository` when `run_cell` computes a hallucination (currently computed but silently dropped — nothing persists it today).
- **Test:** `tests/test_llm_persistence.py` (extend existing test), `tests/test_transactions.py` (extend existing test), new `tests/test_hallucination_persistence.py`, new `tests/test_provenance_persistence.py`, new `tests/test_timestep_persistence.py`, new `tests/test_agent_state_persistence.py`.

---

### Task 1: Merge Phase 2 into main

**Files:**
- No file changes beyond the merge itself.

- [ ] **Step 1: Confirm working tree is clean and on `main`**

Run: `git status`
Expected: `On branch main`, `nothing to commit, working tree clean`

- [ ] **Step 2: Merge the phase2 branch**

Run:
```bash
git merge worktree-phase2-llm-negotiation --no-ff -m "Merge Phase 2 LLM negotiation layer into main"
```
Expected: `Merge made by the 'ort' strategy.` with no `CONFLICT` lines (confirmed conflict-free by the pre-flight `git merge-tree` check above).

- [ ] **Step 3: Install dependencies for the merged code**

Run: `pip install -e ".[llm,observability,market-data]"` (from repo root, inside the project's `.venv`)
Expected: exits 0, `httpx`, `wandb`, `requests` installed alongside the existing Phase 1 dependencies.

- [ ] **Step 4: Run the full test suite to confirm the merge is sound**

Run: `pytest -q`
Expected: all tests pass except the one marked `@pytest.mark.live` in `tests/test_experiment_007_live.py`, which is skipped unless `RUN_LIVE_LLM_TESTS=1` is set (it is not, per `.env.example`'s default). No failures.

- [ ] **Step 5: No commit needed** — the merge commit from Step 2 is already the commit for this task.

---

### Task 2: Store the full raw system prompt on every LLM decision

**Context:** `LLMDecisionRecord` currently stores `rendered_prompt_hash` (a SHA-256 hash) but not the prompt text itself. A hash lets you *verify* a prompt hasn't changed but not *read* what the model actually saw — the user asked for maximally detailed raw data, so the full text needs to be stored alongside the hash, not instead of it.

**Files:**
- Modify: `database/models.py` (`LLMDecisionRecord`)
- Modify: `database/repository.py` (`LLMDecisionLogEntry`)
- Modify: `experiments/experiment_007_governance_prompting.py` (`run_cell`)
- Test: `tests/test_llm_persistence.py`

**Interfaces:**
- Produces: `LLMDecisionLogEntry.system_prompt: str` — every future task that constructs an `LLMDecisionLogEntry` must supply this field.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_llm_persistence.py`, inside `test_llm_decision_repository_persists_full_record` (add one line to the existing `LLMDecisionLogEntry(...)` call and one new assertion at the end):

```python
    entry = LLMDecisionLogEntry(
        decision_id="dec-1",
        simulation_id="sim-1",
        timestep=3,
        agent_id="buyer-1",
        agent_type="buyer",
        requested_model="claude-sonnet-5",
        actual_model="claude-sonnet-5",
        fallback_used=False,
        fallback_reason=None,
        model_attempts=["claude-sonnet-5"],
        prompt_version="buyer_prompt@v1",
        rendered_prompt_hash="abc123",
        system_prompt="You are a buyer agent. Candidates: USDC on ethereum...",
        action="OFFER",
        currency="USDC",
        chain="ethereum",
        amount=100.0,
        price=99.5,
        reported_reasoning="USDC offers the best governance/liquidity trade-off.",
        negotiation_id="neg-1",
        round=1,
        risk_profile="low",
        utility_type="crra",
        utility_parameters={"risk_aversion": 3.0},
        scenario="baseline",
        domestic_or_cross_border="domestic",
        governance_prompt_enabled=True,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(LLMDecisionRecord).all()
    assert len(rows) == 1
    assert rows[0].decision_id == "dec-1"
    assert rows[0].actual_model == "claude-sonnet-5"
    assert rows[0].model_attempts == ["claude-sonnet-5"]
    assert rows[0].fallback_used is False
    assert rows[0].system_prompt == "You are a buyer agent. Candidates: USDC on ethereum..."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_persistence.py::test_llm_decision_repository_persists_full_record -v`
Expected: FAIL with `pydantic_core._pydantic_core.ValidationError` (missing `system_prompt`) or `TypeError: 'system_prompt' is an invalid keyword argument` once the field is added to the test but not yet to the model — confirm it fails for the right reason (field doesn't exist yet).

- [ ] **Step 3: Add the field to the DB model**

In `database/models.py`, inside `LLMDecisionRecord`, add one line directly after `rendered_prompt_hash`:

```python
    rendered_prompt_hash: Mapped[str] = mapped_column(String)
    system_prompt: Mapped[str] = mapped_column(String)
```

- [ ] **Step 4: Add the field to the Pydantic log entry**

In `database/repository.py`, inside `LLMDecisionLogEntry`, add one line directly after `rendered_prompt_hash`:

```python
    rendered_prompt_hash: str
    system_prompt: str
```

(No change needed to `LLMDecisionRepository.record` — it already does `**entry.model_dump()`, so the new field flows through automatically.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_llm_persistence.py -v`
Expected: PASS (both tests in the file).

- [ ] **Step 6: Wire the real prompt text through in experiment_007**

In `experiments/experiment_007_governance_prompting.py`, inside `run_cell`, in the `LLMDecisionLogEntry(...)` construction, add one line directly after `rendered_prompt_hash=hash_rendered_prompt(prompt),`:

```python
                rendered_prompt_hash=hash_rendered_prompt(prompt),
                system_prompt=prompt,
```

- [ ] **Step 7: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass.

- [ ] **Step 8: Commit**

```bash
git add database/models.py database/repository.py experiments/experiment_007_governance_prompting.py tests/test_llm_persistence.py
git commit -m "feat: persist full raw system prompt text on LLM decision records"
```

---

### Task 3: Add fx_tax_paid to the transaction ledger

**Context:** `Experiment.md`'s `transactions_ledger` schema requires an `fx_tax_paid` column for cross-border conversion friction. That conversion-tax *computation* is built later (in the matrix-runner plan, when cross-border sandboxes are implemented) — this task only adds the column and the plumbing to persist it, defaulting to `0.0`, so no later migration is needed.

**Files:**
- Modify: `src/transactions/transaction.py` (`Transaction`)
- Modify: `database/models.py` (`TransactionRecord`)
- Modify: `database/repository.py` (`TransactionRepository.record`)
- Test: `tests/test_transactions.py`

**Interfaces:**
- Produces: `Transaction.fx_tax_paid: float` (default `0.0`) — later cross-border settlement code sets this explicitly; domestic transactions leave it at the default.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transactions.py`:

```python
def test_transaction_defaults_fx_tax_paid_to_zero():
    tx = _make_tx()

    assert tx.fx_tax_paid == 0.0


def test_transaction_accepts_explicit_fx_tax_paid():
    tx = Transaction(
        buyer_id="buyer-1",
        seller_id="seller-1",
        good_name="cloud_compute",
        currency_symbol="USDC",
        chain_name="ethereum",
        gas_fee=0.5,
        expected_value=100.0,
        paid_value=100.0,
        timestep=0,
        fx_tax_paid=2.5,
    )

    assert tx.fx_tax_paid == 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transactions.py -v`
Expected: FAIL with `ValidationError` / `AttributeError: 'Transaction' object has no attribute 'fx_tax_paid'`.

- [ ] **Step 3: Add the field to Transaction**

In `src/transactions/transaction.py`, add one line directly after `status`:

```python
    status: TransactionStatus = TransactionStatus.PENDING
    fx_tax_paid: float = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transactions.py -v`
Expected: PASS (all tests in the file, including the two new ones).

- [ ] **Step 5: Add the column to TransactionRecord and wire the repository**

In `database/models.py`, inside `TransactionRecord`, add one line directly after `status`:

```python
    status: Mapped[str] = mapped_column(String)
    fx_tax_paid: Mapped[float] = mapped_column(Float, default=0.0)
```

In `database/repository.py`, inside `TransactionRepository.record`, add one line to the `TransactionRecord(...)` construction, directly after `status=tx.status.value,`:

```python
                status=tx.status.value,
                fx_tax_paid=tx.fx_tax_paid,
```

- [ ] **Step 6: Add a persistence-level test**

Add to `tests/test_llm_persistence.py`:

```python
from database.models import TransactionRecord
from database.repository import TransactionRepository
from src.transactions.transaction import Transaction


def test_transaction_repository_persists_fx_tax_paid():
    session = _session()
    repo = TransactionRepository(session)
    tx = Transaction(
        buyer_id="buyer-1",
        seller_id="seller-1",
        good_name="cloud_compute",
        currency_symbol="USDC",
        chain_name="ethereum",
        gas_fee=0.5,
        expected_value=100.0,
        paid_value=100.0,
        timestep=0,
        fx_tax_paid=1.75,
    )

    repo.record(tx)
    session.commit()

    rows = session.query(TransactionRecord).all()
    assert len(rows) == 1
    assert rows[0].fx_tax_paid == 1.75
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_llm_persistence.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/transactions/transaction.py database/models.py database/repository.py tests/test_transactions.py tests/test_llm_persistence.py
git commit -m "feat: add fx_tax_paid to the transaction ledger for cross-border conversion friction"
```

---

### Task 4: Wire up hallucination telemetry persistence

**Context:** `HallucinationRecord` exists in `database/models.py` but nothing in the codebase ever writes to it — `experiment_007_governance_prompting.py`'s `run_cell` calls `detect_hallucination(...)` and returns `hallucination_direction` in a printed dict, but the `HallucinationResult` is discarded, never persisted. There is also no `HallucinationRepository`. This task closes both gaps and extends the schema to match `Experiment.md`'s `hallucination_telemetry` table: a `direction` string, an `is_hallucination` boolean, and a `decision_id` link (since hallucination detection can happen on any LLM decision, not only ones that become a settled `Transaction` — so `transaction_id` must become nullable).

**Files:**
- Modify: `database/models.py` (`HallucinationRecord`)
- Modify: `database/repository.py` (add `HallucinationLogEntry`, `HallucinationRepository`)
- Modify: `experiments/experiment_007_governance_prompting.py` (`run_cell`, `main`)
- Test: new `tests/test_hallucination_persistence.py`

**Interfaces:**
- Consumes: `HallucinationResult` from `src/llm/hallucination_detector.py` (`expected_value`, `paid_value`, `absolute_error`, `percentage_error`, `direction: HallucinationDirection`, `currency_symbol`, `actual_model`) — already implemented, unchanged.
- Produces: `HallucinationLogEntry` (Pydantic input model), `HallucinationRepository.record(entry: HallucinationLogEntry) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hallucination_persistence.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, HallucinationRecord
from database.repository import HallucinationLogEntry, HallucinationRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_hallucination_repository_persists_without_a_transaction():
    """A hallucination can be detected on a raw LLM decision before any
    transaction settles -- transaction_id must be optional."""
    session = _session()
    repo = HallucinationRepository(session)
    entry = HallucinationLogEntry(
        decision_id="dec-1",
        transaction_id=None,
        expected_price=100.0,
        paid_price=150.0,
        overpayment_pct=50.0,
        direction="OVERPAYMENT",
        is_hallucination=True,
        currency_symbol="USDC",
        model_name="anthropic/claude-sonnet-5",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(HallucinationRecord).all()
    assert len(rows) == 1
    assert rows[0].transaction_id is None
    assert rows[0].decision_id == "dec-1"
    assert rows[0].direction == "OVERPAYMENT"
    assert rows[0].is_hallucination is True


def test_hallucination_repository_persists_accurate_decisions_too():
    """Accurate (non-hallucinated) decisions are recorded too, with
    is_hallucination=False -- the table is a complete telemetry record, not
    just a log of failures."""
    session = _session()
    repo = HallucinationRepository(session)
    entry = HallucinationLogEntry(
        decision_id="dec-2",
        transaction_id="tx-1",
        expected_price=100.0,
        paid_price=102.0,
        overpayment_pct=2.0,
        direction="ACCURATE",
        is_hallucination=False,
        currency_symbol="EURC",
        model_name="openai/gpt-5.6-luna",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(HallucinationRecord).all()
    assert rows[0].is_hallucination is False
    assert rows[0].transaction_id == "tx-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hallucination_persistence.py -v`
Expected: FAIL with `ImportError: cannot import name 'HallucinationLogEntry'`.

- [ ] **Step 3: Extend HallucinationRecord**

In `database/models.py`, replace the existing `HallucinationRecord` class body:

```python
class HallucinationRecord(Base):
    """Every hallucination check, whether or not it ties to a settled
    transaction -- detection happens on any LLM decision, and a decision can
    be rejected/countered long before (or without ever) settling."""

    __tablename__ = "hallucinations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str | None] = mapped_column(String, nullable=True)
    transaction_id: Mapped[str | None] = mapped_column(String, ForeignKey("transactions.id"), nullable=True)
    expected_price: Mapped[float] = mapped_column(Float)
    paid_price: Mapped[float] = mapped_column(Float)
    overpayment_pct: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String)
    is_hallucination: Mapped[bool] = mapped_column(Boolean)
    currency_symbol: Mapped[str] = mapped_column(String)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
```

- [ ] **Step 4: Add HallucinationLogEntry and HallucinationRepository**

In `database/repository.py`, add near `LLMDecisionLogEntry`/`LLMDecisionRepository`:

```python
class HallucinationLogEntry(BaseModel):
    decision_id: str | None = None
    transaction_id: str | None = None
    expected_price: float
    paid_price: float
    overpayment_pct: float
    direction: str
    is_hallucination: bool
    currency_symbol: str
    model_name: str | None = None


class HallucinationRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: HallucinationLogEntry) -> None:
        self.session.add(HallucinationRecord(**entry.model_dump()))
```

Add `HallucinationRecord` to the existing `from database.models import (...)` block at the top of `database/repository.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hallucination_persistence.py -v`
Expected: PASS.

- [ ] **Step 6: Wire real persistence into experiment_007**

In `experiments/experiment_007_governance_prompting.py`, change `run_cell`'s signature and body to accept a `HallucinationRepository` and persist the result when one is computed. Replace:

```python
def run_cell(
    model_id: str,
    governance_prompt_enabled: bool,
    client: httpx.Client,
    repository: LLMDecisionRepository | None = None,
) -> dict:
```

with:

```python
def run_cell(
    model_id: str,
    governance_prompt_enabled: bool,
    client: httpx.Client,
    repository: LLMDecisionRepository | None = None,
    hallucination_repository: HallucinationRepository | None = None,
) -> dict:
```

Then find this existing block inside `run_cell`:

```python
    hallucination = None
    if decision.action in (DecisionAction.OFFER, DecisionAction.COUNTER_OFFER, DecisionAction.ACCEPT):
        hallucination = detect_hallucination(
            GOOD_TRUE_PRICE, decision.price, currency_symbol=decision.proposed_currency, actual_model=model_id
        )
```

and add immediately after it (still inside `run_cell`, before the `if repository is not None:` block):

```python
    if hallucination is not None and hallucination_repository is not None:
        hallucination_repository.record(
            HallucinationLogEntry(
                decision_id=None,
                transaction_id=None,
                expected_price=hallucination.expected_value,
                paid_price=hallucination.paid_value,
                overpayment_pct=hallucination.percentage_error,
                direction=hallucination.direction.value,
                is_hallucination=hallucination.direction != HallucinationDirection.ACCURATE,
                currency_symbol=hallucination.currency_symbol or "",
                model_name=hallucination.actual_model,
            )
        )
```

Add `HallucinationDirection` to the existing `from src.llm.hallucination_detector import detect_hallucination` import line (`from src.llm.hallucination_detector import HallucinationDirection, detect_hallucination`), and add `HallucinationLogEntry, HallucinationRepository` to the existing `from database.repository import (...)` import block.

- [ ] **Step 7: Wire it into main()**

In `experiments/experiment_007_governance_prompting.py`'s `main()`, directly after `repository = LLMDecisionRepository(session)`, add:

```python
    hallucination_repository = HallucinationRepository(session)
```

Then find the call site(s) that invoke `run_cell(...)` inside `main()` and add `hallucination_repository=hallucination_repository` as an argument.

- [ ] **Step 8: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass.

- [ ] **Step 9: Commit**

```bash
git add database/models.py database/repository.py experiments/experiment_007_governance_prompting.py tests/test_hallucination_persistence.py
git commit -m "feat: wire hallucination telemetry through to persistence"
```

---

### Task 5: Add simulation_runs provenance table

**Context:** `Experiment.md` §6 requires every simulation run to record `random_seed`, exact model identifiers used, `prompt_version_hash`, `git_commit_hash`, `config_hash`, and a timestamp. Because Phase 3 assigns **one model per agent** (not one model per run, per the approved design spec §3.4), the field is named `model_roster_summary` rather than the spec's singular `openrouter_model_id` — it holds a short descriptor (e.g. a hash of the full agent-to-model assignment, or `"100 agents across 90 models, see agent_states"`) rather than one model ID. This deviation is documented here and in the field's docstring; nothing else about the provenance requirement changes.

**Files:**
- Modify: `database/models.py` (add `SimulationRunRecord`)
- Modify: `database/repository.py` (add `SimulationRunLogEntry`, `SimulationRunRepository`)
- Test: new `tests/test_provenance_persistence.py`

**Interfaces:**
- Produces: `SimulationRunLogEntry`, `SimulationRunRepository.record(entry: SimulationRunLogEntry) -> None`. Later plans (matrix runner) construct one `SimulationRunLogEntry` per run and record it before the run starts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_provenance_persistence.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, SimulationRunRecord
from database.repository import SimulationRunLogEntry, SimulationRunRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_simulation_run_repository_persists_provenance():
    session = _session()
    repo = SimulationRunRepository(session)
    entry = SimulationRunLogEntry(
        run_id="run-master-seed-0",
        scenario_name="master_simulation",
        research_mode="factual",
        random_seed=0,
        model_roster_summary="100 agents across 90 OpenRouter models",
        prompt_version_hash="deadbeef",
        git_commit_hash="abc1234",
        config_hash="feedface",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(SimulationRunRecord).all()
    assert len(rows) == 1
    assert rows[0].run_id == "run-master-seed-0"
    assert rows[0].research_mode == "factual"
    assert rows[0].random_seed == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provenance_persistence.py -v`
Expected: FAIL with `ImportError: cannot import name 'SimulationRunRecord'`.

- [ ] **Step 3: Add SimulationRunRecord**

In `database/models.py`, add:

```python
class SimulationRunRecord(Base):
    """Provenance metadata captured once per run, before the first timestep.

    model_roster_summary is a short descriptor of the run's agent-to-model
    assignment (Phase 3 assigns one model per agent, not one per run) rather
    than a single openrouter_model_id -- see docs/superpowers/plans/
    2026-07-29-phase3-01-foundation-persistence.md Task 5 for why this
    deviates from Experiment.md's singular field name.
    """

    __tablename__ = "simulation_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    scenario_name: Mapped[str] = mapped_column(String)
    research_mode: Mapped[str] = mapped_column(String)
    random_seed: Mapped[int] = mapped_column(Integer)
    model_roster_summary: Mapped[str] = mapped_column(String)
    prompt_version_hash: Mapped[str] = mapped_column(String)
    git_commit_hash: Mapped[str] = mapped_column(String)
    config_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)
```

- [ ] **Step 4: Add SimulationRunLogEntry and SimulationRunRepository**

In `database/repository.py`, add:

```python
class SimulationRunLogEntry(BaseModel):
    run_id: str
    scenario_name: str
    research_mode: str
    random_seed: int
    model_roster_summary: str
    prompt_version_hash: str
    git_commit_hash: str
    config_hash: str


class SimulationRunRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: SimulationRunLogEntry) -> None:
        self.session.add(SimulationRunRecord(**entry.model_dump(), created_at=datetime.now(timezone.utc)))
```

Add `SimulationRunRecord` to the `from database.models import (...)` block at the top of the file.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_provenance_persistence.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add database/models.py database/repository.py tests/test_provenance_persistence.py
git commit -m "feat: add simulation_runs provenance table"
```

---

### Task 6: Add intervention_logs table

**Files:**
- Modify: `database/models.py` (add `InterventionLogRecord`)
- Modify: `database/repository.py` (add `InterventionLogEntry`, `InterventionLogRepository`)
- Test: `tests/test_provenance_persistence.py` (extend)

**Interfaces:**
- Produces: `InterventionLogEntry`, `InterventionLogRepository.record(entry: InterventionLogEntry) -> None`. The shock-engine plan's `apply_shock`/event-log wiring calls this once per fired shock.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_provenance_persistence.py`:

```python
from database.models import InterventionLogRecord
from database.repository import InterventionLogEntry, InterventionLogRepository


def test_intervention_log_repository_persists_shock_event():
    session = _session()
    repo = InterventionLogRepository(session)
    entry = InterventionLogEntry(
        run_id="run-master-seed-0",
        timestep=212,
        shock_type="inflation",
        target_currency=None,
        target_issuer=None,
        magnitude=0.085,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(InterventionLogRecord).all()
    assert len(rows) == 1
    assert rows[0].shock_type == "inflation"
    assert rows[0].timestep == 212


def test_intervention_log_repository_persists_targeted_shock():
    session = _session()
    repo = InterventionLogRepository(session)
    entry = InterventionLogEntry(
        run_id="run-master-seed-0",
        timestep=610,
        shock_type="depeg_event",
        target_currency="USDT",
        target_issuer=None,
        magnitude=0.08,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(InterventionLogRecord).all()
    assert rows[0].target_currency == "USDT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provenance_persistence.py -v`
Expected: FAIL with `ImportError: cannot import name 'InterventionLogRecord'`.

- [ ] **Step 3: Add InterventionLogRecord**

In `database/models.py`, add:

```python
class InterventionLogRecord(Base):
    __tablename__ = "intervention_logs"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"))
    timestep: Mapped[int] = mapped_column(Integer)
    shock_type: Mapped[str] = mapped_column(String)
    target_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    target_issuer: Mapped[str | None] = mapped_column(String, nullable=True)
    magnitude: Mapped[float] = mapped_column(Float)
```

- [ ] **Step 4: Add InterventionLogEntry and InterventionLogRepository**

In `database/repository.py`, add:

```python
class InterventionLogEntry(BaseModel):
    run_id: str
    timestep: int
    shock_type: str
    target_currency: str | None = None
    target_issuer: str | None = None
    magnitude: float


class InterventionLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: InterventionLogEntry) -> None:
        self.session.add(InterventionLogRecord(**entry.model_dump()))
```

Add `InterventionLogRecord` to the `from database.models import (...)` block.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_provenance_persistence.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add database/models.py database/repository.py tests/test_provenance_persistence.py
git commit -m "feat: add intervention_logs table for step-indexed shock events"
```

---

### Task 7: Add timestep_logs table

**Files:**
- Modify: `database/models.py` (add `TimestepLogRecord`)
- Modify: `database/repository.py` (add `TimestepLogEntry`, `TimestepLogRepository`)
- Test: new `tests/test_timestep_persistence.py`

**Interfaces:**
- Produces: `TimestepLogEntry`, `TimestepLogRepository.record(entry: TimestepLogEntry) -> None`. Primary key is `(run_id, timestep)`, matching `Experiment.md` §8 exactly (one row per run per day).

- [ ] **Step 1: Write the failing test**

Create `tests/test_timestep_persistence.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, TimestepLogRecord
from database.repository import TimestepLogEntry, TimestepLogRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_timestep_log_repository_persists_daily_macro_state():
    session = _session()
    repo = TimestepLogRepository(session)
    entry = TimestepLogEntry(
        run_id="run-master-seed-0",
        timestep=5,
        inflation_rate=0.03,
        confidence_index=0.95,
        eth_gas_fee_gwei=25.0,
        solana_gas_fee_usd=0.0007,
        eur_usd_exchange_rate=1.08,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(TimestepLogRecord).all()
    assert len(rows) == 1
    assert rows[0].run_id == "run-master-seed-0"
    assert rows[0].timestep == 5
    assert rows[0].eur_usd_exchange_rate == 1.08


def test_timestep_log_primary_key_is_run_id_and_timestep():
    session = _session()
    repo = TimestepLogRepository(session)
    repo.record(
        TimestepLogEntry(
            run_id="run-a",
            timestep=1,
            inflation_rate=0.02,
            confidence_index=1.0,
            eth_gas_fee_gwei=20.0,
            solana_gas_fee_usd=0.0005,
            eur_usd_exchange_rate=1.08,
        )
    )
    repo.record(
        TimestepLogEntry(
            run_id="run-b",
            timestep=1,
            inflation_rate=0.05,
            confidence_index=0.8,
            eth_gas_fee_gwei=40.0,
            solana_gas_fee_usd=0.001,
            eur_usd_exchange_rate=1.07,
        )
    )
    session.commit()

    rows = session.query(TimestepLogRecord).order_by(TimestepLogRecord.run_id).all()
    assert len(rows) == 2
    assert [r.run_id for r in rows] == ["run-a", "run-b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_timestep_persistence.py -v`
Expected: FAIL with `ImportError: cannot import name 'TimestepLogRecord'`.

- [ ] **Step 3: Add TimestepLogRecord**

In `database/models.py`, add:

```python
class TimestepLogRecord(Base):
    __tablename__ = "timestep_logs"

    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"), primary_key=True)
    timestep: Mapped[int] = mapped_column(Integer, primary_key=True)
    inflation_rate: Mapped[float] = mapped_column(Float)
    confidence_index: Mapped[float] = mapped_column(Float)
    eth_gas_fee_gwei: Mapped[float] = mapped_column(Float)
    solana_gas_fee_usd: Mapped[float] = mapped_column(Float)
    eur_usd_exchange_rate: Mapped[float] = mapped_column(Float)
```

- [ ] **Step 4: Add TimestepLogEntry and TimestepLogRepository**

In `database/repository.py`, add:

```python
class TimestepLogEntry(BaseModel):
    run_id: str
    timestep: int
    inflation_rate: float
    confidence_index: float
    eth_gas_fee_gwei: float
    solana_gas_fee_usd: float
    eur_usd_exchange_rate: float


class TimestepLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: TimestepLogEntry) -> None:
        self.session.add(TimestepLogRecord(**entry.model_dump()))
```

Add `TimestepLogRecord` to the `from database.models import (...)` block.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_timestep_persistence.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add database/models.py database/repository.py tests/test_timestep_persistence.py
git commit -m "feat: add timestep_logs table for daily macro state telemetry"
```

---

### Task 8: Add agent_states table

**Context:** `Experiment.md` §8 specifies fixed `usd_balance`/`eur_balance`/`gold_balance` columns, but this codebase's currency universe has nine currencies (`USDC`, `USDT`, `FDUSD`, `DAI`, `EURC`, `EURT`, `PAXG`, `XAUT`, `TDUSD`), not three. A fixed three-column schema would silently drop balance data for six of them, which conflicts directly with the user's "raw data as detailed as possible" requirement. This task uses a `wallet_balances: dict[str, float]` JSON column instead, capturing every currency an agent holds. This deviation is documented here and in the field's docstring.

**Files:**
- Modify: `database/models.py` (add `AgentStateRecord`)
- Modify: `database/repository.py` (add `AgentStateLogEntry`, `AgentStateRepository`)
- Test: new `tests/test_agent_state_persistence.py`

**Interfaces:**
- Produces: `AgentStateLogEntry`, `AgentStateRepository.record(entry: AgentStateLogEntry) -> None`. Primary key `(run_id, timestep, agent_id)`, one row per agent per day.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_state_persistence.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, AgentStateRecord
from database.repository import AgentStateLogEntry, AgentStateRepository


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_agent_state_repository_persists_full_wallet_snapshot():
    session = _session()
    repo = AgentStateRepository(session)
    entry = AgentStateLogEntry(
        run_id="run-master-seed-0",
        timestep=10,
        agent_id="buyer-1",
        risk_profile="low",
        crra_sigma=1.5,
        real_purchasing_power=987.3,
        wallet_balances={"USDC": 800.0, "EURC": 200.0, "PAXG": 1.5},
        utility_score=0.72,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(AgentStateRecord).all()
    assert len(rows) == 1
    assert rows[0].wallet_balances == {"USDC": 800.0, "EURC": 200.0, "PAXG": 1.5}
    assert rows[0].crra_sigma == 1.5


def test_agent_state_primary_key_is_run_timestep_agent():
    session = _session()
    repo = AgentStateRepository(session)
    repo.record(
        AgentStateLogEntry(
            run_id="run-a",
            timestep=1,
            agent_id="buyer-1",
            risk_profile="low",
            crra_sigma=0.0,
            real_purchasing_power=1000.0,
            wallet_balances={"USDC": 1000.0},
            utility_score=1.0,
        )
    )
    repo.record(
        AgentStateLogEntry(
            run_id="run-a",
            timestep=2,
            agent_id="buyer-1",
            risk_profile="low",
            crra_sigma=0.0,
            real_purchasing_power=990.0,
            wallet_balances={"USDC": 990.0},
            utility_score=0.99,
        )
    )
    session.commit()

    rows = session.query(AgentStateRecord).order_by(AgentStateRecord.timestep).all()
    assert len(rows) == 2
    assert [r.timestep for r in rows] == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_state_persistence.py -v`
Expected: FAIL with `ImportError: cannot import name 'AgentStateRecord'`.

- [ ] **Step 3: Add AgentStateRecord**

In `database/models.py`, add:

```python
class AgentStateRecord(Base):
    """Per-agent-per-day snapshot. wallet_balances is a JSON dict keyed by
    currency symbol rather than Experiment.md's fixed usd_balance/
    eur_balance/gold_balance columns -- this codebase's currency universe
    has nine currencies, not three, and a fixed schema would silently drop
    six of them. See docs/superpowers/plans/
    2026-07-29-phase3-01-foundation-persistence.md Task 8."""

    __tablename__ = "agent_states"

    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"), primary_key=True)
    timestep: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"), primary_key=True)
    risk_profile: Mapped[str] = mapped_column(String)
    crra_sigma: Mapped[float] = mapped_column(Float)
    real_purchasing_power: Mapped[float] = mapped_column(Float)
    wallet_balances: Mapped[dict] = mapped_column(JSON)
    utility_score: Mapped[float] = mapped_column(Float)
```

- [ ] **Step 4: Add AgentStateLogEntry and AgentStateRepository**

In `database/repository.py`, add:

```python
class AgentStateLogEntry(BaseModel):
    run_id: str
    timestep: int
    agent_id: str
    risk_profile: str
    crra_sigma: float
    real_purchasing_power: float
    wallet_balances: dict[str, float]
    utility_score: float


class AgentStateRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: AgentStateLogEntry) -> None:
        self.session.add(AgentStateRecord(**entry.model_dump()))
```

Add `AgentStateRecord` to the `from database.models import (...)` block.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_agent_state_persistence.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add database/models.py database/repository.py tests/test_agent_state_persistence.py
git commit -m "feat: add agent_states table for per-agent-per-day snapshots"
```

---

### Task 9: Add agent_memory_logs table

**Files:**
- Modify: `database/models.py` (add `AgentMemoryLogRecord`)
- Modify: `database/repository.py` (add `AgentMemoryLogEntry`, `AgentMemoryLogRepository`)
- Test: `tests/test_agent_state_persistence.py` (extend)

**Interfaces:**
- Produces: `AgentMemoryLogEntry`, `AgentMemoryLogRepository.record(entry: AgentMemoryLogEntry) -> None`. Later plans (shock engine's crisis-relevant memory extension) call this once per notable memory event.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_state_persistence.py`:

```python
from database.models import AgentMemoryLogRecord
from database.repository import AgentMemoryLogEntry, AgentMemoryLogRepository


def test_agent_memory_log_repository_persists_episodic_text():
    session = _session()
    repo = AgentMemoryLogRepository(session)
    entry = AgentMemoryLogEntry(
        run_id="run-master-seed-0",
        timestep=12,
        agent_id="buyer-1",
        memory_type="Depeg",
        memory_text="On day 12 I was mid-transaction in USDT when it depegged 8%.",
    )

    repo.record(entry)
    session.commit()

    rows = session.query(AgentMemoryLogRecord).all()
    assert len(rows) == 1
    assert rows[0].memory_type == "Depeg"
    assert rows[0].memory_text == "On day 12 I was mid-transaction in USDT when it depegged 8%."


def test_agent_memory_log_repository_allows_multiple_entries_per_agent():
    session = _session()
    repo = AgentMemoryLogRepository(session)
    repo.record(
        AgentMemoryLogEntry(
            run_id="run-master-seed-0", timestep=5, agent_id="buyer-1", memory_type="Network",
            memory_text="USDC is currently accepted by 97% of local merchants.",
        )
    )
    repo.record(
        AgentMemoryLogEntry(
            run_id="run-master-seed-0", timestep=6, agent_id="buyer-1", memory_type="GasSpike",
            memory_text="Ethereum gas exploded to 180 Gwei in timestep 391.",
        )
    )
    session.commit()

    rows = session.query(AgentMemoryLogRecord).filter_by(agent_id="buyer-1").all()
    assert len(rows) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_state_persistence.py -v`
Expected: FAIL with `ImportError: cannot import name 'AgentMemoryLogRecord'`.

- [ ] **Step 3: Add AgentMemoryLogRecord**

In `database/models.py`, add:

```python
class AgentMemoryLogRecord(Base):
    __tablename__ = "agent_memory_logs"

    memory_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.run_id"))
    timestep: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"))
    memory_type: Mapped[str] = mapped_column(String)
    memory_text: Mapped[str] = mapped_column(String)
```

- [ ] **Step 4: Add AgentMemoryLogEntry and AgentMemoryLogRepository**

In `database/repository.py`, add:

```python
class AgentMemoryLogEntry(BaseModel):
    run_id: str
    timestep: int
    agent_id: str
    memory_type: str
    memory_text: str


class AgentMemoryLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: AgentMemoryLogEntry) -> None:
        self.session.add(AgentMemoryLogRecord(**entry.model_dump()))
```

Add `AgentMemoryLogRecord` to the `from database.models import (...)` block.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_agent_state_persistence.py -v`
Expected: PASS.

- [ ] **Step 6: Run the entire suite one final time for this plan**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass — this confirms every table/field added across Tasks 2-9 coexists cleanly.

- [ ] **Step 7: Commit**

```bash
git add database/models.py database/repository.py tests/test_agent_state_persistence.py
git commit -m "feat: add agent_memory_logs table for episodic memory records"
```

---

## What comes after this plan

This plan only builds the persistence foundation. Subsequent Phase 3 plans (each written and approved separately, per the design spec's decomposition):

1. **Shock engine + trust ledger + historical context** — the 8 new shock types, `TrustLedger`, `event_log.py`, and the `CurrencyHistory`/`MacroHistory` prompt extensions from `Untitled document.md`, writing into `intervention_logs` and `agent_memory_logs` via the repositories built here.
2. **Agent population generation** — the 100-agent population (roles, currency zones, per-agent σ, per-agent OpenRouter model assignment with preflight verification).
3. **Matrix runner / experiment orchestration** — the master simulation + 7 sandboxes + cross-border repeats, 365 days, 5 seeds each, writing into `simulation_runs`, `timestep_logs`, and `agent_states` via the repositories built here.
4. **Econometrics engine** — H1-H5 regression outputs (β, SE, 95% CI, p-value, R²).
5. **Full-scale run launch** — an explicit, separate go/no-go checkpoint with the user before any billed API calls happen.
