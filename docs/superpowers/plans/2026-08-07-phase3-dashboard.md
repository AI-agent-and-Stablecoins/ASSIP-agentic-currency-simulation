# Phase 3 Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit dashboard (`dashboard/`) with Start/Stop/Pause/Resume controls and live progress views for the Phase 3 experiment matrix, without changing the simulation core (`run_matrix`, `run_timestep`, `distributed_matrix_runner.py`) at all.

**Architecture:** A subprocess-based runner (`dashboard/runner.py`) wraps the existing `run_matrix`/`run_matrix_distributed` entry points and writes a small JSON status file (PID, dry-run flag, best-effort token usage, final results). The Streamlit UI (`dashboard/app.py`) launches/stops that subprocess and reads two things: the status file (for control-plane state) and the project's own SQLite database directly (for live progress — day counts, transaction/decision counts), via read-only queries. The dashboard and the simulation process are fully decoupled; closing/restarting one never affects the other.

**Tech Stack:** `streamlit>=1.36` (new), `psutil>=6.0` (new, for cross-platform process-liveness checks), plus the project's existing `sqlalchemy`/`pydantic`/`httpx` stack.

## Global Constraints

- No changes to `src/simulation/matrix_runner.py`, `src/simulation/timestep.py`, or `src/simulation/distributed_matrix_runner.py` — this plan is purely additive tooling on top of already-existing hooks (`progress_callback`, `usage_callback`, checkpoint/resume).
- No dollar-cost estimate anywhere in the UI — no per-model pricing table exists in this codebase.
- "Pause" and "Stop" are the same underlying action (process termination) — do not build a separate suspend/resume-mid-execution mechanism.
- A real (non-dry-run) launch requires the user to type the exact `matrix_run_id` as a confirmation step before the launch can proceed — never a single-click real launch.
- Follow the project's existing atomic-file-write pattern for the status file (temp file + `Path.replace()`, matching `src/simulation/matrix_runner.py`'s `_save_checkpoint`) so a crash mid-write never leaves a half-written, corrupted status file.

---

### Task 1: `dashboard/status_store.py` — status file schema + read/write

**Files:**
- Create: `dashboard/__init__.py` (empty — makes `dashboard` an importable package, matching `src/__init__.py`/`database/__init__.py`'s convention)
- Create: `dashboard/status_store.py`
- Test: `tests/test_dashboard_status_store.py`

**Interfaces:**
- Produces:
  - `class StatusFileCorruptedError(Exception)` — raised by `read_status` when the file exists but isn't valid JSON matching the expected shape.
  - `write_status(matrix_run_id: str, **fields) -> None` — read-modify-write merge of `fields` into the existing status dict for `matrix_run_id` (creates it if absent), always sets `last_updated` to the current UTC ISO-8601 timestamp, atomic write.
  - `read_status(matrix_run_id: str) -> dict | None` — `None` if no status file exists yet for this `matrix_run_id`; raises `StatusFileCorruptedError` if the file exists but can't be parsed.
  - `set_active_run_id(matrix_run_id: str) -> None` / `get_active_run_id() -> str | None` — a single pointer file recording the most recently started/resumed run, so the UI can default to showing it after a page reload without the user needing to retype the ID.
  - `state_dir() -> Path` — `dashboard/state/`, created if missing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dashboard_status_store.py`:

```python
import pytest

from dashboard.status_store import (
    StatusFileCorruptedError,
    get_active_run_id,
    read_status,
    set_active_run_id,
    state_dir,
    write_status,
)


def test_read_status_returns_none_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    assert read_status("no-such-run") is None


def test_write_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    write_status("run-1", pid=1234, state="running", dry_run=True)

    status = read_status("run-1")
    assert status["matrix_run_id"] == "run-1"
    assert status["pid"] == 1234
    assert status["state"] == "running"
    assert status["dry_run"] is True
    assert "last_updated" in status


def test_write_status_merges_fields_rather_than_replacing(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    write_status("run-1", pid=1234, state="running")
    write_status("run-1", state="stopped")

    status = read_status("run-1")
    assert status["pid"] == 1234  # preserved from the first write
    assert status["state"] == "stopped"  # updated by the second write


def test_read_status_raises_a_clear_error_on_corrupted_file(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run-1.json").write_text("not valid json{{{", encoding="utf-8")

    with pytest.raises(StatusFileCorruptedError):
        read_status("run-1")


def test_active_run_id_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    assert get_active_run_id() is None

    set_active_run_id("run-1")
    assert get_active_run_id() == "run-1"

    set_active_run_id("run-2")
    assert get_active_run_id() == "run-2"


def test_state_dir_is_created_if_missing(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "state"
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", target)
    assert not target.exists()

    result = state_dir()

    assert result == target
    assert target.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_status_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard'`

- [ ] **Step 3: Create `dashboard/__init__.py`**

Empty file.

- [ ] **Step 4: Implement `dashboard/status_store.py`**

```python
"""JSON-backed status file for one dashboard-tracked simulation run.

One file per `matrix_run_id`, at `dashboard/state/<matrix_run_id>.json`.
This file is the control-plane state the dashboard's Streamlit process
and the subprocess runner (`dashboard/runner.py`) share -- NOT live
simulation progress, which is read directly from the project's own
SQLite database instead (see `dashboard/queries.py`). This file covers
only what the database doesn't hold: which PID owns the currently-tracked
run, the dry-run flag, best-effort token usage, and the final
results/failures once the run completes.

Schema (all fields optional except matrix_run_id, since write_status
merges incrementally):
{
    "matrix_run_id": str,
    "pid": int,
    "state": "running" | "stopped" | "completed" | "failed" | "crashed",
    "dry_run": bool,
    "checkpoint_dir": str | None,
    "started_at": str (UTC ISO-8601),
    "last_updated": str (UTC ISO-8601),
    "cumulative_usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
    "failures": list[[cell_key, seed, message]] | None,
    "error": str | None
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.constants import REPO_ROOT

_STATE_DIR = REPO_ROOT / "dashboard" / "state"
_ACTIVE_RUN_POINTER = "_active_run_id.json"


class StatusFileCorruptedError(Exception):
    def __init__(self, matrix_run_id: str, path: Path):
        self.matrix_run_id = matrix_run_id
        self.path = path
        super().__init__(f"Status file for {matrix_run_id!r} at {path} is corrupted/unreadable")


def state_dir() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR


def _status_path(matrix_run_id: str) -> Path:
    return state_dir() / f"{matrix_run_id}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)  # atomic on both POSIX and Windows


def write_status(matrix_run_id: str, **fields) -> None:
    path = _status_path(matrix_run_id)
    existing = read_status(matrix_run_id) or {"matrix_run_id": matrix_run_id}
    existing.update(fields)
    existing["last_updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(path, existing)


def read_status(matrix_run_id: str) -> dict | None:
    path = _status_path(matrix_run_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise StatusFileCorruptedError(matrix_run_id, path) from exc


def set_active_run_id(matrix_run_id: str) -> None:
    _atomic_write_json(state_dir() / _ACTIVE_RUN_POINTER, {"matrix_run_id": matrix_run_id})


def get_active_run_id() -> str | None:
    path = state_dir() / _ACTIVE_RUN_POINTER
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["matrix_run_id"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_status_store.py -v`
Expected: PASS (6/6).

- [ ] **Step 6: Run the full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass (purely additive new module/files).

- [ ] **Step 7: Commit**

```bash
git add dashboard/__init__.py dashboard/status_store.py tests/test_dashboard_status_store.py
git commit -m "feat: add dashboard status_store for control-plane state"
```

---

### Task 2: `dashboard/queries.py` — read-only live-progress queries

**Files:**
- Create: `dashboard/queries.py`
- Test: `tests/test_dashboard_queries.py`

**Interfaces:**
- Consumes: `database.models.TimestepLogRecord`, `LLMDecisionRecord` (existing).
- Produces:
  - `class CellSeedProgress(pydantic.BaseModel)`: `cell_key: str, seed: int, run_id: str, current_day: int, total_llm_decisions: int`
  - `get_progress_for_run(session: Session, matrix_run_id: str) -> list[CellSeedProgress]` — one row per `(cell_key, seed)` combination that has AT LEAST one persisted day so far under this `matrix_run_id` (a combination with zero persisted rows yet — not started — is simply absent from the result; the caller is expected to know the full target set from its own launch config to distinguish "not started" from "this list is exhaustive").

  **Note on scope**: `database.models.TransactionRecord` has NO run-scoping column at all (confirmed by reading `database/models.py` directly — its own docstring explains why: `buyer_id`/`seller_id` are the same unscoped agent-id strings that collide across cells sharing a seed, the same root issue the Plan 6a schema fix addressed for `agents`/`wallets`, but `transactions` itself was never fixed since nothing read it before). Adding that column would be a real schema change, out of scope for this plan (user decision, 2026-08-07) — this task deliberately does NOT report a transaction count. Only `current_day` (from `TimestepLogRecord`, which IS properly `run_id`-scoped) and `total_llm_decisions` (from `LLMDecisionRecord.simulation_id`, which IS the run_id) are shown.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard_queries.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from dashboard.queries import get_progress_for_run
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_get_progress_for_run_reports_current_day_and_decision_count_per_cell_seed():
    session = _session()
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=3,
        dry_run=True,
        session=session,
        matrix_run_id="progress-test",
        cell_keys=["master"],
    )

    progress = get_progress_for_run(session, "progress-test")

    assert len(progress) == 1
    row = progress[0]
    assert row.cell_key == "master"
    assert row.seed == 0
    assert row.run_id == "progress-test-master-seed0"
    assert row.current_day == 2  # 3 days completed -> timesteps 0, 1, 2
    assert row.total_llm_decisions == 0  # dry_run without exercise_llm_path never calls the LLM router


def test_get_progress_for_run_returns_empty_list_for_an_unknown_matrix_run_id():
    session = _session()
    progress = get_progress_for_run(session, "no-such-matrix-run")
    assert progress == []


def test_get_progress_for_run_only_includes_this_matrix_run_ids_rows():
    session = _session()
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        session=session,
        matrix_run_id="run-a",
        cell_keys=["master"],
    )
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        session=session,
        matrix_run_id="run-b",
        cell_keys=["master"],
    )

    progress_a = get_progress_for_run(session, "run-a")
    assert len(progress_a) == 1
    assert progress_a[0].run_id == "run-a-master-seed0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_queries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.queries'`

- [ ] **Step 3: Implement `dashboard/queries.py`**

```python
"""Read-only live-progress queries against the project's own SQLite
database -- this is the dashboard's source of truth for "how far has each
cell/seed gotten", since it's shared between single-process (`run_matrix`)
and cross-process (`run_matrix_distributed`) runs alike, and needs zero
new callback plumbing in either. Every `run_id` this project ever writes
is `f"{matrix_run_id}-{cell_key}-seed{seed}"` (see
`src/simulation/matrix_runner.py`), which is parsed back apart here.
"""

from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import LLMDecisionRecord, TimestepLogRecord


class CellSeedProgress(BaseModel):
    cell_key: str
    seed: int
    run_id: str
    current_day: int
    total_llm_decisions: int


def _parse_run_id(run_id: str, matrix_run_id: str) -> tuple[str, int] | None:
    """Splits f"{matrix_run_id}-{cell_key}-seed{seed}" back into
    (cell_key, seed). Returns None if run_id doesn't start with this
    matrix_run_id's prefix (a different run_matrix invocation sharing the
    same database) or doesn't match the expected "-seed{N}" suffix shape."""
    prefix = f"{matrix_run_id}-"
    if not run_id.startswith(prefix):
        return None
    remainder = run_id[len(prefix) :]
    if "-seed" not in remainder:
        return None
    cell_key, _, seed_str = remainder.rpartition("-seed")
    if not cell_key or not seed_str.isdigit():
        return None
    return cell_key, int(seed_str)


def get_progress_for_run(session: Session, matrix_run_id: str) -> list[CellSeedProgress]:
    run_ids = [
        row[0]
        for row in session.query(TimestepLogRecord.run_id)
        .filter(TimestepLogRecord.run_id.like(f"{matrix_run_id}-%"))
        .distinct()
        .all()
    ]

    results = []
    for run_id in run_ids:
        parsed = _parse_run_id(run_id, matrix_run_id)
        if parsed is None:
            continue
        cell_key, seed = parsed

        max_day = session.query(func.max(TimestepLogRecord.timestep)).filter(
            TimestepLogRecord.run_id == run_id
        ).scalar()
        decision_count = session.query(func.count(LLMDecisionRecord.id)).filter(
            LLMDecisionRecord.simulation_id == run_id
        ).scalar()

        results.append(
            CellSeedProgress(
                cell_key=cell_key,
                seed=seed,
                run_id=run_id,
                current_day=max_day or 0,
                total_llm_decisions=decision_count or 0,
            )
        )

    return results
```

`TimestepLogRecord.run_id` and `LLMDecisionRecord.id`/`LLMDecisionRecord.simulation_id` are confirmed exact column names from `database/models.py` (`TimestepLogRecord`'s primary key is `(run_id, timestep)`; `LLMDecisionRecord`'s primary key is `id`, and `simulation_id` is its run_id-equivalent field per that model's own docstring). `TransactionRecord` is deliberately not imported or queried here — see the "Note on scope" above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_dashboard_queries.py -v`
Expected: PASS (3/3).

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat: add dashboard read-only live-progress queries"
```

---

### Task 3: `dashboard/runner.py` — subprocess entrypoint

**Files:**
- Create: `dashboard/runner.py`
- Test: `tests/test_dashboard_runner.py`

**Interfaces:**
- Consumes: `dashboard.status_store.write_status` (Task 1); `src.simulation.matrix_runner.run_matrix`; `src.simulation.distributed_matrix_runner.run_matrix_distributed`; `src.llm.llm_router.build_openrouter_client`, `get_cumulative_usage`; `src.llm.market_intelligence.build_polygon_client`; `database.session.get_engine`, `create_all_tables`, `new_session`.
- Produces: a `main()` function invoked via `python -m dashboard.runner <args>`, parsed by `argparse`. Writes status updates via `status_store.write_status` as it runs, and a final `state` (`"completed"` or `"failed"`) plus `failures` when `run_matrix`/`run_matrix_distributed` returns.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard_runner.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

from src.utils.constants import REPO_ROOT


def test_runner_dry_run_writes_a_completed_status(tmp_path, monkeypatch):
    """Runs dashboard/runner.py as a REAL subprocess (matching how
    process_control.py will actually invoke it in Task 4), pointed at a
    throwaway SQLite file and a throwaway status-file directory, and
    confirms it writes a final "completed" status with zero failures."""
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path / "state")
    db_path = tmp_path / "runner_test.db"
    matrix_run_id = "runner-dry-run-test"

    env = {**__import__("os").environ, "DATABASE_URL": f"sqlite:///{db_path}"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dashboard.runner",
            "--matrix-run-id",
            matrix_run_id,
            "--seeds",
            "0",
            "--num-days",
            "1",
            "--cell-keys",
            "master",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr

    from dashboard.status_store import read_status

    # This test's own monkeypatch of _STATE_DIR only affects THIS process,
    # not the subprocess -- so read the subprocess's real status file
    # location directly instead of relying on the patched path.
    real_status_path = REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.json"
    try:
        status = json.loads(real_status_path.read_text(encoding="utf-8"))
        assert status["state"] == "completed"
        assert status["failures"] == []
    finally:
        real_status_path.unlink(missing_ok=True)
```

(This test writes to the REAL `dashboard/state/` directory, since the subprocess doesn't inherit the parent test process's `monkeypatch` — matching the same constraint documented in `tests/test_distributed_matrix_runner.py`'s sanitization test for `ProcessPoolExecutor`'s `spawn` semantics. The `finally` block cleans up the real file it created.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_runner.py -v`
Expected: FAIL — `dashboard/runner.py` doesn't exist yet, so the subprocess exits non-zero with a `No module named dashboard.runner` error on stderr, and the assertion on `result.returncode == 0` fails.

- [ ] **Step 3: Implement `dashboard/runner.py`**

```python
"""Subprocess entrypoint wrapping run_matrix/run_matrix_distributed.

Invoked as `python -m dashboard.runner <args>` by
`dashboard/process_control.py`. Writes live status updates to
`dashboard/status_store` as it runs: this is the ONLY place token-usage
and final-completion state get reported (live progress -- day counts,
transaction/decision counts -- is read directly from the database by
`dashboard/queries.py` instead, needing no cooperation from this script
at all).
"""

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

from dashboard import status_store
from src.utils.constants import CONFIG_ROOT, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")


def _load_model_candidates() -> list[str]:
    """Full real-model roster for a real launch; a single placeholder for
    a dry run (dry_run's rule-based path never calls the LLM router at
    all unless exercise_llm_path is also set, which this dashboard never
    does, so the actual model ID value is irrelevant there)."""
    with open(CONFIG_ROOT / "llm" / "model_roster_full.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [m["id"] for m in data["models"]]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--cell-keys", default=None, help="Comma-separated; omit for all 13 cells")
    parser.add_argument("--seeds", required=True, help="Comma-separated integers")
    parser.add_argument("--num-days", required=True, type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num-processes", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    matrix_run_id = args.matrix_run_id
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    cell_keys = [c.strip() for c in args.cell_keys.split(",")] if args.cell_keys else None
    dry_run = args.dry_run
    checkpoint_dir = REPO_ROOT / "checkpoints" / matrix_run_id

    status_store.write_status(
        matrix_run_id,
        pid=os.getpid(),
        state="running",
        dry_run=dry_run,
        checkpoint_dir=str(checkpoint_dir),
        failures=None,
        error=None,
    )
    status_store.set_active_run_id(matrix_run_id)

    from src.llm.llm_router import LLMUsage

    def _write_usage(usage) -> None:
        status_store.write_status(
            matrix_run_id,
            cumulative_usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )

    _distributed_usage_total = LLMUsage()

    def _distributed_usage_callback(group_usage) -> None:
        # run_matrix_distributed's usage_callback receives ONE WORKER
        # GROUP's own final total per call, not a running grand total
        # across all groups (each worker process has its own separate
        # token counter -- see that function's own docstring) -- accumulate
        # here so the status file always reflects everything completed so
        # far, not just whichever group finished most recently.
        nonlocal _distributed_usage_total
        _distributed_usage_total = _distributed_usage_total + group_usage
        _write_usage(_distributed_usage_total)

    def _single_process_usage_callback(cell_key, seed, day, usage) -> None:
        # `usage` here is already this process's own running cumulative
        # total (run_matrix's usage_callback contract passes
        # src.llm.llm_router.get_cumulative_usage()'s snapshot directly),
        # so it can be written as-is with no extra accumulation needed.
        _write_usage(usage)

    try:
        model_candidates = _load_model_candidates()

        if args.distributed:
            from database.session import get_engine

            from src.simulation.distributed_matrix_runner import run_matrix_distributed

            openrouter_client_factory = None
            polygon_client_factory = None
            if not dry_run:
                openrouter_client_factory, polygon_client_factory = _real_client_factories()

            results, failures = run_matrix_distributed(
                model_candidates=model_candidates,
                seeds=seeds,
                num_days=args.num_days,
                database_url=str(get_engine().url),
                dry_run=dry_run,
                openrouter_client_factory=openrouter_client_factory,
                polygon_client_factory=polygon_client_factory,
                matrix_run_id=matrix_run_id,
                num_processes=args.num_processes,
                checkpoint_dir=checkpoint_dir,
                usage_callback=_distributed_usage_callback,
            )
        else:
            from database.session import create_all_tables, new_session

            from src.simulation.matrix_runner import run_matrix

            create_all_tables()
            session = new_session()

            openrouter_client = None
            polygon_client = None
            if not dry_run:
                openrouter_client, polygon_client = _real_clients()

            results, failures = run_matrix(
                model_candidates=model_candidates,
                seeds=seeds,
                num_days=args.num_days,
                dry_run=dry_run,
                openrouter_client=openrouter_client,
                polygon_client=polygon_client,
                session=session,
                matrix_run_id=matrix_run_id,
                cell_keys=cell_keys,
                checkpoint_dir=checkpoint_dir,
                usage_callback=_single_process_usage_callback,
            )
    except Exception as exc:  # noqa: BLE001 -- must always record a final
        # status rather than leaving "running" stale if launch/setup itself
        # blows up before run_matrix's own per-cell handling ever engages.
        status_store.write_status(matrix_run_id, state="failed", error=f"{type(exc).__name__}: {exc}")
        raise

    status_store.write_status(
        matrix_run_id,
        state="completed",
        failures=[[cell_key, seed, str(exc)] for cell_key, seed, exc in failures],
    )


def _real_clients():
    from src.llm.llm_router import build_openrouter_client
    from src.llm.market_intelligence import build_polygon_client

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    polygon_key = os.getenv("POLYGON_API_KEY") or os.getenv("Polygon_API_KEY")
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    if not polygon_key:
        raise RuntimeError("POLYGON_API_KEY is not set in .env")
    return build_openrouter_client(openrouter_key), build_polygon_client(polygon_key)


def _real_client_factories():
    def openrouter_client_factory():
        return _real_clients()[0]

    def polygon_client_factory():
        return _real_clients()[1]

    return openrouter_client_factory, polygon_client_factory


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_dashboard_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add dashboard/runner.py tests/test_dashboard_runner.py
git commit -m "feat: add dashboard/runner.py subprocess entrypoint"
```

---

### Task 4: `dashboard/process_control.py` — start/stop/resume

**Files:**
- Create: `dashboard/process_control.py`
- Test: `tests/test_dashboard_process_control.py`

**Interfaces:**
- Consumes: `dashboard.status_store` (Task 1).
- Produces:
  - `class RunConfig(pydantic.BaseModel)`: `matrix_run_id: str, cell_keys: list[str] | None, seeds: list[int], num_days: int, dry_run: bool, distributed: bool = False, num_processes: int = 4`
  - `start(config: RunConfig) -> None` — builds the `python -m dashboard.runner ...` command line from `config`, launches it via `subprocess.Popen` (detached so it outlives the launching Streamlit rerun), writes the initial status (`state="running"`, `pid=...`).
  - `stop(matrix_run_id: str) -> None` — reads the PID from the status file and terminates that process; writes `state="stopped"`. A no-op (no error raised) if the process is already dead.
  - `resume(matrix_run_id: str) -> None` — reads the existing status for `matrix_run_id` (must already exist, since Resume needs to know the prior config), rebuilds a `RunConfig` from it, and calls `start` again with the SAME `matrix_run_id` so `run_matrix`'s own checkpoint-resume logic (unchanged) picks up where it left off.
  - `is_alive(pid: int) -> bool` — cross-platform liveness check via `psutil`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dashboard_process_control.py`:

```python
import time

import pytest

from dashboard import status_store
from dashboard.process_control import RunConfig, is_alive, resume, start, stop


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.status_store._STATE_DIR", tmp_path)
    yield


def test_is_alive_is_true_for_the_current_process_and_false_for_a_bogus_pid():
    import os

    assert is_alive(os.getpid()) is True
    assert is_alive(999_999_999) is False


def test_start_launches_a_real_process_and_records_its_pid():
    """`start()`'s own status_store.write_status call (Task 4's `start`
    function) happens synchronously in THIS test process, using the
    `_isolated_state_dir` fixture's patched location -- so the PID is
    already on record the moment `start()` returns, no polling needed.
    The spawned subprocess itself is real (matching Task 3's own runner
    test) and writes its OWN separate status update to the REAL
    dashboard/state/ directory once it starts, since a fresh subprocess
    can't inherit this test's monkeypatch -- that real file is cleaned up
    in the `finally` block below so this test leaves no residue.
    """
    matrix_run_id = "process-control-start-test"
    config = RunConfig(
        matrix_run_id=matrix_run_id,
        cell_keys=["master"],
        seeds=[0],
        num_days=1,
        dry_run=True,
    )

    try:
        start(config)

        status = status_store.read_status(matrix_run_id)
        assert status is not None
        assert is_alive(status["pid"])

        stop(matrix_run_id)
        time.sleep(1.0)
        final_status = status_store.read_status(matrix_run_id)
        assert final_status["state"] in ("stopped", "completed")
    finally:
        from src.utils.constants import REPO_ROOT

        (REPO_ROOT / "dashboard" / "state" / f"{matrix_run_id}.json").unlink(missing_ok=True)


def test_stop_is_a_noop_when_no_process_is_recorded():
    # Must not raise even though no run has ever been started under this id.
    stop("never-started")


def test_resume_requires_an_existing_status_record():
    with pytest.raises(ValueError, match="no prior status recorded"):
        resume("never-started")
```

(Note: `test_start_launches_a_real_process_and_records_its_pid` genuinely spawns a subprocess and waits briefly — it is not mocked, since the whole point is proving `start`/`stop` actually control a real OS process. Give it a generous `timeout` if your test runner needs one; a 1-simulated-day dry run against `cell_keys=["master"]` completes in well under the polling window above.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_process_control.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.process_control'`

- [ ] **Step 3: Implement `dashboard/process_control.py`**

```python
"""Start/stop/resume control for the dashboard's subprocess runner
(`dashboard/runner.py`). This module owns the ONLY place a simulation
process is spawned or killed from the dashboard -- Streamlit's own
rerun-on-every-interaction model never touches the subprocess directly.
"""

import subprocess
import sys

import psutil
from pydantic import BaseModel

from dashboard import status_store
from src.utils.constants import REPO_ROOT


class RunConfig(BaseModel):
    matrix_run_id: str
    cell_keys: list[str] | None = None
    seeds: list[int]
    num_days: int
    dry_run: bool
    distributed: bool = False
    num_processes: int = 4


def is_alive(pid: int) -> bool:
    return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE


def _build_command(config: RunConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dashboard.runner",
        "--matrix-run-id",
        config.matrix_run_id,
        "--seeds",
        ",".join(str(s) for s in config.seeds),
        "--num-days",
        str(config.num_days),
    ]
    if config.cell_keys:
        command += ["--cell-keys", ",".join(config.cell_keys)]
    command.append("--dry-run" if config.dry_run else "--real")
    if config.distributed:
        command += ["--distributed", "--num-processes", str(config.num_processes)]
    return command


def start(config: RunConfig) -> None:
    command = _build_command(config)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach: must outlive this Streamlit rerun
        )
    except OSError as exc:
        status_store.write_status(config.matrix_run_id, state="failed", error=f"launch failed: {exc}")
        raise
    # seeds/num_days/cell_keys/distributed/num_processes are written here
    # (not by runner.py itself) specifically so `resume()` can reconstruct
    # a full RunConfig later purely from the status file, without needing
    # runner.py to duplicate this bookkeeping.
    status_store.write_status(
        config.matrix_run_id,
        pid=process.pid,
        state="running",
        dry_run=config.dry_run,
        seeds=config.seeds,
        num_days=config.num_days,
        cell_keys=config.cell_keys,
        distributed=config.distributed,
        num_processes=config.num_processes,
    )


def stop(matrix_run_id: str) -> None:
    status = status_store.read_status(matrix_run_id)
    if status is None or status.get("pid") is None:
        return
    pid = status["pid"]
    if is_alive(pid):
        psutil.Process(pid).terminate()
    status_store.write_status(matrix_run_id, state="stopped")


def resume(matrix_run_id: str) -> None:
    status = status_store.read_status(matrix_run_id)
    if status is None:
        raise ValueError(f"Cannot resume {matrix_run_id!r}: no prior status recorded for it")
    config = RunConfig(
        matrix_run_id=matrix_run_id,
        dry_run=status.get("dry_run", True),
        seeds=status.get("seeds", [0]),
        num_days=status.get("num_days", 1),
        cell_keys=status.get("cell_keys"),
        distributed=status.get("distributed", False),
        num_processes=status.get("num_processes", 4),
    )
    start(config)
```

**Known, accepted limitation**: `start()` writes the status file with the real PID right after `subprocess.Popen()` returns, and `runner.py`'s own `main()` (Task 3) also writes its own initial status moments later, once the child process finishes starting up. Both use `write_status`'s read-modify-write merge (Task 1), so there's a narrow theoretical race if the child's write happened to land between the parent's read and write. In practice the child's Python interpreter startup (importing `yaml`, `dotenv`, and several `src.*` modules) takes far longer than the parent's write, and any single lost field self-heals on the next write (`runner.py` writes again every time `usage_callback` fires, and again at completion) — not worth adding file locking for a dashboard control-plane file. Not fixed further here.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_dashboard_process_control.py -v`
Expected: PASS (4/4).

- [ ] **Step 5: Add `psutil` as a dependency**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
dashboard = ["streamlit>=1.36", "psutil>=6.0"]
```

(Both `streamlit` and `psutil` are added together here since Task 4 is the first task needing `psutil`, and Task 5 will need `streamlit` — one dependency-group edit covers both rather than touching `pyproject.toml` twice.)

Also update `requirements.txt`'s header comment area is not required (it deliberately mirrors only `[project.dependencies]`, not optional groups, per its own comment) — no change needed there.

Install it: `pip install -e ".[dashboard]"`

- [ ] **Step 6: Run the full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add dashboard/process_control.py tests/test_dashboard_process_control.py pyproject.toml
git commit -m "feat: add dashboard start/stop/resume process control"
```

---

### Task 5: `dashboard/app.py` — Streamlit UI

**Files:**
- Create: `dashboard/app.py`
- Test: `tests/test_dashboard_app.py`
- Modify: `.gitignore` (add `dashboard/state/`)

**Interfaces:**
- Consumes: `dashboard.status_store` (Task 1), `dashboard.queries.get_progress_for_run` (Task 2), `dashboard.process_control.RunConfig/start/stop/resume/is_alive` (Task 4), `database.session.new_session`, `src.simulation.matrix_runner._build_cell_specs`, `src.utils.helpers.generate_id`.
- Produces: a runnable Streamlit script, launched by the user as `streamlit run dashboard/app.py` (not imported by anything else in the codebase).

- [ ] **Step 1: Add `dashboard/state/` to `.gitignore`**

Add to `.gitignore` (anywhere in the file, e.g. near the end):

```
# Dashboard control-plane state (PIDs, run status) -- machine-local, never checked in
dashboard/state/
```

- [ ] **Step 2: Write the failing test**

Streamlit ships a headless testing harness (`streamlit.testing.v1.AppTest`, available in `streamlit>=1.28`, so `streamlit>=1.36` already has it) that runs a script without a browser and lets you inspect/interact with rendered widgets.

Create `tests/test_dashboard_app.py`:

```python
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_app_renders_the_start_configuration_form():
    at = AppTest.from_file("dashboard/app.py")
    at.run()

    assert not at.exception
    # The matrix_run_id text input and the dry-run toggle must both be present.
    assert any("matrix_run_id" in inp.label.lower() or "run id" in inp.label.lower() for inp in at.text_input)
    assert any("dry" in cb.label.lower() for cb in at.checkbox)


def test_clicking_start_in_dry_run_mode_calls_process_control_start_without_extra_confirmation():
    with patch("dashboard.process_control.start") as mock_start:
        at = AppTest.from_file("dashboard/app.py")
        at.run()

        # Dry run is the default state -- clicking Start should call
        # process_control.start directly, no confirmation step required.
        start_buttons = [b for b in at.button if b.label.lower() == "start"]
        assert len(start_buttons) == 1
        start_buttons[0].click().run()

        assert mock_start.called


def test_turning_dry_run_off_requires_typing_the_matrix_run_id_before_a_real_launch_is_possible():
    with patch("dashboard.process_control.start") as mock_start:
        at = AppTest.from_file("dashboard/app.py")
        at.run()

        dry_run_checkboxes = [cb for cb in at.checkbox if "dry" in cb.label.lower()]
        assert len(dry_run_checkboxes) == 1
        dry_run_checkboxes[0].uncheck().run()

        # With dry_run off, the plain "Start" button must be gone/disabled,
        # replaced by a confirmation flow -- clicking whatever primary
        # action button remains, WITHOUT having typed the confirmation
        # text, must not launch anything.
        confirm_buttons = [b for b in at.button if "confirm" in b.label.lower()]
        assert len(confirm_buttons) == 1
        confirm_buttons[0].click().run()

        assert not mock_start.called
```

**Note on the patch target**: `patch("dashboard.process_control.start")` works because `streamlit.testing.v1.AppTest` re-executes `dashboard/app.py`'s full source (including its top-level `from dashboard.process_control import start` line) fresh on every `.run()` call, rather than relying on Python's normal one-time module-import caching — so the patch is picked up as long as it's active during `.run()`. If this assumption turns out wrong when actually run (the mock's `.called` stays `False` even though the button click clearly executed), patch `"dashboard.app.start"` instead (the name as re-bound inside `app.py`'s own namespace) and note the correction in your task report.

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_dashboard_app.py -v`
Expected: FAIL with `FileNotFoundError` or `ModuleNotFoundError` — `dashboard/app.py` doesn't exist yet.

- [ ] **Step 4: Implement `dashboard/app.py`**

```python
"""Streamlit dashboard for the Phase 3 experiment matrix.

Run with: streamlit run dashboard/app.py

Reads live progress directly from the project's own database
(dashboard/queries.py) and control-plane state from a small status file
(dashboard/status_store.py); starts/stops the actual simulation as a
separate OS process (dashboard/process_control.py) that survives this
Streamlit script's own rerun-on-every-interaction execution model.
"""

import streamlit as st

from dashboard import status_store
from dashboard.process_control import RunConfig, is_alive, resume, start, stop
from dashboard.queries import get_progress_for_run
from database.session import new_session
from src.simulation.matrix_runner import _build_cell_specs
from src.utils.helpers import generate_id

st.set_page_config(page_title="Phase 3 Matrix Dashboard", layout="wide")
st.title("Phase 3 Experiment Matrix Dashboard")

ALL_CELL_KEYS = [spec.key for spec in _build_cell_specs()]

if "matrix_run_id" not in st.session_state:
    st.session_state.matrix_run_id = status_store.get_active_run_id() or generate_id("matrix")

st.header("Run configuration")
matrix_run_id = st.text_input("matrix_run_id", value=st.session_state.matrix_run_id)
st.session_state.matrix_run_id = matrix_run_id

col1, col2 = st.columns(2)
with col1:
    cell_scope = st.radio("Cell scope", ["All 13 cells", "Specific cells"])
    selected_cells = None
    if cell_scope == "Specific cells":
        selected_cells = st.multiselect("Cells", ALL_CELL_KEYS)
    seeds_text = st.text_input("Seeds (comma-separated)", value="0")
    num_days = st.number_input("Days", min_value=1, value=1, step=1)
with col2:
    distributed = st.checkbox("Distributed (multiple processes)", value=False)
    num_processes = st.number_input("num_processes", min_value=1, value=4, step=1, disabled=not distributed)
    dry_run = st.checkbox("Dry run (no real spend)", value=True)

def _build_config() -> RunConfig:
    return RunConfig(
        matrix_run_id=matrix_run_id,
        cell_keys=selected_cells if cell_scope == "Specific cells" else None,
        seeds=[int(s.strip()) for s in seeds_text.split(",") if s.strip()],
        num_days=int(num_days),
        dry_run=dry_run,
        distributed=distributed,
        num_processes=int(num_processes),
    )

st.header("Controls")
control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
    if dry_run:
        if st.button("Start"):
            start(_build_config())
            status_store.set_active_run_id(matrix_run_id)
            st.rerun()
    else:
        st.warning("Real launch selected -- type the matrix_run_id below to confirm.")
        confirmation = st.text_input("Type the matrix_run_id above to confirm a REAL (paid) launch")
        if st.button("Confirm real launch"):
            if confirmation == matrix_run_id:
                start(_build_config())
                status_store.set_active_run_id(matrix_run_id)
                st.rerun()
            else:
                st.error("Typed text does not match matrix_run_id -- launch not started.")

with control_col2:
    if st.button("Stop"):
        stop(matrix_run_id)
        st.rerun()

with control_col3:
    if st.button("Pause"):
        st.info("Pause behaves the same as Stop -- there is no mid-execution suspend. Safe to Resume afterward.")
        stop(matrix_run_id)
        st.rerun()

with control_col4:
    status = status_store.read_status(matrix_run_id)
    resume_disabled = status is None or status.get("state") not in ("stopped", "failed")
    if st.button("Resume", disabled=resume_disabled):
        resume(matrix_run_id)
        st.rerun()

st.header("Status")
try:
    status = status_store.read_status(matrix_run_id)
except status_store.StatusFileCorruptedError:
    st.error("Status file for this matrix_run_id is unreadable. Live progress below still works from the database.")
    status = None

if status is None:
    st.info("No run has been started under this matrix_run_id yet.")
else:
    pid = status.get("pid")
    reported_state = status.get("state", "unknown")
    if reported_state == "running" and pid is not None and not is_alive(pid):
        st.error(f"Process (pid {pid}) is no longer running but never reported completion -- likely crashed.")
    else:
        st.write(f"State: **{reported_state}**")

    usage = status.get("cumulative_usage")
    if usage:
        u1, u2, u3 = st.columns(3)
        u1.metric("Prompt tokens", usage.get("prompt_tokens", 0))
        u2.metric("Completion tokens", usage.get("completion_tokens", 0))
        u3.metric("Total tokens", usage.get("total_tokens", 0))

    failures = status.get("failures")
    if failures:
        st.subheader("Failures")
        st.table(failures)
    elif status.get("state") == "running":
        st.caption("Failures are only known once each cell/seed's run completes -- not live mid-run.")

st.header("Live progress")
session = new_session()
progress_rows = get_progress_for_run(session, matrix_run_id)
if not progress_rows:
    st.info("No progress recorded yet for this matrix_run_id.")
else:
    st.dataframe(
        [
            {
                "Cell": row.cell_key,
                "Seed": row.seed,
                "Day": row.current_day,
                "LLM decisions": row.total_llm_decisions,
            }
            for row in progress_rows
        ],
        use_container_width=True,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_dashboard_app.py -v`
Expected: PASS (3/3).

- [ ] **Step 6: Manual verification**

Run: `streamlit run dashboard/app.py`
Expected: the app opens in a browser tab. Confirm: the config form renders; toggling "Dry run" off replaces the Start button with the confirmation flow; clicking Start with dry_run on launches a real subprocess (check with `tasklist`/`ps` for a `dashboard.runner` process); the Live progress table updates on rerun as the dry run proceeds; Stop terminates the process; Resume re-launches and picks up from the checkpoint.

- [ ] **Step 7: Run the full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add dashboard/app.py tests/test_dashboard_app.py .gitignore
git commit -m "feat: add dashboard Streamlit UI"
```

---

## Self-Review Notes

- **Spec coverage**: §1 architecture -> Tasks 1-4 (status_store, queries, runner, process_control); §2 data sources -> Task 2 (queries.py) + Task 3 (runner.py's usage_callback); §3 controls -> Task 4 (start/stop/resume) + Task 5 (Pause relabeled to Stop, real-launch confirmation); §4 config form -> Task 5; §5 error handling -> Task 1 (`StatusFileCorruptedError`) + Task 5 (crash detection via `is_alive` check); §6 file layout -> matches exactly; §7 out of scope -> confirmed nothing in this plan builds a true pause, a cost estimate, or touches the simulation core.
- **Placeholder scan**: none found — every step shows complete, real code.
- **Corrections made during self-review**: (1) Task 2's original draft queried `TransactionRecord` for a transaction count, but that table has no run-scoping column at all (confirmed against the real `database/models.py`) — removed from `CellSeedProgress` and Task 5's display, and the design spec itself updated for consistency (user confirmed dropping it rather than adding a new schema column). (2) Task 3's distributed-mode `usage_callback` originally discarded the `usage` argument `run_matrix_distributed` passes and re-queried the parent process's own (irrelevant) counter instead — fixed to accumulate the per-group `LLMUsage` values it's actually given. (3) Task 4's `start()` was drafted with a "go back and add these fields" instruction rather than the fields already being in the initial code block — fixed to show the complete, correct function directly.
- **Type consistency**: `RunConfig` (Task 4) fields (`matrix_run_id, cell_keys, seeds, num_days, dry_run, distributed, num_processes`) match exactly what `dashboard/app.py` (Task 5) constructs in `_build_config()`, and match what `runner.py` (Task 3)'s CLI args accept (`--matrix-run-id`, `--cell-keys`, `--seeds`, `--num-days`, `--dry-run`/`--real`, `--distributed`, `--num-processes`) via `process_control._build_command`'s translation — verified consistent across all three tasks.
