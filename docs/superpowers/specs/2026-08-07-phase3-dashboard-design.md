# Phase 3 Dashboard — Design Spec

## 0. Source documents and why this spec exists

The project's own master-instructions document (`dashboard.md`, repo root) states dashboard work is reserved for other group members and that Claude should not write it. The user explicitly overrode this (2026-08-07): "that instruction no longer applies." No `dashboard/` directory exists yet in the repository — this is a from-scratch build.

Per [[feedback-no-assumptions]], every design decision below was resolved by asking the user directly during brainstorming, not guessed. **User decisions (2026-08-07):**

1. Controls are Start / Stop / Resume; there is no true mid-execution pause. A "Pause" button in the UI is present (per the user's original request) but behaves identically to Stop — both map to the same stop mechanism (§3). This avoids inventing a new suspend/continue mechanism that doesn't otherwise exist.
2. Stack: Streamlit (pure Python, fastest to build/iterate, no new frontend toolchain).
3. Stop mechanism: process termination (no new cooperative-cancellation code inside `run_matrix`'s day loop). Relies entirely on the existing checkpoint/resume mechanism (Plan 4) for safety. Accepted cost: up to one simulated day's already-spent real LLM calls may need to be redone on Resume, since checkpoints only reflect fully-committed days.
4. A real (non-dry-run, real spend) launch requires an explicit confirmation step in the UI, separate from the Start action itself — mirrors the existing go/no-go gate used for CLI-launched real runs (Plan 4/6), just adapted to a UI control.
5. Live progress (day counts, transaction/decision counts) is read directly from the SQLite database, not from a custom callback-fed status file — the database is already the shared source of truth for both single-process (`run_matrix`) and cross-process (`run_matrix_distributed`) runs, and this requires **zero new code in the simulation core** (`run_matrix`, `distributed_matrix_runner.py`). A small status file covers only what the database doesn't hold: which PID owns the currently-tracked run, the dry-run flag, and best-effort token usage.

## 1. Architecture

Three cleanly-separated pieces, none of which touch the tested simulation core:

- **`dashboard/app.py`** (Streamlit UI). Renders controls and live data. Never runs the simulation itself — reads the database and a small status file, and starts/stops a separate OS process.
- **`dashboard/runner.py`** (subprocess entrypoint). A new, thin wrapper script invoked as `python dashboard/runner.py --matrix-run-id ... --cell-keys ... --seeds ... --num-days ... [--dry-run | --real] [--distributed --num-processes N]`. Calls the existing `run_matrix`/`run_matrix_distributed` with those parameters, using the already-existing `usage_callback` hook to write best-effort token usage to the status file. Writes a final status (`completed`/`failed`, plus the `failures` list) to the status file when `run_matrix`/`run_matrix_distributed` returns.
- **`dashboard/process_control.py`**. `start(config) -> None`: builds the command line, launches `runner.py` via `subprocess.Popen` (detached — must survive the launching Streamlit rerun), writes `{pid, matrix_run_id, dry_run, started_at, state: "running"}` to the status file. `stop() -> None`: reads the PID from the status file, terminates that process, updates `state` to `"stopped"`. `is_alive() -> bool`: OS-level liveness check on the recorded PID (a process object surviving does not by itself mean it's healthy — see §5 error handling). `resume(matrix_run_id, checkpoint_dir) -> None`: same as `start`, but passes the existing `matrix_run_id`/`checkpoint_dir` through so `run_matrix`'s own checkpoint-resume logic (Plan 4) picks up where it left off; no new resume logic is added here.

The dashboard's own process (Streamlit) and the simulation's process (`runner.py`) are fully decoupled — restarting or closing the dashboard never touches a running simulation, and a running simulation's crash never touches the dashboard.

## 2. Data sources

- **Live progress (day counts, transaction counts, decision counts)**: direct read-only SQL queries against the project's SQLite database (`database/session.py`'s `DATABASE_URL`), scoped by the tracked `matrix_run_id`. Specifically: `MAX(timestep)` per `run_id` from `timestep_logs` (current day), row counts from `transactions`/`llm_decisions` filtered by `run_id` prefix. WAL mode (already enabled, Plan 6a) makes concurrent reads safe alongside the simulation's own writes, in both single-process and distributed modes.
- **Token usage**: best-effort from the status file. Single-process mode: smooth per-day updates (the runner's `usage_callback` fires once per simulated day). Distributed mode: coarser, one update per completed worker *group*, not per day — this is the same limitation Plan 6a's own design spec already documents and accepts (each worker process has its own separate token counter; `run_matrix_distributed`'s existing `usage_callback` only fires when a whole group finishes). Not fixed further here.
- **Process liveness**: OS-level check on the PID recorded in the status file at launch time.
- **Failures**: only known with certainty once `runner.py` writes the final `(results, failures)` outcome to the status file when the underlying `run_matrix`/`run_matrix_distributed` call returns (i.e., not "live" mid-run) — a currently-running cell/seed's eventual failure isn't visible before that. This is called out explicitly in the UI (a "failures known at completion" note near the failures panel) rather than silently implying real-time failure detection that doesn't exist.
- **No dollar-cost estimate.** There is no per-model pricing table anywhere in this codebase; the dashboard shows token counts only, not a fabricated cost figure.

## 3. Controls

- **Start**: a config form (§4) followed by a Start button. If the dry-run toggle is on (default), Start launches immediately via `process_control.start()`. If dry-run is off (a real, paid launch), the Start button is replaced by a text input requiring the user to type the exact `matrix_run_id` shown on screen, plus a "Confirm real launch" button that only enables once the typed text matches — mirrors the existing AskUserQuestion-based go/no-go gate used for CLI-launched real runs, adapted to a UI control that can't be triggered by a stray click.
- **Stop** and **Pause**: both call `process_control.stop()` (process termination). The UI presents them as two labeled buttons per the user's original request, but they are the same action — no separate pause state exists. A tooltip on "Pause" notes it behaves like Stop (safe to Resume afterward), so this isn't a silent surprise.
- **Resume**: only enabled when the status file shows `state in ("stopped", "failed")` and a `matrix_run_id`/`checkpoint_dir` are on record. Calls `process_control.resume(...)`.

## 4. Start configuration form

- **Mode**: single-process (`run_matrix`) or distributed (`run_matrix_distributed`, with an additional `num_processes` field shown only in this mode).
- **Cell scope**: "All 13 cells" or a multi-select of specific cell keys (from `_build_cell_specs()`).
- **Seeds**: comma-separated integers, default `0`.
- **Days**: integer, default matches whatever the user is testing (no hardcoded production default baked into the UI — the 3-seed/365-day production scale from Plan 6 is something the user types in deliberately, not a pre-filled trap).
- **Dry run**: toggle, defaults to **on**.
- **`matrix_run_id`**: auto-generated (`generate_id("matrix")`-style), editable — needed for Resume to target a specific prior run, and shown prominently since Resume/Stop both key off it.

## 5. Error handling

- If `process_control.start()`'s `subprocess.Popen` call itself fails (e.g. bad interpreter path), the status file is written with `state: "failed"` and the raised exception's message, surfaced directly in the UI rather than a silent no-op.
- If the recorded PID is no longer alive but the status file still says `"running"` (e.g. the process crashed without writing a final status — an OS-level kill signal or an out-of-memory kill wouldn't get a chance to write anything), the dashboard detects this via the liveness check and displays the run as `"crashed"` (distinct from a clean `"stopped"`/`"completed"`), with the database's own last-known day count still shown from the live query in §2 (that data source doesn't depend on the process being alive).
- The status file itself is a single JSON file per tracked run; if it's corrupted/unreadable, the dashboard shows a clear "status file unreadable, but the database can still be queried directly" message rather than crashing the whole page — it never blocks the read-only database queries in §2.

## 6. File layout

```
dashboard/
  app.py              # Streamlit UI: renders controls + live panels
  runner.py           # subprocess entrypoint: parses CLI args, calls run_matrix/run_matrix_distributed
  process_control.py  # start/stop/resume: Popen, PID liveness check, status file I/O
  status_store.py      # status JSON schema + read/write helpers (shared by runner.py and app.py)
  queries.py           # read-only DB queries backing the live progress panels (§2)
  state/                # gitignored: one status JSON per matrix_run_id currently tracked by the dashboard
```

`streamlit>=1.36` is added as a new optional `dashboard` dependency group in `pyproject.toml` (for `st.fragment(run_every=...)` auto-refresh — no separate third-party autorefresh package needed).

## 7. Out of scope

- True mid-execution pause/suspend (§0.1 — explicitly declined).
- Dollar-cost estimation (§2 — no pricing data exists to base it on).
- Real-time failure detection for an in-progress cell/seed (§2 — only known at process completion).
- Any change to `run_matrix`, `run_timestep`, or `distributed_matrix_runner.py`'s existing behavior — this dashboard is purely additive tooling on top of already-existing hooks (`progress_callback`, `usage_callback`, checkpoint/resume).
