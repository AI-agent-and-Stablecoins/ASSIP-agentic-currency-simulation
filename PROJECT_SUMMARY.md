# ASSIP Future of Finance Lab — Phase 3: Comprehensive Project Summary

**Team:** Aditya Shah, Daniel Gulley, Jiayi Zhou (GMU, ASSIP program)
**Document purpose:** a single, exhaustive snapshot of the entire project — origin, research goals, architecture, what has been built, what decisions were made and why, what bugs were found and fixed, and what remains open. Written 2026-08-07 to consolidate a very long build history into one place.

---

## 1. What this project is

This is a multi-agent, LLM-driven economic simulation that studies a single core research question:

> **What economic, structural, and governance characteristics of digital money (stablecoins, tokenized deposits, gold-backed tokens) do autonomous AI agents develop preferences for, over time, under negotiation, macroeconomic shocks, and cross-border friction?**

Populations of AI agents (buyers, sellers, banks, investors, regulators), each *piloted by a real LLM via OpenRouter* (not a rule-based bot), transact with each other inside a simulated economy that has real stablecoins (USDC, USDT, EURC, EURT, DAI, FDUSD), a gold-backed token (PAXG/XAUT), a bank-issued tokenized deposit, multiple blockchains/settlement rails (Ethereum, Arbitrum, Base, Solana) with distinct gas-fee dynamics, and a macroeconomic layer (inflation, confidence, FX rates, banking-crisis/depeg shocks). Every agent decision, negotiation turn, settlement, and macro event is persisted to a SQLite database with full provenance (git commit hash, config hash, model IDs, random seed) for reproducibility. A statistical layer then runs clustered-logistic-regression hypothesis tests (H1–H11) against the resulting panel data to answer questions like "does higher risk aversion push agents toward USD stablecoins over EUR stablecoins?" or "does proximity to a banking crisis push agents toward gold?"

The **deterministic/LLM split is the core design invariant**: LLMs act purely as the reasoning/negotiation engine (they choose what to offer, what currency/chain to use, whether to accept a counter-offer). The Python backend has 100% deterministic authority over wallet balances, settlement, fee/tax subtraction, and ledger writes — an LLM can never directly mutate financial state.

## 2. Origin documents and how the project's scope was set

Two chat-log documents at the repo root capcaptured the original requirements handed down before any of this session's implementation work began, and they still govern scope boundaries:

- **`dashboard.md`** — contains (a) the original "Master System Instructions" prompt that scoped Claude's job as backend-only ("implement core logic, configs, database layer, e2b integrations, metrics, utility models, simulation engine"), with an explicit **CRITICAL EXCLUSION: do not write code for `dashboard/` or `experiments/` — group members are handling those**; and (b) a later, much more detailed "ASSIP Future of Finance Lab — Dashboard Specification" chat log describing what the *human dashboard team* originally intended to build (Home/Economy/Currency/Blockchain/Agent Explorer/Agent Network/Wealth/Utility/Transaction Explorer/Negotiation Viewer/Hallucination Dashboard/Preference Evolution/Experiment Comparison pages, React/Next.js/Plotly/D3/Tailwind stack, KPI cards, Sankey diagrams, network graphs, etc.). **This second document was never built by anyone** — it describes an intended human-built frontend that didn't materialize, distinct from the small Streamlit dashboard now being built by Claude (see §8).
- **`Experiment.md`** — contains the "ASSIP Future of Finance Lab — Phase 3 Master Engineering Specification," an extremely detailed, aspirational architecture document (executive summary, five utility-function families including Epstein-Zin recursive utility, social network-effect cascades, dual "Factual vs. Self-Research" agent modes, a 7-row factor-isolation sandbox table, dynamic shock engine, provenance requirements, a full SQL schema, hypotheses H1–H6, and its own Streamlit dashboard section). This is the authoritative reference for Phase 3, but **it is aspirational, not as-built** — see §9 for exactly which of its features are real vs. spec-only.

Per standing user instruction (**[[feedback-no-assumptions]]** in project memory), whenever an ambiguity, error, or unstated design choice was found in these source documents during implementation, it was escalated to the user for an explicit decision rather than resolved silently. That pattern recurs throughout this document.

## 3. Repository layout (as it exists today)

```
configs/            YAML config: agent_profiles, blockchains, currencies, economy, llm, scenarios, simulation
database/           SQLAlchemy models (models.py), session/engine setup (session.py), repository layer (repository.py)
docs/superpowers/   spec/ (design docs) and plans/ (implementation plans) for every Phase — see §6
src/
  agents/           base_agent, buyer/seller/bank/investor/regulator_agent, population, wallet, memory, preferences, wealth
  blockchain/       chain definitions, gas fees, routing/settlement (implicit under simulation/transactions)
  currencies/       stablecoin/gold-token/tokenized-deposit definitions
  economy/          shocks, inflation, confidence, macro_state, monetary_policy, fx_dynamics, fx_tax, risk_adaptation,
                     trust (ledger), event_log, history_builder, sandbox_scenarios
  econometrics/     cell_identity, regression_engine, hypothesis_datasets, hypothesis_regressions, report
  governance/       compliance / issuer risk / reserve models (spec'd; see §9 for build status)
  llm/              llm_router (OpenRouter client + usage tracking), agent_reasoning, decision_schema,
                     decision_adapter, decision_to_transaction, hallucination_detector, market_intelligence
                     (Polygon live price feed), prompts/
  market/           pricing/liquidity/orderbook logic
  negotiation/       multi-turn LLM-vs-LLM negotiation engine
  simulation/       environment, event_queue, scheduler, simulation_runner, timestep, matrix_runner,
                     distributed_matrix_runner, provenance
  transactions/     ledger/settlement/validation
  utility/          cara, crra, risk_neutral, epstein_zin, multi_attribute, risk_profiles, utility_factory
  utils/            constants, helpers, logger
tests/              ~70 test files, one per module/feature area
scripts/            launch_demo_run.py, launch_master_run.py, calibrate_currency_configs.py
metrics/            adoption_curves, currency_usage, hallucinations, wealth_distribution, transaction_stats,
                     chain_selection, compliance_effects, gas_fee_sensitivity, governance_preference,
                     liquidity_sensitivity, wandb_logger
sandbox/            e2b sandbox launcher/dispatcher/result_collector/cleanup (E2B integration scaffolding)
experiments/        experiment_001..011 — reserved for teammates per the original scope exclusion (§2)
notebooks/          analysis.ipynb, currency_adoption.ipynb, final_figures.ipynb, hallucination_analysis.ipynb
checkpoints/        pickled per-cell/seed matrix-run checkpoints (for resume)
dashboard/           NOT YET CREATED — see §8, implementation not started
```

Dependency stack (`pyproject.toml`): Python ≥3.12, `pydantic>=2.6`, `sqlalchemy>=2.0`, `pyyaml`, `pandas`, `e2b`; optional groups `llm` (`httpx`, for OpenRouter + Polygon), `econometrics` (`statsmodels`), `observability` (`wandb`), `market-data` (`requests`). Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`.

## 4. Core simulation mechanics (as actually built)

- **Timestep lifecycle** (`src/simulation/timestep.py`, `environment.py`, `simulation_runner.py`): each simulated day, buyers are matched with sellers, each buyer's full decision (currency, chain, negotiation) is driven by a real LLM call through `src/llm/llm_router.py`, negotiation proceeds via `src/negotiation/`'s multi-turn engine (`run_llm_negotiation`) until acceptance/rejection/max-rounds, and the outcome is validated and settled deterministically (fees, FX/conversion tax, ledger write).
- **Utility models** (`src/utility/`): CARA (constant absolute risk aversion, `U(c) = -e^{-αc}`) is the primary model actually driving decisions, with per-agent α assigned at population-generation time (`src/agents/population.py`). CRRA, risk-neutral, Epstein-Zin, and multi-attribute utility modules also exist in the codebase (`crra.py`, `risk_neutral.py`, `epstein_zin.py`, `multi_attribute.py`, wired through `utility_factory.py`) but CARA is what the Phase 3 hypotheses (H1–H11) are actually built around.
- **Loss-driven risk adaptation** (`src/economy/risk_adaptation.py`): an agent's CARA coefficient increases after realized losses, consistent with the master spec's adaptive-learning section.
- **Cross-border FX tax** (`src/economy/fx_tax.py`): a small conversion tax applied when a transaction's settlement currency doesn't match a party's "home" currency zone (USD vs EUR). This is the *only* cross-border friction mechanism that survived review — an added second "counterparty-zone" tax was explicitly removed by user decision during Plan 4 (real stablecoin rails are valued specifically for avoiding this kind of friction via a shared settlement currency).
- **FX dynamics** (`src/economy/fx_dynamics.py`): daily EUR/USD noise applied to the real accounting exchange rate, added in Plan 5 specifically so H5 (cross-border volatility → USD preference) has genuine variance to regress against. User explicitly chose "noise affects the real accounting rate" over alternative designs.
- **Shock engine** (`src/economy/shocks.py`, `sandbox_scenarios.py`): inflation shocks, banking-crisis/confidence shocks, depeg events, gas-fee spikes; step-indexed and logged (`event_log.py`).
- **Live market data**: `src/llm/market_intelligence.py` wires real Polygon.io price feeds into the LLM decision context (`CurrencyHistory`/`MacroHistory` builders in `history_builder.py`).
- **Hallucination detection**: `src/llm/hallucination_detector.py` compares an LLM-negotiated value against the deterministic fair-value calculation.
- **Provenance**: `src/simulation/provenance.py` records git commit hash, config hash, and model roster per run, per the master spec's reproducibility requirements.

## 5. Research design: hypotheses and sandboxes

**Master simulation ("one big simulation"):** all agent types, all currencies, all chains, all shocks active simultaneously — the full-scale 365-day run.

**Factor-isolation sandboxes** (`src/economy/sandbox_scenarios.py`): controlled pairwise comparisons holding all else constant, each run in both a **domestic** and a **cross-border** (US agent vs. EU agent, FX-tax-bearing) variant — 6 sandboxes × 2 settings = 13 total matrix cells (12 sandbox cells + the master cell):

1. Liquidity vs. Governance
2. Governance vs. Stability
3. Liquidity vs. Stability
4. Asset Backing vs. Liquidity
5. Asset Backing vs. Stability
6. Asset Backing vs. Governance

(A 7th sandbox from the master spec, "Privacy vs. Friction," is explicitly deferred — reserved as **H6** and never built.)

**Hypotheses actually implemented and reported** (`src/econometrics/`, `run_all_hypotheses` → 15 result rows):

| # | Hypothesis | Regressor → outcome |
|---|---|---|
| H1 | Risk aversion vs. currency choice | Higher CARA α → USD-stablecoin preference over EUR |
| H2 | Risk aversion vs. liquidity/fees | Higher CARA α → spread-vs-gas-fee tradeoff preference |
| H3 | Risk aversion vs. governance | Higher CARA α → governance-vs-liquidity preference (pooled domestic+cross_border with a `cell_key` fixed effect) |
| H4 | Crisis proximity vs. gold backing | Closer to a crisis/depeg event → gold-backed token preference |
| H5 | Cross-border volatility | Higher EUR/USD volatility → USD-stablecoin preference in cross-border settlement |
| **H7** | Governance vs. Stability sandbox | which side wins (stability hypothesized) |
| **H8** | Liquidity vs. Stability sandbox | which side wins (stability hypothesized) |
| **H9** | Asset Backing vs. Liquidity sandbox | which side wins (gold hypothesized) |
| **H10** | Asset Backing vs. Stability sandbox | which side wins (FDIC-insured deposit hypothesized; lower confidence) |
| **H11** | Asset Backing vs. Governance sandbox | which side wins (governance hypothesized; lower confidence) |

H7–H11 are each reported **twice** (domestic and cross_border separately) → 10 rows, plus H1/H2/H4/H5 (1 each) and H3 (1, pooled) = 15 total. **H6 is reserved and intentionally skipped** — see §7's Plan 6 entry for why this numbering matters (a real naming collision was caught and fixed here).

Every hypothesis reports: coefficient (β), standard error, 95% CI, p-value, and McFadden pseudo-R² (clustered logistic regression via `statsmodels`, `fit_clustered_logit` in `regression_engine.py`).

## 6. How the project is built: process and document trail

Every phase/plan follows the same cycle (the `superpowers` skill framework): **brainstorm → design spec (`docs/superpowers/specs/`) → implementation plan (`docs/superpowers/plans/`) → subagent-driven execution in an isolated git worktree → task-level review → whole-branch review → merge to `main`.**

Specs (chronological):
1. `2026-07-22-phase2-llm-negotiation-layer-design.md`
2. `2026-07-29-phase3-full-scale-simulation-design.md` (the master spec's own house design doc)
3. `2026-07-29-phase3-plan2-shock-engine-design.md`
4. `2026-07-30-phase3-plan3-agent-population-design.md`
5. `2026-07-31-phase3-plan4-matrix-runner-design.md`
6. `2026-08-02-phase3-plan5-econometrics-design.md`
7. `2026-08-04-phase3-plan6-concurrency-and-sandbox-hypotheses-design.md`
8. `2026-08-07-phase3-dashboard-design.md`

Plans mirror the same list 1:1, plus Plan 6 is split into `6a-concurrency` and `6b-sandbox-hypotheses`, and Plan 1 is `phase3-01-foundation-persistence.md`.

## 7. Phase/Plan-by-plan build history

### Phase 2 — LLM negotiation layer
Done, merged. Built the multi-turn LLM-vs-LLM negotiation engine and OpenRouter integration that everything later depends on.

### Phase 3 Plan 1 — Foundation persistence layer
Done, merged. Base SQLAlchemy schema and repository layer.

### Phase 3 Plan 2 — Shock engine
Done, merged 2026-07-30. Shock/event engine, `TrustLedger`, historical prompt context wiring.

### Phase 3 Plan 3 — Agent population
Done, merged 2026-07-30. 100-agent population generator: roles, currency zones, per-agent CARA coefficient, per-agent OpenRouter model assignment (so different agents can literally be piloted by different LLMs).

### Phase 3 Plan 4 — Matrix runner / experiment orchestration
**Done, merged 2026-08-02** (merge commit `1517770`). This is the plan that turned the population into a real running simulation: wired real per-agent LLM decisions and full LLM-vs-LLM negotiation into the day loop (replacing an old rule-based `choose_currency_and_chain` call — an explicit user decision, since the master spec was ambiguous on this), added full persistence, the cross-border FX tax, loss-driven CARA adaptation, live Polygon price wiring, the 6 factor-isolation sandboxes (domestic + cross-border) plus the 365-day master scenario, and the 13-cell × N-seed matrix runner itself, including a hardened `dry_run` safety gate and cell/seed-level checkpoint/resume. 365 tests passing at merge.

Whole-branch review found 2 Critical + 2 Important issues, all fixed:
1. Transaction-ID collision at matrix-run scale (`generate_id` was truncated to 32 bits of entropy) — pure bug fix.
2. `dry_run=True` didn't actually guarantee zero real spend if a caller supplied a real client anyway — user chose to harden it so `run_matrix` refuses *any* externally-supplied client under `dry_run=True`.
3. An added second, additive cross-border FX tax beyond the originally-approved single tax was removed by user decision (real stablecoin rails are valued specifically for avoiding this kind of friction).
4. The 6 sandboxes' crisis-proximity shock timing had drifted to different values per sandbox (reintroducing a confound the shared-shock-schedule design was meant to prevent) — user delegated the fix; standardized to one identical 10-day gap across all 6.

### Phase 3 Plan 5 — Econometrics engine
**Done, merged 2026-08-04.** Added `src/econometrics/` in full (H1–H5 datasets/regressions/report) and `src/economy/fx_dynamics.py`. Two whole-branch reviews; fixed a `SQLITE_MAX_VARIABLE_NUMBER` crash in H5's negotiation query at real matrix-run scale (fixed by scoping to `simulation_id` and grouping in Python) and a gap where `matrix_run_id` scoping never reached the production entry point `run_all_hypotheses`. 406 tests passing at merge. One task (SQL-side column projection for query performance) explicitly deferred as non-blocking.

### The first real production run (2026-08-04)
`scripts/launch_master_run.py`, `matrix_run_id="phase3-real-run-2026-08-04"`, master cell only, seed 0, target 14 days, real spend against the full 99-model OpenRouter roster, launched under a hard deadline with explicit user go-ahead (the full-scale-run launch gate is always separate from a plan's merge approval). It was killed once unexpectedly (cause unknown, not user-initiated) after days 0–1 and resumed cleanly via checkpoint/resume with no data loss. Observed pace ≈3.5–4 hours per simulated day. **This run was later explicitly abandoned** ("forget the master run") during the Plan 6 conversation — confirmed no process was still running at that point, so no spend was silently ongoing. Its data still exists (`checkpoints/phase3-real-run-2026-08-04/`, `assip.db`) but is not part of active work.

### Phase 3 Plan 6 — Concurrency + 5 new sandbox hypotheses
**Done, merged to main 2026-08-07** (merge commit `f4e1a96`; sub-branches `phase3-plan6a-concurrency` and `phase3-plan6b-hypotheses`, both cleaned up post-merge). This is the largest single body of work in the project's history. Triggered by the user's request to speed the run up "on the GPU" and to "isolate a factor" pairwise across the 6 sandboxes.

**GPU was investigated and dropped**: LLM inference is 100% remote via OpenRouter (no local model weights anywhere in the codebase), so there is no local-compute target GPU acceleration could help — confirmed with the user, who agreed to drop it entirely in favor of concurrency instead.

**What "isolate a factor" turned out to mean**: the 6 sandboxes, their domestic/cross-border split, and the cross-border FX tax already existed exactly as described (built in Plans 3/4) — the actual gap was that only 1 of the 6 pairs (`liquidity_vs_governance`, already H3) had a dedicated "which coin won" regression. The other 5 didn't. This became the H7–H11 work below.

**New hypotheses — a real naming collision, caught and fixed**: the design spec first drafted these as H6–H10, but the master spec already reserves H6 for the still-deferred Privacy-vs-Friction sandbox. This was never checked during design and the collision was only caught by the final whole-branch review. Escalated to the user via AskUserQuestion; **renumbered to H7–H11**, executed personally (not via subagent) across `hypothesis_datasets.py`, `hypothesis_regressions.py`, `report.py`, and the test file. Directional hypotheses (all approved as proposed; H10/H11 explicitly flagged lower-confidence): H7 stability wins over governance; H8 stability wins over liquidity; H9 gold wins over liquidity; H10 FDIC-insured deposit wins over gold; H11 governance wins over asset-backing.

**Concurrency — what was actually built vs. what was designed**: the original design specified a two-phase architecture (all buyers decide in parallel, then settle serially). What was actually implemented settles **inline inside each buyer's own worker thread**, under one shared lock (`_process_buyer_llm_day` in `src/simulation/timestep.py`, dispatched via `ThreadPoolExecutor`, new `max_workers` param on `run_timestep`) — the two-phase design would have broken an existing intra-buyer cross-good dependency. **This is a real, user-approved trade-off**: with `llm_max_workers > 1`, the same random seed is no longer guaranteed to produce an identical run (thread-scheduling can change what wallet balance a concurrently-running buyer sees in its prompt). Accepted explicitly in favor of speed; documented directly in the design spec as an amendment. Cross-process parallelism was also added: `run_matrix_distributed`/`_run_cell_group` (`src/simulation/distributed_matrix_runner.py`) partitions the **full cell×seed cross product** (not just cell keys — an early version only partitioned cells, capping parallelism) across a `ProcessPoolExecutor`, all workers sharing one SQLite DB via WAL mode + busy-timeout (`database/session.py`).

**Scale locked in without a pilot**: the user explicitly declined a timing/cost pilot run and instead locked in **3 seeds × 365 days × 13 cells (39 full-year cell-runs)** directly, substituting real-time cost/token logging (`LLMUsage`, `get_cumulative_usage`, `usage_callback` on both `run_matrix` and `run_matrix_distributed`) as the safety net instead. **This 39-cell production run has not yet been launched** — it remains its own explicit go/no-go gate, same rule as every other real-spend run in this project.

**A major pre-existing (not Plan-6-caused) data-integrity bug found and fixed**: all 13 matrix cells share the same random seed, so agent-population generation produced **identical agent IDs across every cell**, and `AgentRecord`/`WalletRecord` were keyed by bare `agent_id` with no run/cell scoping at all — meaning even a fully *sequential*, *non-concurrent* 13-cell run was silently overwriting one cell's wallet data with the next cell's the entire time, since Plan 4. This was escalated to and approved by the user: `AgentRecord` is now composite-keyed `(run_id, id)`; `WalletRecord` is now composite-keyed `(run_id, agent_id, currency_symbol)` — this took **two review rounds**, because the first fix attempt gave `WalletRecord` only a plain extra `run_id` column rather than a true composite key, and the user explicitly required the real fix once this was caught. `AgentRepository._sync_wallet` was rewritten from a delete-then-reinsert pattern to a proper merge (UPDATE/INSERT/DELETE) to avoid a SQLAlchemy identity-map hazard introduced by the new composite key. `database.session.assert_schema_current` was added as a fail-fast guard so a stale pre-existing `.db` file with the old schema (e.g. `assip.db`, `assip_phase3-real-run-2026-08-04_days0-1.db` — both still on the OLD schema, need regeneration not migration before any new real run) errors immediately instead of burning real LLM spend before failing deep inside the day loop.

**Other bugs found and fixed during Plan 6's reviews**:
- A `SQLITE_LOCKED` race during WAL-mode conversion under concurrent-process writes (`busy_timeout` alone doesn't cover `SQLITE_LOCKED`, only `SQLITE_BUSY`) — found via direct empirical repeated-run testing (~1-in-3 failure rate), fixed with a manual retry loop (`_set_journal_mode_wal_with_retry`).
- A real, measured ~2.5% data-loss race in `LLMUsage`'s cumulative-counter read-modify-write, despite the module being specifically built for multi-threaded use — fixed with a `threading.Lock` plus a genuine `ThreadPoolExecutor`-based regression test that reproduces the race.
- `run_matrix_distributed`'s bare `future.result()` call (no try/except) meant one dead worker discarded every other, already-completed group's results — fixed with per-pair and per-group exception handling, with full traceback preservation (previously only `str(exc)` was kept in sanitized failures).
- H5's agent `currency_zone` lookup wasn't scoped by `run_id`, so it collapsed rows across cells sharing a seed (same root-cause family as the agent-identity bug above, different layer) — fixed.

Two full whole-branch reviews (opus-level) plus a round-2 fix-and-re-review on the schema work. All Critical/Important findings fixed and re-verified: Plan 6a 432 passed/1 skipped, Plan 6b 413 passed/1 skipped; a targeted 91-test run covering every touched file passed clean on merged `main` (full-suite runs kept getting externally interrupted by session/resource pressure late in the very long build session — determined via timing analysis to be environmental, not a code defect).

**Deferred, explicitly non-blocking follow-ups left on the list** (not currently being worked, listed for whoever picks Phase 3 up next):
- H1–H5's econometrics still cluster by bare `agent_id`, which still merges observations across cells sharing a seed (same root-cause family as the fixed schema bug, but at the econometrics-query layer, not yet fixed there).
- `run_all_hypotheses` is all-or-nothing across its 15 fits — one bad cell/landslide result aborts the entire report.
- `build_sandbox_preference_dataset` hydrates full ORM rows per hypothesis (now 15× since H7–H11) rather than pushing filters into SQL — a performance concern at real matrix-run scale, not a correctness bug.
- `llm_max_workers`/`num_processes` aren't recorded in run provenance.
- `scripts/launch_master_run.py` doesn't use any of the Plan 6 concurrency/distributed machinery yet — it's still the older single-process launcher.
- A possible future `TransactionRecord` run-scoping schema fix, if a transaction count is ever wanted on the dashboard (see §8).

## 8. Current, in-progress work: the interactive dashboard

**Status: fully designed and planned, zero code written.** The user asked for a Streamlit dashboard (in a `dashboard/` folder) showing live simulation progress plus whatever else is useful, with Start/Stop/Pause/Resume controls.

**A real scope conflict was found and explicitly overridden**: the project's own original master-instructions document (`dashboard.md`, §2) explicitly reserves all dashboard work for other group members and instructs Claude not to write it. This was flagged directly to the user before proceeding; the user said **"Proceed — that instruction no longer applies."**

**Design decisions, all made by the user directly** (not assumed):
- Controls are Start/Stop/Resume; "Pause" exists as a labeled button (per the original request) but is *identical* to Stop under the hood — there is no true mid-execution pause mechanism, and inventing one wasn't worth the complexity.
- Stack: **Streamlit** (pure Python, fastest to build/iterate).
- Stop = process termination, relying entirely on the existing checkpoint/resume mechanism from Plan 4 for safety (accepted cost: up to one simulated day's already-spent real LLM calls may need to be redone on Resume).
- A real (non-dry-run) launch requires the user to type the exact `matrix_run_id` shown on screen before a "Confirm real launch" button enables — mirrors the existing CLI go/no-go gate, adapted into the UI.
- Live progress is read directly from the SQLite database (already the shared source of truth for both single- and multi-process runs) rather than from a custom callback-fed status file — **zero changes to the tested simulation core** (`run_matrix`, `distributed_matrix_runner.py`).
- **No transaction count on the dashboard**: `database/models.py`'s `TransactionRecord` has no run-scoping column at all (its `buyer_id`/`seller_id` collide across cells the same way `agents`/`wallets` used to, before the Plan 6 fix — but nothing ever read `transactions_ledger` before this dashboard, so it was never fixed). Adding that column is a real schema change and was explicitly ruled out of scope for a "purely additive dashboard" plan; the user chose to just drop transaction counts from the dashboard rather than add a new schema column now.

**Architecture** (`docs/superpowers/specs/2026-08-07-phase3-dashboard-design.md`): three decoupled pieces, none touching the simulation core —
- `dashboard/app.py` — Streamlit UI, renders controls + live panels, never runs the simulation itself.
- `dashboard/runner.py` — a thin subprocess entrypoint (`python -m dashboard.runner --matrix-run-id ... --cell-keys ... --seeds ... --num-days ... [--dry-run|--real] [--distributed --num-processes N]`) that calls the existing `run_matrix`/`run_matrix_distributed`.
- `dashboard/process_control.py` — `start()`/`stop()`/`resume()`/`is_alive()`: launches `runner.py` via a detached `subprocess.Popen`, tracks it via a small JSON status file (`dashboard/status_store.py`) and OS-level PID liveness checks (`psutil`).
- `dashboard/queries.py` — read-only DB queries backing the live progress table (current day and LLM-decision counts per cell/seed, via `TimestepLogRecord` and `LLMDecisionRecord`).

**Implementation plan** (`docs/superpowers/plans/2026-08-07-phase3-dashboard.md`), 5 tasks: (1) status-file schema/I/O, (2) DB queries, (3) subprocess runner entrypoint, (4) process-control layer + new `pyproject.toml` optional dependency group (`streamlit>=1.36`, `psutil>=6.0`), (5) the Streamlit UI itself. During the plan-writing self-review (before any code was written), three of my own drafting mistakes were caught and fixed in the plan itself: the `TransactionRecord` scoping gap above; a distributed-mode usage callback that was written to discard its own `usage` argument and re-query an irrelevant counter; and a "go back and add these fields" placeholder-style instruction that was rewritten as complete inline code.

**Where execution actually stands**: a git worktree (`.worktrees/phase3-dashboard`, branch `phase3-dashboard`, base commit `b590697`) was created, a progress ledger was started, and Task 1's brief was extracted — but **the user said "stop" before any implementer subagent was dispatched**. No dashboard code exists anywhere in the repository. Resuming this work requires an explicit new request from the user; it should not restart automatically.

## 9. Master spec (`Experiment.md`) features: built vs. aspirational-only

To avoid overstating what exists, here is an honest accounting of which of the master spec's more ambitious features are actually implemented in the codebase today, checked directly against `src/`:

| Master spec feature | Status |
|---|---|
| CARA utility, per-agent risk coefficient | **Built**, and is what H1–H11 are actually regressed on |
| CRRA, risk-neutral, Epstein-Zin, multi-attribute utility | Modules exist (`src/utility/`) and are wired through `utility_factory.py`, but CARA is the one actually driving the Phase 3 hypotheses |
| Loss-driven risk-aversion adaptation | **Built** (`src/economy/risk_adaptation.py`) |
| Cross-border FX conversion tax | **Built** (`src/economy/fx_tax.py`), single-tax design after an added second tax was explicitly removed |
| EUR/USD FX volatility for H5 | **Built** (`src/economy/fx_dynamics.py`) |
| Dynamic shock engine (inflation, bank failure, depeg, fee spike) | **Built** (`src/economy/shocks.py`, `sandbox_scenarios.py`) |
| Live market data (Polygon) | **Built** (`src/llm/market_intelligence.py`) |
| Agent episodic memory | A `src/agents/memory.py` module exists; not verified in depth for this summary |
| Social/peer network effects, merchant acceptance cascades | **Not built** — no `network_effects.py` or `social/` module exists anywhere in `src/` |
| Dual "Factual vs. Self-Research" agent modes (tool-calling web/vector-DB research) | **Not built** — no evidence of agent tool-calling for self-directed research in `src/llm/` |
| Provenance metadata (seed, model, prompt hash, git hash, config hash) | **Built** (`src/simulation/provenance.py`) |
| Hallucination telemetry (expected vs. paid value) | **Built** (`src/llm/hallucination_detector.py`) |
| H1–H6 from the master spec | H1–H5 built; **H6 (privacy vs. friction) is explicitly reserved and deferred, never built** |
| Interactive Streamlit dashboard | Designed and planned (§8); **not yet implemented** |
| E2B sandbox integration (`e2b/` / `sandbox/`) | Scaffolding exists (`sandbox/experiment_dispatcher.py`, `sandbox_launcher.py`, `sandbox_manager.py`, `result_collector.py`, `sandbox_cleanup.py`); not covered in depth by this summary |

## 10. Standing rules that govern how this project is worked on

- **No unilateral assumptions** (project memory `feedback_no_assumptions.md`): any modeling, economic, or design decision — however small — gets asked about explicitly before being locked in. This has been followed consistently: every ambiguity or gap found in a master spec document, every architecture trade-off, every naming collision, and every scope conflict in this entire history was escalated via a direct question rather than resolved silently.
- **Full-scale/real-spend runs are their own explicit go/no-go gate**, always separate from a plan's code being merged. Real OpenRouter spend at 39-cell × 3-seed × 365-day scale is estimated (per the master spec's own cost acknowledgment) at low thousands to tens of thousands of dollars — this has not been authorized yet.
- **Reviews are mandatory and real**: every plan has gone through per-task review plus a final whole-branch review before merge, and every review round has found genuine Critical/Important issues that were fixed and re-verified — this is not a formality in this project's history, it has repeatedly caught real bugs (transaction-ID collisions, the agent-identity schema bug, the FX-tax double-charge, the usage-counter race, the H6 naming collision, the distributed-runner result-loss bug).

## 11. Open items / where to pick this back up

1. **Dashboard implementation** — plan is fully written and self-reviewed; needs an explicit "resume" instruction from the user before any implementer subagent is dispatched (§8).
2. **The 39-cell production run** (3 seeds × 365 days × 13 cells) — locked in as the target scale, not yet launched; needs its own explicit go/no-go.
3. All deferred follow-ups listed at the end of §7's Plan 6 entry (econometrics clustering by bare `agent_id`, `run_all_hypotheses`'s all-or-nothing failure mode, dataset-builder query performance, missing concurrency provenance fields, `launch_master_run.py` not yet updated to use Plan 6 machinery).
4. Stray untracked/root-level files present in the working tree that predate or sit alongside this summary (`assip.db`, `assip_phase3-real-run-2026-08-04_days0-1.db`, `checkpoints/`, `demo_pipeline_run.db`, `scripts/launch_demo_run.py`, `scripts/launch_master_run.py`, several `.md` reference documents) — none of these are currently tracked in git; they are working artifacts, not something this summary treats as "in scope" to clean up.
