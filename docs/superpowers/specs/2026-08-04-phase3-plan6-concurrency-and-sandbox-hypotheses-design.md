# Phase 3 Plan 6: Concurrency + Sandbox-Preference Hypotheses — Design Spec

## 0. Source documents and why this spec exists

This plan originates from a direct user request (2026-08-04), not from
`Experiment.md` or the master spec — the user asked to (a) run the
simulation faster via GPU/delegated parallel execution, and (b) test
"which coin wins" for each of the 6 factor-isolation sandboxes (liquidity
vs. governance, governance vs. stability, liquidity vs. stability, asset
backing vs. liquidity, asset backing vs. stability, asset backing vs.
governance), repeated for both domestic and cross-border pairing.

Investigation before this spec was written (see conversation) established:

- **All 6 sandbox pairs and their domestic/cross-border variants already
  exist** (Plan 3/4, `src/currencies/sandbox_currencies.py`,
  `src/simulation/matrix_runner.py` — 12 sandbox cells + 1 master cell =
  13 total). Cross-border FX modeling (50/50 USD/EUR agent zone split,
  small conversion tax when transacting outside one's zone) also already
  exists (`src/agents/population.py`, `src/economy/fx_tax.py`).
- **The actual gap**: of the 6 pairs, only `liquidity_vs_governance` has a
  dedicated "which coin won" regression (H3). The other 5 have no
  dedicated hypothesis. This plan adds H6–H10 to close that gap.
- **GPU acceleration would not help**: LLM inference is 100% remote via
  OpenRouter (`src/llm/llm_router.py`, synchronous `httpx.Client`, no
  local model weights anywhere in the codebase). The user confirmed (after
  this was surfaced) that no local-compute GPU target exists yet, and
  dropped the GPU requirement. The actual lever for wall-clock speed is
  **concurrency** — parallel/async execution — which this plan builds
  instead.
- **Per [[feedback-no-assumptions]]**, every modeling/design choice below
  marked "User decision" was resolved by asking the user directly during
  brainstorming, not guessed.

## 1. New hypotheses H6–H10

**User decisions (2026-08-04):**
1. Each of the 5 new hypotheses is reported **separately for its domestic
   cell and its cross-border cell** (10 result rows, not 5 pooled rows) —
   unlike H3, which pools domestic+cross-border with a `cell_key` fixed
   effect. H3 is left unchanged; the reporting shape is allowed to differ
   between H3 and H6–H10 rather than retrofitting H3.
2. Each hypothesis is a directional claim (mirroring H1–H5's style, not a
   neutral/exploratory report), with the specific direction per pair
   approved as follows:

| Hyp. | Sandbox | Claim | Confidence |
|---|---|---|---|
| H6 | `governance_vs_stability` | Higher CARA `a` → prioritizes peg stability (lower `peg_error`) over governance/compliance | High — mirrors H4's realized-loss-avoidance logic |
| H7 | `liquidity_vs_stability` | Higher CARA `a` → prioritizes peg stability over liquidity | High — same logic vs. execution/slippage risk |
| H8 | `asset_backing_vs_liquidity` | Higher CARA `a` → prioritizes gold/hard-asset backing over liquidity | High — static analogue of H4's flight-to-gold |
| H9 | `asset_backing_vs_stability` | Higher CARA `a` → prioritizes the FDIC-insured deposit option (better peg + insurance) over gold backing | **Lower** — this sandbox swap bundles asset-class AND a large peg_error gap (0.015 vs 0.0001) in one move; the deposit side is the safer bundle here, opposite framing from H8 |
| H10 | `asset_backing_vs_governance` | Higher CARA `a` → prioritizes governance/compliance quality over asset-backing type | **Lower** — governance scores (0.75 vs 0.70) and issuer risk (0.25 vs 0.20) are close between the two options, a subtler contrast than the other pairs |

Both lower-confidence hypotheses (H9, H10) were explicitly approved as-is
by the user rather than revised — flagged here so the eventual report's
interpretation carries that caveat forward.

### 1.1 Implementation shape

Following the existing `src/econometrics/` pattern
(`hypothesis_datasets.py` / `hypothesis_regressions.py` / `report.py`):

- **A single parameterized dataset-builder helper**, not 5 near-duplicate
  functions, since H6–H10 share an identical shape (per-decision logit:
  `1` if chosen currency is the "higher-X" option in a given sandbox
  pair, `0` otherwise; regressor = agent's CARA `a` at decision time;
  clustered by `agent_id`; fixed effects `agent_type`, `actual_model`) —
  the only things that vary per hypothesis are the sandbox key and which
  of the two `CurrencyConfig` options counts as "higher-X". Signature:
  `build_sandbox_preference_dataset(session, sandbox_key, higher_option_selector, cell_variant, matrix_run_id=None) -> pd.DataFrame`
  where `cell_variant` is `"domestic"` or `"cross_border"` (selecting
  exactly one of the sandbox's two cells, unlike H3's `_H3_CELLS` set)
  and `higher_option_selector` picks the "higher-X" symbol from the
  sandbox's `(option_a, option_b)` tuple. Concrete selector per
  hypothesis (each resolves to exactly one option since the sandbox pairs
  were constructed with a clear winner on the named dimension — no ties):
  - H6: `a.symbol if a.peg_error <= b.peg_error else b.symbol` (lower `peg_error` = "higher stability")
  - H7: same selector as H6 (stability side of the pair)
  - H8: `a.symbol if isinstance(a, GoldBackedConfig) else b.symbol` (the gold-backed option)
  - H9: `a.symbol if isinstance(a, TokenizedDepositConfig) else b.symbol` (the FDIC-insured deposit option, not the gold option — H9's claim is deposit-wins, opposite framing from H8)
  - H10: `a.symbol if a.governance_score >= b.governance_score else b.symbol` (higher `governance_score` = "higher governance")
- `regress_h6` .. `regress_h10` in `hypothesis_regressions.py`, each
  calling the shared builder twice (once per `cell_variant`) and
  `fit_clustered_logit` twice, returning two `RegressionResult`s (e.g.
  `hypothesis="H6_domestic"`, `hypothesis="H6_cross_border"`).
- `report.py`'s `run_all_hypotheses` extended to include all 10 new
  results (15 total regression results: H1, H2, H3, H4, H5, then
  H6_domestic, H6_cross_border, ... H10_domestic, H10_cross_border).
  `results_to_dataframe`/`write_report_csv` need no changes — the output
  schema (`hypothesis`, `regressor`, `beta`, `se`, ...) already
  accommodates arbitrary hypothesis labels.
- Mirrors the existing test pattern: unit tests per new dataset builder
  (empty-sample handling, correct cell scoping, correct higher-X
  labeling) and per new regression function, following the existing
  `tests/econometrics/` structure for H1–H5.

## 2. Concurrency architecture

**Investigation finding**: the real 2026-08-04 master run's actual pace
was ~3.5–4 hours per simulated day, sequential (confirmed from
`assip.db`'s `llm_decisions` timestamps: day 0→day 1 gap ≈ 3.5 hours). At
that pace a single 365-day, single-seed run takes ~53 days of
wall-clock — infeasible for a 13-cell × 3-seed matrix without
concurrency.

**Safety analysis finding** (investigated before designing this, not
assumed): parallelizing agent decisions within a simulated day is
economically safe. Different agents' decisions have no same-day
informational dependency on each other (per-day market/price context is
built once, before any agent acts, and never refreshed mid-day); listings
in the marketplace are not consumed/scarce, so buyer processing order
doesn't change counterparty availability; the per-day RNG shuffle is
consumed once upfront, not incrementally. The one real hazard: two
buyers concurrently settling against the same seller race on that
seller's wallet balance (non-atomic read-modify-write in
`Wallet.deposit`/`withdraw`).

### 2.1 Within-day parallelism

Split each simulated day into two phases:

1. **Parallel decision phase** — a thread pool (these are I/O-bound
   network calls to OpenRouter, not CPU-bound work, so threads are
   sufficient; no multiprocessing needed here) runs each buyer-seller
   `run_llm_negotiation` call concurrently. Each individual negotiation
   stays internally sequential (a negotiation's own back-and-forth
   genuinely requires each round to see the prior round's offer —
   confirmed, not assumed), but different negotiations (different
   buyer-seller pairs) run at the same time. Results are collected, not
   yet applied.
2. **Serial settlement + persistence phase** — back on the main thread,
   exactly as today: wallet debits/credits and database writes happen in
   order, one at a time. This avoids the wallet race entirely (only one
   thread ever touches wallets or the DB) without needing any new locking
   primitives.

Existing retry/backoff logic for OpenRouter 429s
(`src/llm/llm_router.py`'s `RetryConfig`) is reused as-is inside each
thread's call — no changes needed there.

### 2.2 Cross-cell/seed parallelism

Each of the 13 cells × 3 seeds is a fully independent simulation (own
environment, own RNG stream) — these run as separate OS processes
(`multiprocessing` or separate subprocess invocations). The only shared
resource is the single `assip.db` SQLite file at each day's persistence
step. To avoid "database is locked" failures under concurrent-process
writes: enable SQLite WAL mode + a busy-timeout/retry on the engine
(`database/session.py`), keeping one unified database rather than
per-process DB files — this way the existing econometrics queries (which
scan one DB scoped by `matrix_run_id`) require no changes.

## 3. Cost/token logging

There is currently no cost or token-usage tracking anywhere in
`src/llm/llm_router.py` — real OpenRouter spend has been invisible even
after the fact. This plan adds minimal logging of tokens/cost per LLM
call (OpenRouter's response includes usage data), surfaced at minimum via
structured log lines and/or a lightweight running total, so spend is
visible during a long matrix run rather than discovered after the fact.

## 4. Scale: locked-in target

**User decision (2026-08-04)**: **3 seeds × 365 days × 13 cells** (39
full-year cell-runs) for the production run, chosen without a pre-run
timing/cost pilot (the user explicitly opted out of piloting first).
Reasoning discussed: a single 365-day seed already yields ~90,000+
decision-rows per cell (100 agents × 365 days × ~2.6 decisions/day),
already large statistical power for the regression itself; the purpose
of 3 seeds is robustness (checking a finding isn't an artifact of one
random draw of agent-model assignment/activation order/shock timing),
not incremental row-count.

**Unchanged from Plan 4**: launching this real run remains a separate,
explicit go/no-go gate requiring the user's confirmation immediately
before launch, distinct from approving this design/implementation
(per [[project-assip-phase3]]). Nothing in this plan authorizes launching
the 39-cell-run matrix on its own.

## 5. Out of scope / explicitly deferred

- GPU acceleration (dropped per user decision, §0).
- A pre-run timing/cost pilot (user opted for a locked-in scale instead,
  §4) — cost logging (§3) is the substitute safeguard.
- Any change to H1–H5's existing methodology or H3's pooled
  domestic+cross-border reporting shape.
- Async/`asyncio` rewrite of the LLM router — thread pool concurrency
  (§2.1) is the chosen mechanism, not a full async rewrite, since the
  existing synchronous `httpx.Client` + retry logic can be reused as-is
  inside worker threads.
