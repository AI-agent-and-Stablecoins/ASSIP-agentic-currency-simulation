# Phase 3 Plan 5: Econometrics Engine — Design Spec

## 0. Source documents and why this spec exists

The master spec (`docs/superpowers/specs/2026-07-29-phase3-full-scale-simulation-design.md`
§2) lists the 6 in-scope hypotheses (H1-H5, H6 deferred) and the required
per-hypothesis output (β, SE, 95% CI, p-value, R², adjusted R²), citing
`Experiment.md` §7 as the source. `Experiment.md` (added to the repo root
2026-08-02) confirms both, and its §5B sandbox table maps 1:1 onto the 6
sandboxes Plan 3/4 actually built:

| Experiment.md §5B sandbox | Built as (`src/currencies/sandbox_currencies.py`) |
|---|---|
| 1. Liquidity vs. Governance | `liquidity_vs_governance` |
| 2. Governance vs. Stability | `governance_vs_stability` |
| 3. Liquidity vs. Stability | `liquidity_vs_stability` |
| 4. Asset Backing vs. Liquidity | `asset_backing_vs_liquidity` |
| 5. Asset Backing vs. Stability | `asset_backing_vs_stability` |
| 6. Asset Backing vs. Governance | `asset_backing_vs_governance` |
| 7. Privacy vs. Friction | deferred (H6) |

**Neither document specifies the actual regression design** (dependent
variable operationalization, functional form, controls, clustering, or an
explicit hypothesis→data-cell mapping) — both only list hypotheses and
required outputs. Per [[feedback-no-assumptions]], the methodology below
was built by (a) reconstructing what data the built cells/schema can
actually support per hypothesis, and (b) asking the user directly on the
four decisions with real methodological consequence, rather than guessing.
**User decisions** (2026-08-02, all four options were the "Recommended"
one offered):

1. **Regression unit for H1-H4 (and, by direct extension, H5 — same
   hypothesis shape, not asked separately but flagged here for override):
   per-decision logistic regression.** Every LLM currency-choice decision
   is one observation (chose option X = 1, else 0); logit on the
   regressor of interest (CARA `a` for H1-H4, EUR/USD volatility for H5),
   clustered by agent. Chosen over a per-agent-per-day panel or a
   per-agent cross-sectional aggregate for statistical power (thousands
   of decisions vs. 100 agents) and because it most directly operationalizes
   "revealed preference" — the agent's actual proposed choice, not an
   aggregate.
2. **H1 and H5 (the two hypotheses needing currency-ZONE variation, which
   none of the 6 sandboxes isolate) are tested using MASTER-simulation
   data only**, not pooled with the cross-border sandbox cells, to avoid
   conflating the zone-preference effect with a sandbox's own isolated
   factor.
3. **H4's "crisis proximity" regressor is continuous**: days since/until
   the nearest `crisis_warning`/`depeg_event`, not a binary "shock active"
   indicator — this matches the master scenario's actual designed
   0/5/10/20-day gap sweep and gives a dose-response gradient instead of
   a single binary contrast.
4. **Standard errors are clustered by agent** (not two-way agent+day, not
   unclustered) — standard panel-data practice for repeated per-agent
   observations.

## 1. What data source and cell(s) feed each hypothesis

| Hyp. | Claim | Cell(s) | Dependent variable | Key regressor |
|---|---|---|---|---|
| H1 | Higher CARA `a` → stronger preference for USD-zone stablecoins over EUR-zone stablecoins | **master** only (only cell with real zone variation) | Per-decision: `1` if `LLMDecisionRecord.currency`'s zone (`src.economy.fx_tax.currency_zone_of`) is USD, `0` if EUR (gold-backed/zone-neutral decisions excluded from this hypothesis's sample) | Agent's CARA `a` at decision time |
| H2 | Higher CARA `a` → prioritizing low bid-ask spread over low gas fees | **master** only | See §2 below (resolved) | Agent's CARA `a` at decision time |
| H3 | Higher CARA `a` → prioritizing GENIUS Act compliance over liquidity | `liquidity_vs_governance` sandbox (domestic + cross-border pooled, with a cell-identity fixed effect) | Per-decision: `1` if chosen currency is the higher-governance option, `0` if the higher-liquidity option | Agent's CARA `a` at decision time |
| H4 | Closer crisis/depeg proximity → stronger shift to gold-backed tokens | `asset_backing_vs_liquidity` + `asset_backing_vs_stability` sandboxes (the only 2 with a gold option) pooled with a cell-identity fixed effect, **plus** master's own H4 sweep instances | Per-decision: `1` if chosen currency is the gold-backed option, `0` otherwise | Days since/until nearest `crisis_warning`/`depeg_event` (signed, or two separate regressors for "approaching" vs "past") |
| H5 | Higher EUR/USD volatility → stronger preference for USD stablecoins in cross-border settlement | **master** only, filtered to decisions where buyer/seller `currency_zone` differ (naturally occurring — master's pairing is zone-agnostic, and the population is 50/50 USD/EUR) | Per-decision: `1` if chosen currency is USD-zone, `0` if EUR-zone (same construction as H1, different sample filter + regressor) | Realized EUR/USD volatility (rolling std of `TimestepLogRecord.eur_usd_exchange_rate` over a trailing window — window length TBD by implementer, default 30 days, flagged for review) |
| H6 | Privacy premium threshold | **Deferred** — no privacy-rail currency/chain config exists (see master spec §7) | — | — |

**Cell identity** (which of the 13 matrix cells a given `LLMDecisionRecord`
row came from) is recovered by parsing `LLMDecisionRecord.simulation_id`
(== the matrix runner's `run_id`, `f"{matrix_run_id}-{spec.key}-seed{seed}"`
per `src/simulation/matrix_runner.py`), **not** from the `scenario` or
`domestic_or_cross_border` columns. Confirmed by direct code trace:
`domestic_or_cross_border` is unconditionally the literal string
`"unknown"` for every row the production `run_matrix` → `persist_full_
timestep` pipeline ever writes (`database/repository.py`'s
`_llm_decision_log_entry`, a known and already-documented placeholder —
the per-decision domestic/cross-border context only exists inside
`run_timestep`'s buyer/seller loop, not in the `TimestepLLMDecisionRecord`
this function receives). `scenario` is *also* insufficient alone: a
sandbox's domestic and cross-border cells share the exact same
`ScenarioConfig` object (`build_sandbox_scenario` is called once per
sandbox, not once per cross-border variant), so `scenario` is IDENTICAL
for e.g. `liquidity_vs_governance_domestic` and
`liquidity_vs_governance_cross_border` — it cannot tell them apart.
`simulation_id` is the only field that fully disambiguates all 13 cells
(it embeds the literal `spec.key`, e.g. `"master"` or
`"liquidity_vs_governance_cross_border"`, between the `matrix_run_id-`
prefix and the `-seed{N}` suffix). Plan 5 adds a small parsing helper
(`cell_key_from_run_id`) rather than fixing `domestic_or_cross_border`
itself — parsing an already-fully-disambiguating field is sufficient and
doesn't require touching the already-merged Plan 4 persistence code
beyond the one addition in §2.1.

All dependent variables are read from `LLMDecisionRecord.currency`
(the agent's *proposed* choice), not `TransactionRecord` (settled trades
only). Reasoning: a transaction's SETTLED status depends on the
counterparty's independent accept/reject decision, so filtering to
settled trades only would contaminate a buyer's revealed *preference*
with the seller's independent acceptance behavior — a selection effect
unrelated to what these hypotheses claim to measure. This is a
**Resolution (not a stakes decision)** — flagged for the user to override
if settled-only is actually intended.

The regressor "agent's CARA `a` at decision time" reads
`AgentStateRecord.cara_coefficient`, joined on `(run_id, timestep,
agent_id)` matching the decision's `(simulation_id, timestep, agent_id)`
— **not** `LLMDecisionRecord.utility_parameters["risk_aversion"]` (a
tempting-looking but WRONG source: `_llm_decision_utility_parameters`,
`database/repository.py`, only sets the `"risk_aversion"` key when
`agent.risk_aversion is not None`, which is `None` for `utility_type ==
"risk_neutral"` — i.e. exactly the `a == 0.0` agents. Using it would
silently drop every risk-neutral agent's decisions from the regression
sample, which is a real, silent selection bias for a continuous-`a`
regressor.) `AgentStateRecord.cara_coefficient` correctly holds `0.0`
(not `None`/absent) for risk-neutral agents (`agent.cara_coefficient`,
set at construction per `src.agents.agent_factory.build_agent`'s
`nominal_cara` logic) and reflects day-to-day adaptation (Plan 4 added
loss-driven CARA adaptation, `src.economy.risk_adaptation
.adapt_cara_coefficient`, which mutates `agent.cara_coefficient` — the
same attribute `AgentStateRecord` is built from every day).

## 2. H2 design (resolved)

Unlike H1/H3/H4/H5, H2 ("prioritizes low spread over low gas fees") has no
sandbox that isolates spread-vs-gas-fee as a factor pair (the 6 sandboxes
vary liquidity/governance/stability/asset-backing at the *currency* level;
gas fee is a *chain*-level property, orthogonal to all 6). **User decision
(2026-08-02): the explicit tradeoff-sample design.** For each decision,
identify which candidate (currency, chain) pair was "spread-optimal" vs.
"gas-optimal" among that round's `generate_candidates` offering; keep
only decisions where these two differ (a genuine tradeoff existed that
round); regress "chose the spread-optimal option" (1/0) on `a`. Chosen
over two independent regressions (one per factor) because it more
literally tests the hypothesis's "over" framing — a genuine head-to-head
tradeoff, not two separately-estimated preferences — despite the smaller,
selected sample and the extra data this requires (§2.1).

**Correction from an earlier draft of this spec**: there is no
`bid_ask_spread` field anywhere in the codebase (`CurrencyChainOption` in
`src/blockchain/routing_engine.py`, and `CurrencyConfig` in
`src/currencies/currency.py`, both confirmed by direct read). The
established stand-in, used this way since Plan 2's design spec
(`docs/superpowers/specs/2026-07-22-phase2-llm-negotiation-layer-design.md`
§"Liquidity vs. gas fees"), is `liquidity_score` (`CurrencyChainOption
.liquidity_score`, `float` in `[0, 1]`, higher = better/tighter spread).
"Spread-optimal" therefore means the candidate with the HIGHEST
`liquidity_score` among that round's candidates; "gas-optimal" means the
candidate with the LOWEST `gas_fee`. This is a factual correction (using
the codebase's one existing spread proxy, not a new modeling choice), not
something that needs re-confirming.

### 2.1 New requirement: persist each decision's spread-optimal/gas-optimal choice (resolved)

`LLMDecisionRecord` currently stores only the *chosen* `currency`/`chain`,
not the candidate set `generate_candidates` offered that round — needed
to compute "spread-optimal"/"gas-optimal" per decision for H2.
`generate_candidates` is called exactly once per buyer/good/day
(`src/simulation/timestep.py`, `candidates = generate_candidates(buyer
.wallet.balances, env.currencies, env.chains, env.liquidity_pools,
trust_ledger=env.trust_ledger)`), shared by both the LLM-driven and
rule-based branches that follow, and is not persisted anywhere today.

**User decision (2026-08-02): persist this going forward**, via a small
additive change to the already-merged Plan 4 persistence code, rather
than reconstructing it after the fact by replaying `TrustLedger`/
liquidity-pool state from `intervention_logs` (rejected: real
replay-correctness risk reproducing time-decayed trust/liquidity state
exactly, and substantially more implementation work for a less verifiable
result).

**Refinement (not a stakes decision, a scope-minimization of the above):**
rather than persisting the FULL candidate list (which could be
`len(currencies) x len(chains)` entries per decision — non-trivial extra
storage at real-run scale, 13 cells x 5 seeds x 365 days x many decisions
per day), persist only the two DERIVED values H2's analysis actually
needs: which candidate was spread-optimal (`currency_symbol`/`chain_name`
of the highest-`liquidity_score` candidate) and which was gas-optimal
(`currency_symbol`/`chain_name` of the lowest-`gas_fee` candidate), as 4
new nullable string columns on `LLMDecisionRecord`
(`spread_optimal_currency`, `spread_optimal_chain`, `gas_optimal_currency`,
`gas_optimal_chain`). Nullable so every pre-existing row/test (which never
populated these) stays valid; computed once per buyer/good/day from the
existing `candidates` list already in scope at the call site, no new
network/compute cost.

**Sequencing consequence, not a stakes decision but important to flag
explicitly:** this only applies to runs recorded *after* this change
merges — it does NOT retroactively add candidate data to any
already-collected run. Since the real full-scale launch has not happened
yet (per the standing go/no-go gate), this change must land and be
verified *before* that launch, not after, or H2 will have no usable data
once the real run starts. This is Plan 5's Task 1 (see the implementation
plan), and touches `database/models.py` / `database/repository.py`
/`src/simulation/timestep.py` again despite Plan 4 already being merged —
a legitimate additive extension of the persisted schema, not a redo of
already-reviewed work.

## 3. Statistical methodology

- **Model**: logistic regression (`statsmodels.api.Logit` or
  `statsmodels.discrete.discrete_model.Logit`), one model per hypothesis
  (H4 may be one pooled model across its 2 gold sandboxes + master H4
  instances, with a cell fixed effect — see §1).
- **Controls / fixed effects**: agent role (`agent_type`) and assigned LLM
  model (`actual_model`) as categorical fixed effects in every regression.
  Rationale (**Resolution, not a stakes decision** — flagged for
  override): Plan 3 assigns each agent one fixed OpenRouter model for the
  whole run; if models have systematically different behavioral quirks
  unrelated to CARA `a`, omitting a model fixed effect would let that
  confound the estimated `a` coefficient.
- **Standard errors**: clustered by `agent_id`
  (`statsmodels`' `cov_type="cluster"`, `cov_kwds={"groups": agent_ids}`)
  — per the user's decision in §0.
- **R² for a logit model**: McFadden's pseudo-R² (`1 - LL_model/LL_null`)
  and McFadden's adjusted pseudo-R² (`1 - (LL_model - k)/LL_null`, `k` =
  number of estimated parameters), reported in place of OLS's R²/adjusted
  R² — the standard logit analogs, since Experiment.md's "R² and adjusted
  R²" requirement doesn't specify a model family and OLS's R² isn't
  defined for a binary outcome. **Resolution (not a stakes decision)** —
  flagged for override if a different pseudo-R² variant (Cox-Snell,
  Nagelkerke) is preferred.
- **Sample size (N)** is reported alongside the required 6 outputs for
  every hypothesis, since with clustered SEs and pooled multi-cell
  samples (H3/H4) N isn't otherwise obvious from the output table alone.
- **New dependency**: `statsmodels` is not currently installed anywhere in
  this repo (`pyproject.toml` confirmed by direct read — only `pandas` is
  a core dependency; `numpy`/`scipy`/`statsmodels` are absent even
  transitively beyond what `pandas` itself pulls in). Added as a new
  `[project.optional-dependencies]` group (`econometrics = ["statsmodels>=0.14"]`),
  following the existing `observability`/`market-data`/`llm` pattern —
  not a core dependency, since nothing outside this new package needs it.

## 4. Module structure and output

- New `src/econometrics/` package (not `metrics/` — the existing
  `metrics/` directory is purely descriptive statistics, e.g. market
  share/adoption curves/governance-preference averages; none of it does
  regression, so this is new functionality, not an extension of existing
  files):
  - `src/econometrics/hypothesis_datasets.py` — per-hypothesis query
    functions building the regression-ready dataset (one row per LLM
    decision) from the persisted `LLMDecisionRecord`/`TimestepLogRecord`/
    `AgentRecord` tables, per §1's per-hypothesis table.
  - `src/econometrics/hypothesis_regressions.py` — the actual
    `statsmodels` Logit fit + clustered SEs + McFadden pseudo-R² per
    hypothesis, returning a typed result object (β, SE, CI lower/upper,
    p-value, pseudo-R², adjusted pseudo-R², N).
  - `src/econometrics/report.py` — assembles all 5 hypotheses' results
    into one output table (format TBD by implementation plan — CSV for
    the paper, and/or a structured object the (not-yet-built) Streamlit
    dashboard can read directly).
- Depends only on the database (via `database.session`) — no dependency
  on `src/simulation/matrix_runner.py` or any live `Environment`, since
  this runs AFTER a real (or dry-run test) matrix has already persisted
  its data. Fully independent of API keys/network access.

## 5. What's explicitly out of scope for this plan

- **H6** (privacy premium) — deferred per master spec §7, no privacy-rail
  currency/chain config exists.
- **The Streamlit dashboard** (`dashboard/app.py`) — deferred per master
  spec §7, built after this data run completes as a viewer over collected
  data, not before. This plan produces the regression *engine* and its
  output table/object; wiring it into a dashboard view is separate,
  later work.
- **Comparative Factual vs. Self-Research mode statistics** (master spec
  §4) — self-research mode itself is deferred (master spec §7), so there
  is no second mode's data to compare against yet.
