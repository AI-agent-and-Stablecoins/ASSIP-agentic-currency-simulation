# Phase 3: Full-Scale Multi-Agent Simulation — Design Spec

**Status:** Approved by user (2026-07-29), pending write-up self-review.
**Scope:** This is the final data-collection phase. Output must be complete,
detailed raw data (including full transaction and negotiation logs) suitable
for the econometrics engine and for later dashboard/paper use. Nothing in
this spec should be modified, added to, or reinterpreted without checking
back with the user first — this run produces the final dataset.

Source documents this spec reconciles: `Experiment.md` (Phase 3 master
engineering spec), `Untitled document.md` (shocks/trust/historical-context
extension, marked "not yet implemented" in the source), `Phase 4 Model
List.md` (OpenRouter model roster), `Agentic AI and Preferences for Medium
of Exchange.md` and `ASSIP Presentation notes.md` (hypotheses and framing),
and the Phase 2 slide deck (architecture/results already produced).

---

## 1. Branch Strategy

Phase 2's LLM negotiation layer (`llm_router.py`, `agent_reasoning.py`,
`decision_schema.py`, `decision_adapter.py`, `hallucination_detector.py`,
`market_intelligence.py`, `experiment_007_governance_prompting.py`, and the
`configs/llm/models.yaml` 5-model roster) exists only on the
`worktree-phase2-llm-negotiation` branch. It has never been merged to
`main`, where `src/llm/*.py` are still empty stubs.

**Action:** merge `worktree-phase2-llm-negotiation` into `main` before any
Phase 3 code is written. All Phase 3 work happens on `main` after that merge.

---

## 2. Hypotheses In Scope

All six hypotheses from `Experiment.md` §7, using its H1–H6 numbering as
authoritative (this supersedes the differently-numbered "H1–H3" subset in
`Untitled document.md`, which mapped to originals #3–#5 only):

| # | Hypothesis | Directionality tested |
|---|---|---|
| H1 | Higher CRRA σ → stronger preference for USD stablecoins over EUR stablecoins | risk aversion → currency bloc |
| H2 | Higher CRRA σ → stronger preference for low bid-ask spread over low gas fees | risk aversion → liquidity vs. fee |
| H3 | Higher CRRA σ → stronger preference for GENIUS Act compliance over liquidity | risk aversion → governance vs. liquidity |
| H4 | Closer perceived crisis/depeg proximity → stronger shift to gold-backed tokens (PAXG/XAUT) | crisis proximity → gold |
| H5 | Higher EUR/USD volatility → stronger preference for USD stablecoins in cross-border settlement | FX volatility → USD bloc |
| H6 | Privacy premium threshold (USDCx vs. anonymous rail) | **Deferred** — no privacy-rail currency/chain config exists yet; out of scope for this run, to be built as a follow-up |

Each hypothesis is evaluated with the full econometric output specified in
`Experiment.md` §7: coefficient (β), standard error, 95% CI, p-value, R² and
adjusted R².

---

## 3. Agent Population — 100 Agents

### 3.1 Role composition

Mapped onto the 5 existing agent profiles (`configs/agent_profiles/*.yaml`)
— no new "Regulator" profile is built:

| Role | Count | Existing profile |
|---|---|---|
| Consumer (buyer) | 35 | `consumer.yaml` |
| Merchant (seller) | 35 | `merchant.yaml` |
| Bank | 10 | `bank.yaml` |
| Investor | 10 | `investor.yaml` |
| Institution | 10 | `institution.yaml` |

Consumer/Merchant are weighted heaviest since they drive the majority of
transaction volume the negotiation engine and hypotheses depend on.

### 3.2 Home currency zone (for H1/H5 cross-border logic)

Each of the 100 agents is tagged with a home currency zone, **USD or EUR,
split 50/50** (50 agents each) for balanced regression power. This is
independent of role — e.g. both US-zone and EU-zone consumers exist.

### 3.3 Per-agent CRRA risk aversion (σ)

Rather than one fixed σ per role (as the current YAML profiles define, e.g.
`consumer.yaml`'s flat `risk_aversion: 3.0`), **each of the 100 agents is
individually assigned a σ** sampled across `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0}`
(0 = risk neutral per the user's stated convention). This is what gives
H1–H3 genuine within-sample variance to regress on, instead of 5 discrete
role clusters. σ is not static: it adapts across the run per
`Experiment.md` §3E's loss-driven scaling formula (`σ_{t+1} = min(σ_max, σ_t
+ η_risk · Loss_t / W_real_t)`), so realized losses during shocks
increase an agent's σ over time.

Role-level `utility_type` and `multi_attribute` weights (e.g. merchant's
governance/liquidity/gas/volatility/compliance weights) stay as currently
configured per role — only σ is individualized.

### 3.4 Per-agent LLM model assignment

Each agent is assigned **one fixed model for the entire run**, sampled from
the ~90-model list in `Phase 4 Model List.md`, shuffled to fill 100 slots
(a handful of models get 2 agents to reach 100). Model becomes a per-agent
trait exactly like risk profile or currency zone, joinable against decision
outcomes for analysis (e.g. "does model family correlate with governance
sensitivity independent of assigned σ?").

**Preflight step:** before assignment, verify every candidate model ID
against OpenRouter's live `/models` endpoint (same pattern as the existing
`verify_model_roster()` in `src/llm/llm_router.py`). Some entries in the
Phase 4 list (e.g. older/deprecated models) may no longer be available by
run time — any that fail preflight are excluded from the sampling pool and
reported to the user before the run launches, not silently dropped after
the fact.

---

## 4. Shock Engine, Trust Ledger, Historical Context

Implemented exactly as specified in `Untitled document.md`, which is
promoted from "design spec, not yet implemented" to implemented in this
phase:

- **12 shock types** total: the 4 existing (`inflation`, `bank_failure`,
  `gold_rally`, `fee_spike`) plus 8 new (`regulatory_enforcement`,
  `liquidity_crunch`, `governance_downgrade`, `depeg_event`,
  `crisis_warning`, `fx_volatility_shock`, `fx_rate_shock`,
  `capital_controls`), each carrying a `target_currency` or `target_issuer`
  field. `bank_failure` is extended (not replaced) with an optional
  `target_issuer` field for contagion tests.
- **`TrustLedger`** (`src/economy/trust.py`): per-currency dynamic trust
  score, initialized at `governance_score`, decaying toward baseline on
  quiet days (`λ_recover`), dropping sharply on shock days (`λ_shock`),
  with smaller contagion hits (`λ_contagion`) to same-asset-class/issuer
  currencies. Constants live in `configs/economy/trust_params.yaml`
  (`λ_shock ≈ 0.5`, `λ_recover ≈ 0.03`, `λ_contagion ≈ 0.2·λ_shock`, rolling
  window `W` for volatility perception, e.g. 30 days), never hardcoded.
- **Event log** (`src/economy/event_log.py`): append-only
  `{day, shock_type, target, severity}` records, written by
  `timestep.py` whenever `apply_shock` fires.
- **`AgentObservation` history extension** (`src/llm/agent_reasoning.py`):
  adds `CurrencyHistory` (trust_now, trust_30d_ago, trust_min_90d, trend,
  depeg_events_90d, last_event_days_ago, recent_events) and `MacroHistory`
  (confidence_now, confidence_30d_ago, days_since_last_shock,
  last_shock_type) to the prompt, rendered as a distinct "History" section
  so the LLM can reason about trajectory, not just point-in-time state.
- **Agent personal memory** (`base_agent.py`'s existing `self.memory`) is
  extended to record crisis-relevant first-person experience (e.g. "On day
  5 I held USDC through a banking crisis and lost nothing").

### H4 proximity design (avoids needing separate runs per warning-gap value)

Within the single 365-day master simulation run, schedule **multiple
`crisis_warning` → `depeg_event` pairs** at different points in the year,
each targeting a different currency, each with a different warning gap —
**0, 5, 10, and 20 days** — so gold-preference-vs-proximity variation exists
*within* one run's data rather than requiring 4 separate run conditions.

---

## 5. Experiment Matrix

| Component | Agents | Duration | Seeds |
|---|---|---|---|
| Master simulation ("one big simulation") | all 100, all currencies/chains/mechanisms | 365 days | 5 |
| 7 factor-isolation sandboxes (H1–H4 pairs from `Experiment.md` §5B, domestic) | same 100 agents, tradable currency universe restricted to that sandbox's Option A/B pair | 365 days | 5 each |
| Same 7 sandboxes, repeated cross-border (US-zone vs. EU-zone agents paired, FX conversion tax applies) | same 100 agents, same restriction, cross-border pairing forced by currency zone | 365 days | 5 each |

Sandboxes reuse the same 100-agent population (identical σ, model, and
currency-zone assignments) rather than a separately constructed smaller
population — this keeps every agent's identity, model, and risk aversion
consistent across every experiment cell, so a given agent's choices can be
compared across master sim vs. each sandbox vs. cross-border variants.
Only the tradable currency universe changes per sandbox (restricted to
that sandbox's two options), which is what makes it a factor-isolation
cell rather than a free-choice one.

Research mode: **Factual only** for this run (agents receive facts injected
directly into the prompt — issuer audit scores, peg history, governance
data — via the existing `market_intelligence.py` corpus). Self-research
mode (tool-calling over vector DB/news archives/web search) is explicitly
**deferred** — see §7.

---

## 6. Data Persistence

Extends the existing SQLAlchemy schema (`database/models.py`) rather than
reintroducing the removed `schema.sql`. New tables, matching
`Experiment.md` §8's raw-persistence requirements:

- `simulation_runs` — provenance: `run_id`, `scenario_name`,
  `research_mode`, `random_seed`, `openrouter_model_id` (or roster hash for
  multi-model runs), `prompt_version_hash`, `git_commit_hash`,
  `config_hash`, `created_at`.
- `intervention_logs` — step-indexed shock events (`timestep`,
  `shock_type`, `target_variable`, `magnitude`), fed by the new event log.
- `timestep_logs` — daily macro state (`inflation_rate`,
  `confidence_index`, `eth_gas_fee_gwei`, `solana_gas_fee_usd`,
  `eur_usd_exchange_rate`).
- `agent_states` — per-agent per-day snapshot (`risk_profile`,
  `crra_sigma`, `real_purchasing_power`, per-currency balances,
  `utility_score`).
- `agent_memory_logs` — episodic memory text per agent per day.

Existing tables extended:

- `llm_decisions` — add a `system_prompt` text field storing the **full
  rendered prompt**, not just `rendered_prompt_hash`. The user explicitly
  asked for maximally detailed raw data; a hash alone doesn't let anyone
  inspect what the model actually saw.
- `hallucinations` — already close to `Experiment.md`'s
  `hallucination_telemetry` design; add `is_hallucination` boolean
  (thresholded classification) alongside the existing `overpayment_pct`.
- `transactions` — add `fx_tax_paid` for cross-border conversion friction.

---

## 7. Explicitly Deferred (follow-up items, not in this run)

- **H6 privacy sandbox** — needs a new currency/chain type (USDCx /
  Aleo-style anonymous rail) built from scratch; no config exists today.
- **Self-research mode** — tool-calling agent access to vector DBs/news
  archives/web search, per `Experiment.md` §4. The user will provide a
  background-research file to seed this later; **remind the user about
  this follow-up once the full-scale run is complete**, and ask them for
  that file at that time (it was referenced mid-conversation but not yet
  attached).
- **Streamlit dashboard** (`dashboard/app.py`) — built after this data run
  completes, as a viewer over the collected data, not before.

---

## 8. Cost & Scale (acknowledged by user)

Rough order of magnitude for the full matrix (100 agents × ~90 models ×
365 days × 5 seeds × master sim + 7 domestic sandboxes + 7 cross-border
sandboxes): **~800K–1M+ individual LLM API calls**, likely **multiple
weeks of wall-clock time**, and a **real OpenRouter API spend plausibly in
the low thousands to tens of thousands of dollars** given several of the
90 models are frontier-tier. The user has explicitly approved proceeding
at this full scale. A second, explicit confirmation will be requested
immediately before the run that actually incurs this spend is launched.

---

## 9. Provenance & Reproducibility

Per `Experiment.md` §6: every run captures `random_seed`, exact
`openrouter_model_id` per agent, `prompt_version_hash`, `git_commit_hash`,
UTC `timestamp`, and `config_hash` (SHA-256 of resolved YAML configs).
Every shock/intervention is logged with its exact step index.
