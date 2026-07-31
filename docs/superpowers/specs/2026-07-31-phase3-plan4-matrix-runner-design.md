# Phase 3 Plan 4: Matrix Runner / Experiment Orchestration — Design Spec

**Status:** Design spec, not yet implemented.
**Scope:** Wires the fully-built Plan 1-3 machinery (persistence schema, shock
engine/TrustLedger/history, 100-agent population) into a real, LLM-driven,
multi-day, fully-persisted simulation, plus the two remaining unbuilt economic
mechanisms (cross-border FX tax, loss-driven CARA adaptation), plus the
6-sandbox factor-isolation matrix and the 365-day master scenario. This is
the largest and most consequential Phase 3 plan — it is what actually
produces runnable code for the final dataset, though **launching the real,
billed 65-run matrix remains a separate, explicit go/no-go gate**, per the
master spec §8 and every prior plan's hand-off notes. Nothing in this spec
should be modified, added to, or reinterpreted without checking back with
the user first.

**Every non-trivial decision below was confirmed with the user directly
during design (2026-07-30/31), not assumed** — see each subsection's "User
decision" note. Where a decision was mine to resolve (a mechanical
completion of already-established Plan 1/2/3 intent, or a low-stakes
telemetry formula with no hypothesis-critical stakes), it's marked
"Resolution (not a stakes decision)" and flagged for the user to override on
review.

---

## 1. Real LLM decisions replace the deterministic day-loop path

**User decision:** wire real per-agent LLM decisions into the day loop,
replacing `run_timestep`'s current `buyer.choose_currency_and_chain(candidates)`
call. **User decision:** negotiation rounds use the full LLM-vs-LLM engine
(`run_llm_negotiation`), not the deterministic `negotiate()` formula.

### 1.1 What already exists (confirmed by direct code inspection, not assumed)

- `src/llm/agent_reasoning.py`'s `decide(agent_class, context, roster, client,
  supported_currencies, supported_chains, policy_name, retry_config,
  max_correction_attempts, deterministic_fallback) -> LLMDecisionOutcome` is a
  complete one-shot orchestrator: renders the prompt, calls
  `call_with_fallback_chain`, adapts/validates via `adapt_decision`, retries
  with a correction message on validation failure, falls back to a
  caller-supplied deterministic closure on total failure. Works for all 4
  agent-class prompt templates (buyer/seller/investor/bank — verified
  structurally identical, parametric).
- `src/negotiation/llm_negotiation_engine.py`'s `run_llm_negotiation(buyer_id,
  seller_id, buyer_decide, seller_decide, max_rounds=10) -> NegotiationSession`
  is a real round-based negotiation loop, alternating buyer/seller turns via
  two injected `Callable[[NegotiationSession], NegotiationAction]` closures,
  terminating on ACCEPT/REJECT/WALK_AWAY or `max_rounds`. **Currently
  untested against a real or even mocked `decide()` call** — only exercised
  with hand-written fake decision callables. This plan is the first to wire
  it to `decide()`.
- `src/llm/hallucination_detector.detect_hallucination(...)` is a pure
  function, safe to call once per settled/attempted decision.

### 1.2 What must be built (confirmed gaps, not assumed)

- **A `Decision`/`NegotiationAction` → `Transaction` adapter does not exist
  anywhere.** `adapt_decision` stops at `NegotiationAction`
  (currency/chain/price/amount/reasoning) — turning a `NegotiationSession`'s
  final accepted offer into a settleable `Transaction` (needs `gas_fee`,
  matching a `CurrencyChainOption` from `generate_candidates`) is new code
  this plan must write.
- **`adapt_decision`'s currency/chain validity check is weaker than a true
  anti-hallucination guard**: it validates against caller-supplied
  `supported_currencies`/`supported_chains` sets, not automatically narrowed
  to the specific candidates offered in that prompt. This plan's wiring must
  explicitly pass `{c.currency_symbol for c in candidates}` /
  `{c.chain_name for c in candidates}` (not the full universe) at every call
  site, so a currency the LLM invents that happens to be valid elsewhere in
  the system but wasn't actually offered this round is correctly rejected.
- **`AgentDecisionContext.opponent_offer` is a declared but dead field** —
  never rendered into any prompt template. This plan's negotiation wiring
  must either start using it (add an `_format_opponent_offer_block` and wire
  it into all 4 templates) or continue relying on `conversation_history`
  (plain strings) for the LLM to see the other side's last offer — **decided
  here: use `conversation_history`, appending a formatted one-line summary
  of each round's `NegotiationAction` after it happens.** This avoids a
  4-template edit for a mechanism (`opponent_offer`) nothing else uses, and
  `conversation_history` already renders and is exercised by tests.
- **`httpx` is an optional dependency** (`pyproject.toml`'s `[project.optional-dependencies] llm`), not core. The LLM-driven day-loop path must be additive — `run_timestep` keeps a `use_llm: bool = False` parameter (default `False`, preserving every existing test's behavior and import-without-`httpx` guarantee); when `True`, the caller must have the `llm` extra installed and must supply a `ModelRosterConfig`-equivalent... **actually, per Plan 3, per-agent models come from `BaseAgent.assigned_model`, not a shared roster/policy** — `decide()`'s `policy_name`/`roster` parameters assume Phase 2's old shared-roster model. This plan's wiring calls `call_with_fallback_chain` (or `decide()`'s internals directly, bypassing its roster/policy resolution) with **exactly one model ID: `agent.assigned_model`**, with no fallback chain (Phase 3 assigns one fixed model per agent for the whole run, per the master spec §3.4 — a fallback chain would silently substitute a *different* agent's model identity, corrupting the per-agent model trait). Confirm at implementation time whether to call `decide()` with a single-element `model_ids` list or bypass it for a thinner direct `call_model` wrapper — **Resolution (not a stakes decision): bypass `decide()`'s roster/policy resolution and call a new thin wrapper that does render→`call_model(single model_id)`→`adapt_decision`→retry-correction, since `decide()`'s `policy_name` concept doesn't fit Phase 3's per-agent-fixed-model design and forcing it through would be more confusing than a small new function.**

### 1.3 Cost/safety guardrail (new, since none exists in `src/llm/` today)

No `dry_run`/rate-limit/cost-estimation code exists anywhere in `src/llm/`
today — the only existing guardrail is `RUN_LIVE_LLM_TESTS`/`@pytest.mark.live`
at the pytest-collection level. This plan adds a `dry_run: bool` parameter
threaded through the new day-loop LLM wiring and the matrix runner (Sec 10):
when `True` (the default for every test and for any non-full-scale
invocation), the LLM call layer must be given a mock/fake `httpx.Client`
(matching the existing per-test `MockTransport` convention — no shared
fixture exists yet; this plan adds one reusable test helper rather than
hand-rolling it in every new test file, unlike the rest of the codebase's
convention so far, since this plan adds many new LLM-calling tests). The
matrix runner (Sec 10) refuses to construct a real `httpx.Client` unless
`dry_run=False` is passed explicitly by its caller — never a config default.

---

## 2. Live Polygon price wiring

**User decision:** wire in live Polygon.io price fetching, in addition to
the already-wired static currency-profile corpus.

`src/llm/market_intelligence.py`'s `fetch_live_price(ticker, client) ->
LivePriceSnapshot` and `build_polygon_client(api_key, transport=None)` are
fully built and unit-tested in isolation but never called by production
code. This plan adds the orchestration: once per simulated day (not once per
agent-decision, to avoid redundant API calls for the same tickers), fetch a
live price snapshot for every tradable currency's underlying reference
asset, build the `live_price_snapshots: dict[str, LivePriceSnapshot]` dict,
and pass it into every `build_decision_context` call that day. Uses the
`.env.example`'s `Polygon_API_KEY` — **flagging the casing inconsistency
found during research: `.env.example` has `Polygon_API_KEY` but no code
anywhere reads it (case-sensitive `os.getenv`), so this plan's new code must
pick and use one exact casing consistently; recommend matching
`OPENROUTER_API_KEY`'s all-caps convention as `POLYGON_API_KEY` and updating
`.env.example` to match, flagged here since it touches a file outside this
plan's own new code.**

Same `dry_run` convention as Sec 1.3 applies — Polygon calls also go through
a mock client unless `dry_run=False`.

---

## 3. `CurrencyHistory`/`MacroHistory` auto-population

**Resolution (not a stakes decision):** this is finishing Plan 2's own
explicitly-stated hand-off ("Plan 3/4 is expected to construct
CurrencyHistory/MacroHistory from [TrustLedger.history()] and instantiate
EventLog for real runs" — Plan 2's final review comment, and Plan 2's design
spec §3.4), not a new modeling choice.

- `Environment` gains `self.event_log = EventLog()` (constructed alongside
  `self.trust_ledger` in `__init__`).
- `run_timestep` records every `due_shock` into `env.event_log` (one line,
  `env.event_log.record(shock)` per due shock) — this was explicitly
  deferred by Plan 2's review as an "unconsumed pipe," now closed.
- New helper (module TBD by the implementation plan — likely
  `src/economy/history_builder.py` or a function inside `agent_reasoning.py`
  itself): `build_currency_history(ledger: TrustLedger, event_log: EventLog,
  symbol: str, day: int) -> CurrencyHistory` and `build_macro_history(env:
  Environment, day: int) -> MacroHistory`, computing `trust_now`/
  `trust_30d_ago`/`trust_min_90d`/`trend`/`depeg_events_90d`/
  `last_event_days_ago`/`recent_events` from `TrustLedger.history()` +
  `EventLog.all_events()` filtered by day window, per Plan 2's design spec
  §3.4's exact field definitions (already specified there — this plan
  implements, not designs, those fields).

---

## 4. Cross-border FX conversion tax

**User decision (after researching real-world benchmarks):** rate = **0.02%
(2 basis points)**, matching the sourced institutional native EURC/USDC
conversion cost. **User decision:** trigger condition = tax applies when the
settlement currency's zone differs from the buyer's `currency_zone`;
gold-backed currencies (`peg == "XAU"`) are zone-neutral and never trigger
the tax.

- New config `configs/economy/fx_params.yaml`: `fx_tax_rate: 0.0002` (the
  project's "no hardcoded economic constants" rule — this lives in YAML,
  never hardcoded in `src/`).
- Zone mapping derived from `CurrencyConfig.peg`: `"USD" -> "USD"`, `"EUR" ->
  "EUR"`, `"XAU" -> None` (zone-neutral, per the confirmed trigger rule).
- Applied in `timestep.py`'s transaction-building step (or a new
  `src/transactions/fx_tax.py` helper, task breakdown decides): compute
  `fx_tax_paid = agreed_price * fx_tax_rate if zone_of(chosen.currency_symbol)
  not in (None, buyer.currency_zone) else 0.0`, set on the `Transaction`
  before `settle()` — and `settle()`/`validate_transaction` must actually
  debit `fx_tax_paid` from the buyer's wallet balance (confirmed gap: today
  `settle()` does a pure same-currency debit/credit with zero tax logic —
  this plan adds the debit, on top of the existing `fx_tax_paid` column
  that's persisted but never populated).

---

## 5. Loss-driven CARA-coefficient adaptation

**User decision:** build a real price-index tracker (not a nominal-wealth
stand-in). **User decision:** `eta_risk = 1.0`, `a_max = 5.0`, as a
documented starting point in a new config file.

### 5.1 Price index (new, built from scratch — confirmed nothing like this exists)

- New small stateful piece, owned by `Environment`: `self.price_index: float
  = 1.0` at construction, updated once per day in `run_timestep` (after
  shocks apply, using that day's `env.macro_state.inflation`):
  `env.price_index *= (1 + env.macro_state.inflation)`. Since
  `MacroState.inflation` is an additive-rate field shocks bump (not itself a
  compounding level), this plan's price index is the new compounding series
  the master spec's `I_price,t` denominator requires.
- `W_real_t` for agent `i`: `agent.wallet.total_value_usd(env.exchange_rates)
  / env.price_index`.

### 5.2 Adaptation formula (exact, from master spec §3.3E, `a` substituted for `σ`)

Only applies to the **55 CARA-eligible agents** (those with
`cara_coefficient is not None`, i.e. consumer/bank/investor per Plan 3's
resolved ambiguity — merchant/institution `multi_attribute` agents have no
`a` to adapt).

```
Loss_t   = max(0.0, W_real_{t-1} - W_real_t)     # only realized decreases count
a_next   = min(a_max, a_t + eta_risk * Loss_t / W_real_t)
```

- New config `configs/economy/risk_adaptation_params.yaml`: `eta_risk: 1.0`,
  `a_max: 5.0`.
- Computed once per day per CARA-eligible agent, after that day's
  transactions settle (so `W_real_t` reflects the day's actual outcome).
- **`a` only ever increases or holds (never decreases)** — this is the
  literal formula, not a simplification; `Loss_t` is floored at 0, so gains
  never reduce `a`. This matches the master spec's own framing ("realized
  losses... increase an agent's a over time") and is not this plan's
  invention.
- Since `a` can ratchet from a negative starting value up through exactly
  `0.0` and beyond into positive territory over the course of a run, the
  same `a == 0.0 -> risk_neutral` branch Plan 3 built into
  `agent_factory.build_agent`'s `cara_override` must be reapplied **every
  time `a` changes**, not just at population-construction time — `agent.
  utility_fn` must be rebuilt via `build_utility_function` whenever `a`
  crosses into or out of exactly `0.0` (an exact-zero landing is
  measure-zero-probability in continuous terms but must still be handled
  defensively, reusing Plan 3's existing branch logic rather than
  duplicating it — task breakdown should factor this into a shared helper
  both `population.py` and this plan's adaptation step call).
- Updates `agent.cara_coefficient` (nominal-turned-time-varying on the
  in-memory `BaseAgent` — no schema conflict, since `AgentRecord.
  cara_coefficient` remains the one-time initial snapshot and
  `AgentStateRecord.cara_coefficient`, already nullable per Plan 3's final
  review fix, is where each day's adapted value is persisted).

---

## 6. Environment population support + sandbox currency universes

### 6.1 `Environment.build_from_population`

New classmethod alongside the existing `Environment.build` (which stays
completely unchanged for backward compatibility — confirmed its only
callers are `experiments/experiment_007..011` and existing tests, none of
which this plan touches):

```python
@classmethod
def build_from_population(
    cls,
    scenario_name: str,
    agents: list[BaseAgent],
    currencies: dict[str, CurrencyConfig] | None = None,  # None = full real universe
    goods: list[Good] | None = None,
) -> "Environment":
```

When `currencies` is `None`, behaves exactly like `Environment.build`'s
`load_currency_universe()` call (the master simulation and its cross-border
repeat use the full real 9-currency universe). When supplied, it's used
as-is — this is the hook the 6 factor-isolation sandboxes use (Sec 6.2).

### 6.2 Synthetic sandbox currency pairs

**User decision:** build synthetic per-sandbox `CurrencyConfig` pairs
matching Experiment.md §5B's exact numeric specs, rather than filtering the
real 9-currency universe to the closest analog pair (confirmed: the real
currencies are internally consistent — better governance correlates with
better peg stability throughout the real dataset — so several of
Experiment.md's rows describe a deliberate tradeoff, e.g. "compliant coin,
PegError=0.02 vs non-compliant, PegError=0.00," that no real currency pair
exhibits).

**User decision (confirmed by cross-referencing two of the user's own source
documents):** there are **6** factor-isolation sandboxes, not 7.
`Untitled document.md`'s own numbered list ("Isolate liquidity v governance;
Governance v stability; Liquidity v stability; Asset backing v liquidity;
Asset backing v stability; Asset backing v governance") gives exactly 6
items, stated twice (once for domestic, once for "Cross Border repeat
analysis"). The master design spec's "7 factor-isolation sandboxes" count
was taking Experiment.md §5B's table literally, whose 7th row is the Privacy
sandbox (H6) — but H6 is explicitly deferred elsewhere in that same master
spec (§7). This plan builds exactly the 6 sandboxes both source documents
agree on, each run once domestically and once cross-border (12 sandbox
cells total) + the master simulation = **13 experiment cells**, not 14 or 15.

Six synthetic currency pairs, each isolating exactly one dimension while
holding the others constant at a shared neutral value (task breakdown
constructs these as `CurrencyConfig` instances directly — no new YAML
loading convention needed, since these are sandbox-scoped, not part of the
persisted real universe; exact `CurrencyConfig` field names to be confirmed
against `src/currencies/currency.py` at implementation time):

| Sandbox | Dimension isolated | Option A | Option B |
|---|---|---|---|
| 1. Liquidity vs. Governance | governance_score, liquidity_score | liquidity=0.99, governance=0.55, non-compliant | liquidity=0.90, governance=0.95, compliant |
| 2. Governance vs. Stability | governance_score, peg_error | governance=0.95, compliant, peg_error=0.02 | governance=0.55, non-compliant, peg_error=0.0001 |
| 3. Liquidity vs. Stability | liquidity_score, peg_error | liquidity=0.99, peg_error=0.04 | liquidity=0.75, peg_error=0.0001 |
| 4. Asset Backing vs. Liquidity | asset_class, liquidity_score | gold_backed, liquidity=0.70 | stablecoin, liquidity=0.99 |
| 5. Asset Backing vs. Stability | asset_class, peg_error | gold_backed, peg_error=0.015 | tokenized_deposit, peg_error=0.0001 |
| 6. Asset Backing vs. Governance | asset_class, issuer_risk/governance | tokenized_deposit (bank credit risk), governance=0.75, issuer_risk=0.25 | stablecoin (algorithmic/decentralized), governance=0.70, issuer_risk=0.20 |

All other `CurrencyConfig` fields (whatever the schema requires beyond these
— confirmed at implementation time) held at the same neutral value across
both options in a given sandbox, so only the named dimension differs.
**These exact numbers are a first-draft proposal for the user's review, not
a final locked table** — flagged here explicitly since, unlike the FX
tax/adaptation constants, these weren't individually walked through
question-by-question the way those were; the user should treat this table
as part of what they're reviewing in this spec, not as already-confirmed.

Each sandbox cell restricts `Environment.build_from_population`'s
`currencies` to exactly that sandbox's 2-entry dict.

---

## 7. Persistence wiring

Extends (does not replace) `persist_timestep`, or a new `persist_full_timestep`
(task breakdown decides which is cleaner) called once per day per run:

- **`SimulationRunRepository.record(...)`** — once per run, before day 0,
  using Sec 8's provenance helpers.
- **`TimestepLogRepository.record(...)`** — once per day, from
  `env.macro_state` (`inflation_rate=macro_state.inflation`,
  `confidence_index`, gas fees from chain configs, `eur_usd_exchange_rate`
  from `macro_state.peg_reference_rates`).
- **`AgentStateRepository.record(...)`** — once per agent per day:
  `real_purchasing_power = wallet.total_value_usd(rates) / env.price_index`
  (Sec 5.1's index), `wallet_balances = dict(agent.wallet.balances)`,
  `cara_coefficient = agent.cara_coefficient` (post-adaptation value, `None`
  for multi_attribute agents). **`utility_score` resolution (not a stakes
  decision, flagged for review):** `agent.utility_fn.evaluate()` applied to
  the agent's own realized wealth via a neutral identity candidate
  (`safety_multiplier=1.0`, `gas_fee=0.0`) — i.e. the utility of the agent's
  actual end-of-day real purchasing power, not tied to any specific
  transaction. This is descriptive per-day telemetry, not a hypothesis
  treatment variable, so a reasonable default is used here rather than
  raising it as its own question — override on review if a different
  formula is wanted.
- **`InterventionLogRepository.record(...)`** — once per `result.
  fired_shocks` entry.
- **`AgentMemoryLogRepository.record(...)`** — once per `result.
  memory_events` tuple (this pipe already exists from Plan 2, just never
  called — see Plan 2's design spec §3.5).
- **`LLMDecisionRepository.record(...)`** / **`HallucinationRepository.
  record(...)`** — once per LLM decision (Sec 1), using
  `LLMDecisionLogEntry`'s existing 26-field shape (already includes
  `domestic_or_cross_border`/`governance_prompt_enabled`, confirmed
  pre-shaped for this by Plan 1) and `detect_hallucination`'s result.
- **`AgentRepository.upsert_agent(...)`** — once per agent, at run start
  (confirmed existing behavior: only sets fields on first insert, a
  pre-existing limitation flagged during Plan 3's review, not something
  this plan needs to fix since agent identity/model/zone/nominal-`a` never
  change mid-run — only `AgentStateRecord`'s per-day row needs updating,
  and it already gets a fresh row every day by design).

---

## 8. Provenance helpers (new, built from scratch)

New module `src/simulation/provenance.py`:

- `compute_git_commit_hash() -> str` — `subprocess.run(["git", "rev-parse",
  "HEAD"], ...)`, no new dependency (git is already a hard requirement of
  the dev environment; no GitPython in `pyproject.toml`).
- `compute_config_hash(paths: list[Path]) -> str` — SHA-256 over the
  concatenated bytes of every resolved YAML config file involved in a given
  run (scenario file + currency universe or sandbox pair + chain universe +
  trust params + fx params + risk adaptation params + agent profiles),
  matching the master spec §9's "SHA-256 of resolved YAML configs" wording.
  **Exact file set per run type (master vs. sandbox) is defined by the
  matrix runner (Sec 10), since it's the only place that knows which
  configs a given cell actually used.**
- `model_roster_summary_for(agents: list[BaseAgent]) -> str` — per Plan 1's
  already-documented deviation ("100 agents across N models, see
  agent_states"), a short descriptor, e.g. `f"{len(agents)} agents across
  {len({a.assigned_model for a in agents})} OpenRouter models"`.

---

## 9. Master 365-day scenario + H4 proximity sweep

New `configs/scenarios/master_simulation.yaml`: `duration_days: 365`,
`initial_state` matching `baseline.yaml`'s pattern, and a `shocks: [...]`
list scheduling:
- Multiple `crisis_warning` → `depeg_event` pairs at **0, 5, 10, 20-day
  gaps** (H4's proximity variable, per master spec §4 — each pair targeting
  a different currency so they don't confound each other), spread across
  the year.
- At least one instance of each of the other 10 shock types (inflation,
  bank_failure, gold_rally, fee_spike, regulatory_enforcement,
  liquidity_crunch, governance_downgrade, fx_volatility_shock,
  fx_rate_shock, capital_controls), scheduled at distinct days so effects
  don't confound.

**Exact day-numbers/targets/magnitudes are the implementation task's job to
draft against Plan 2's `ShockEvent`/`ShockType` interfaces — this spec fixes
the requirements (0/5/10/20-day H4 gaps, all 12 shock types represented,
non-confounding spacing), not the literal YAML, since that's better
reviewed as an artifact once drafted than speculated in prose here.**

The 6 sandbox cells (domestic + cross-border) reuse this same shock schedule
(only the currency universe differs per Sec 6.2) — **Resolution (not a
stakes decision): reusing one shock schedule across all 13 cells keeps every
cell's macro/shock conditions identical, isolating the currency-universe
restriction as the only difference between cells, which is what makes a
sandbox a valid factor-isolation cell rather than a second confounded
variable.**

---

## 10. The matrix runner

New `src/simulation/matrix_runner.py`:

```python
def run_matrix(
    model_candidates: list[str],
    seeds: list[int],
    dry_run: bool = True,
    openrouter_client: httpx.Client | None = None,
    polygon_client: httpx.Client | None = None,
) -> list[SimulationResult]:
```

- Iterates the 13 cells (master + 6 domestic sandboxes + 6 cross-border
  sandboxes) × `seeds` (5 for the real run, task breakdown's tests use 1-2
  fake seeds over a tiny `num_days`).
- Per cell/seed: `verify_model_candidates` (once, cached across the whole
  matrix run, not per cell) → `generate_agent_population(seed,
  available_models)` → `Environment.build_from_population(...)` (full
  universe for master+cross-border-repeat-of-master, sandbox pair for the
  12 sandbox cells) → cross-border pairing forced for the 6 cross-border
  cells (**mechanism: task breakdown's job to define exactly how buyer/seller
  matching is forced by `currency_zone` in the marketplace-listing step,
  since `timestep.py`'s current listing/matching logic has no zone-awareness
  at all today** — flagged as an implementation-time design detail, not
  fully specified here) → day loop calling the LLM-wired `run_timestep`
  (Sec 1) for `scenario.duration_days` days → full persistence (Sec 7) →
  provenance (Sec 8) recorded before day 0.
- `dry_run=True` (default): both clients are mock-transport fakes
  constructed internally if not supplied; **refuses to proceed with
  `dry_run=False` unless the caller explicitly supplies both a real
  `openrouter_client` and `polygon_client`** — this is the code-level half of
  the master spec's "explicit second confirmation before the run that
  incurs spend" gate; the actual launch decision remains a conversation with
  the user, not something this function's default enables.

---

## 11. What Plan 5 inherits from this plan

Full per-timestep, per-agent, per-decision raw data in `simulation_runs`,
`timestep_logs`, `agent_states`, `agent_memory_logs`, `intervention_logs`,
`llm_decisions`, `hallucinations`, `transactions`, `negotiations` — the
complete raw dataset the econometrics engine (H1-H5 regressions) reads from.
No further data-collection code should be needed; Plan 5 is pure analysis
over what this plan persists.
