# Phase 3 Plan 2: Shock Engine, Trust Ledger & Historical Context — Design Spec

**Status:** Written under the user's blanket authorization to proceed through
the remaining Phase 3 plans without stopping for per-plan approval (the
master spec, `docs/superpowers/specs/2026-07-29-phase3-full-scale-simulation-design.md`,
is already approved). Self-reviewed below; no interactive Q&A round for this
plan.

**Source:** `Untitled document.md` (pasted into the original conversation),
which is explicitly a "design spec, not yet implemented" for the shock/trust/
history extension. This plan promotes it to implemented, resolving a few
mechanism-level ambiguities the source document left open (documented
in §3 below) using the same judgment standard as Plan 1's documented
deviations.

## 1. Scope

Builds `src/economy/trust.py` (dynamic per-currency trust/offset ledger),
extends `src/economy/shocks.py` with 8 new shock types (4 exist today:
`inflation`, `bank_failure`, `gold_rally`, `fee_spike`), adds
`src/economy/event_log.py`, and extends `src/llm/agent_reasoning.py`'s
`AgentDecisionContext`/prompt rendering with `CurrencyHistory`/`MacroHistory`
sections. Wires shock events into Plan 1's `InterventionLogRepository`, and
notable agent memory events into Plan 1's `AgentMemoryLogRepository`.

Explicitly NOT in this plan's scope: authoring the actual 365-day scenario
YAML with the H4 proximity-sweep shock schedule (0/5/10/20-day warning
gaps) — that's an experiment-configuration decision that belongs to Plan 4
(matrix runner), which is the plan that owns scenario authoring for the
master simulation and all sandboxes. This plan only builds the *mechanism*
(shock types + their effects); Plan 4 decides *when* they fire in the
365-day run.

## 2. Two Distinct Kinds of Shock Effect

Re-reading `Untitled document.md` §1.2 closely, its 8 new shock types split
into two mechanically different categories, and conflating them would be a
design error:

**(a) Permanent structural change** — mutates a currency's baseline
attributes in `CurrencyConfig` directly, the same way `bank_failure`
already permanently steps `macro_state.confidence_index` down (no decay
back). Applies to:
- `governance_downgrade` → permanently lowers `governance_score` (no
  "temporarily" language in the source doc, unlike its siblings below —
  a reserve-audit failure or transparency scandal is a lasting reputational
  fact, not a transient wobble).
- `regulatory_enforcement`'s `issuer_risk` spike (the fine/enforcement
  action itself is a permanent record), though its accompanying liquidity
  effect is temporary (see (b)).

**(b) Temporary, decaying effect** — the source doc explicitly says
"temporarily drops" / "decaying back to baseline over a configurable
number of days" for these, in contrast to `bank_failure`'s permanence.
These need a live decay mechanism, not a one-shot mutation:
- `depeg_event` → `peg_error` spikes, decays back to the currency's
  baseline `peg_error` over N days.
- `liquidity_crunch` → `liquidity_score` drops temporarily.
- `regulatory_enforcement`'s liquidity-friction component (frozen
  redemptions create real friction "not just reputational risk").
- `fx_volatility_shock` → `peg_error` spikes for all EUR-pegged currencies.
- `capital_controls` → effective liquidity/friction penalty in
  cross-border transactions specifically.

`crisis_warning` and `fx_rate_shock` are macro-level (small
`confidence_index` dip; `macro_state.peg_reference_rates["EUR"]` shock
respectively) and don't need per-currency decay machinery at all — they
reuse the existing `MacroState`/`apply_shock` pattern directly.

## 3. Mechanism Design (resolving the source doc's open questions)

### 3.1 Permanent effects: extend `apply_shock`'s existing pattern

`apply_shock(state: MacroState, shock: ShockEvent) -> MacroState` already
returns a `model_copy(deep=True)` with fields changed. This plan adds a
parallel `apply_currency_shock(currencies: dict[str, CurrencyConfig], shock:
ShockEvent) -> dict[str, CurrencyConfig]` for the shocks that mutate a
currency's own baseline attributes permanently (`governance_downgrade`,
`regulatory_enforcement`'s `issuer_risk` component). It returns a new dict
with only the targeted currency's `CurrencyConfig` replaced via
`model_copy(update={...})` after clamping the new value into its field's
valid range with `src/utils/helpers.py`'s existing `clamp()` — `model_copy`
skips validators, so clamping first is required to keep e.g.
`governance_score` inside `[0, 1]`.

### 3.2 Temporary effects: generalize `TrustLedger` to a shock-offset ledger

Rather than inventing a second, separate decay mechanism, `TrustLedger`
(`src/economy/trust.py`) tracks **two independent decaying channels per
currency**, using the identical asymmetric shock/recover formula from
`Untitled document.md` §2.2 for both:

```python
class TrustLedger:
    def __init__(self, currencies: dict[str, CurrencyConfig], params: TrustParams): ...

    def trust_score(self, symbol: str) -> float: ...          # τ_c(t), baseline = governance_score
    def peg_error_offset(self, symbol: str) -> float: ...     # decays toward 0.0
    def liquidity_offset(self, symbol: str) -> float: ...     # decays toward 0.0

    def effective_peg_error(self, symbol: str, baseline: float) -> float:
        return max(0.0, baseline + self.peg_error_offset(symbol))

    def effective_liquidity_score(self, symbol: str, baseline: float) -> float:
        return clamp(baseline + self.liquidity_offset(symbol), 0.0, 1.0)

    def update(self, day: int, fired_shocks: list[ShockEvent], asset_class_of: dict[str, str]) -> None: ...
    def history(self, symbol: str, days: int) -> list[float]: ...  # trust_score time series, for CurrencyHistory
```

- `trust_score` always exists per currency (every currency has ongoing
  reputational dynamics, shocked or not) and is what `CurrencyHistory`'s
  `trust_now`/`trend`/`stdev`-based perceived-trust math (§3.4) reads from.
- `peg_error_offset`/`liquidity_offset` are the decay state for `(b)`-type
  shocks: `depeg_event`/`fx_volatility_shock` write to `peg_error_offset`;
  `liquidity_crunch`/`regulatory_enforcement`/`capital_controls` write to
  `liquidity_offset`. A quiet currency's offsets are `0.0` (no effect).
- All three channels share the same `λ_shock`/`λ_recover`/`λ_contagion`
  constants from `configs/economy/trust_params.yaml` — the source doc's own
  framing ("trust crashes fast, rebuilds slowly") is exactly the same
  economic intuition as "a depeg spike fades over days," so one constant
  set, one update loop, three tracked quantities is more honest to the
  actual mechanism than three separate config files would be.
- Callers needing a currency's *current effective* peg_error/liquidity_score
  call `trust_ledger.effective_peg_error(symbol, currency_config.peg_error)`
  rather than reading `CurrencyConfig.peg_error` directly — this is the one
  call-site change needed in `src/blockchain/routing_engine.py`'s
  `generate_candidates` and `src/llm/agent_reasoning.py`'s candidate
  formatting.

### 3.3 New ShockType additions

```python
class ShockType(str, Enum):
    INFLATION = "inflation"              # existing
    BANK_FAILURE = "bank_failure"        # existing, extended with target_issuer
    GOLD_RALLY = "gold_rally"            # existing
    FEE_SPIKE = "fee_spike"              # existing
    REGULATORY_ENFORCEMENT = "regulatory_enforcement"
    LIQUIDITY_CRUNCH = "liquidity_crunch"
    GOVERNANCE_DOWNGRADE = "governance_downgrade"
    DEPEG_EVENT = "depeg_event"
    CRISIS_WARNING = "crisis_warning"
    FX_VOLATILITY_SHOCK = "fx_volatility_shock"
    FX_RATE_SHOCK = "fx_rate_shock"
    CAPITAL_CONTROLS = "capital_controls"


class ShockEvent(BaseModel):
    day: int
    type: ShockType
    magnitude: float
    target_currency: str | None = None   # new, optional
    target_issuer: str | None = None     # new, optional
    decay_days: int | None = None        # new, optional -- only meaningful for (b)-type shocks
```

`apply_shock` (macro-level effects) and the new `apply_currency_shock`
(permanent currency-level effects) both get called from `timestep.py`'s
daily loop; `TrustLedger.update()` is called once per day regardless of
whether a shock fired (to advance decay/recovery).

### 3.4 CurrencyHistory / MacroHistory prompt extension

Exactly as `Untitled document.md` §3.2 specifies, with one clarification:
`perceived_trust_i,c(t) = τ_c(t) - r_i · stdev(τ_c, last W days)` uses the
**CARA coefficient `a`** (per the 2026-07-29 correction to the master
design spec) as `r_i` — a more risk-averse agent (`a` more positive)
penalizes recent trust volatility more heavily, exactly the source doc's
intent, just with the corrected parameter identity. This computation lives
in `src/utility/` (reads an agent's `risk_aversion` field, which is `a` for
CARA agents) rather than in `TrustLedger` itself, since `TrustLedger` has
no notion of individual agents — it exposes `history()` and `stdev()`
helpers; the per-agent perception calculation is the caller's job
(`src/llm/agent_reasoning.py`, building each agent's prompt context).

### 3.5 Event log → Plan 1's `intervention_logs`

`src/economy/event_log.py`'s append-only log is an in-memory list during a
run (`EventLog.record(day, shock_type, target, severity)`), and
`timestep.py`'s daily loop calls `InterventionLogRepository.record(...)`
(Plan 1) directly whenever a shock fires — no separate persistence step
needed since Plan 1 already built exactly this table/repository.

### 3.6 Agent narrative memory → Plan 1's `agent_memory_logs`

`src/agents/memory.py`'s `AgentMemory` (currently just per-currency
success/fail counts) gets one new field:

```python
class AgentMemory(BaseModel):
    outcomes: dict[str, dict[str, int]] = Field(default_factory=dict)
    narrative_events: list[str] = Field(default_factory=list)  # new, capped at last 10

    def record_narrative(self, event_text: str, max_events: int = 10) -> None:
        self.narrative_events.append(event_text)
        if len(self.narrative_events) > max_events:
            self.narrative_events = self.narrative_events[-max_events:]
```

`timestep.py` calls `agent.memory.record_narrative(...)` and
`AgentMemoryLogRepository.record(...)` (Plan 1) together whenever a shock
notably affects an agent's held currency (e.g. the agent was holding a
currency that just got hit by `depeg_event`) — both the in-memory rolling
list (for the next prompt) and the durable per-run log (for post-hoc
analysis) get the same event text.

## 4. File Structure

- **Create:** `src/economy/trust.py` (`TrustLedger`, `TrustParams`)
- **Create:** `configs/economy/trust_params.yaml` (`lambda_shock`,
  `lambda_recover`, `lambda_contagion`, `rolling_window_days`)
- **Create:** `src/economy/event_log.py` (`EventLog`, thin wrapper feeding
  `InterventionLogRepository`)
- **Modify:** `src/economy/shocks.py` (8 new `ShockType` members, extended
  `ShockEvent` fields, new `apply_currency_shock()`, extended
  `apply_shock()` for `crisis_warning`/`fx_rate_shock`)
- **Modify:** `src/economy/macro_state.py` (no field changes expected —
  `peg_reference_rates` already supports `fx_rate_shock`'s target)
- **Modify:** `src/blockchain/routing_engine.py` (`generate_candidates`
  takes an optional `TrustLedger` to read effective peg_error/liquidity)
- **Modify:** `src/llm/agent_reasoning.py` (`CurrencyHistory`,
  `MacroHistory` models; extended `AgentDecisionContext`; new
  `history_block` prompt field)
- **Modify:** `src/llm/prompts/{buyer,seller,investor,bank}_prompt.txt`
  (add a `# History` section rendering `{history_block}`)
- **Modify:** `src/agents/memory.py` (`narrative_events` field)
- **Modify:** `src/simulation/timestep.py` (wire shock application, event
  log persistence, narrative memory persistence into the daily loop)
- **Test:** new `tests/test_trust_ledger.py`, new `tests/test_shocks_extended.py`,
  new `tests/test_agent_reasoning_history.py`, extend `tests/test_agents.py`

## 5. Global Constraints (carried from the master spec)

- No hardcoded economic constants — `λ_shock`/`λ_recover`/`λ_contagion`/
  rolling window live in `configs/economy/trust_params.yaml`, never
  hardcoded in `trust.py`.
- Follow Plan 1's repository pattern exactly when writing to
  `intervention_logs`/`agent_memory_logs` — never construct those ORM
  records directly from `src/economy/` or `src/simulation/`.
- Python >=3.12, Pydantic >=2.6, SQLAlchemy >=2.0 — no new dependencies.
- This is the final data-collection phase — no scope beyond what this doc
  and the master spec describe.
