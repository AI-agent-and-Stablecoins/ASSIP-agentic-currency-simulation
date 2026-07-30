# Phase 3 Plan 2: Shock Engine, Trust Ledger & Historical Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dynamic shock/trust/history mechanism specified in `docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md`: 8 new shock types, a `TrustLedger` tracking per-currency trust and temporary peg/liquidity offsets, an event log, and `CurrencyHistory`/`MacroHistory` prompt context — so agents can reason about trajectory ("USDT wobbled twice recently"), not just point-in-time state.

**Architecture:** `TrustLedger` (`src/economy/trust.py`) is the single source of dynamic per-currency state, updated once per simulated day from that day's fired shocks. Permanent shock effects (`governance_downgrade`, `regulatory_enforcement`'s issuer-risk component) mutate `CurrencyConfig` copies directly via a new `apply_currency_shock()`, mirroring the existing `apply_shock()` pattern for `MacroState`. `timestep.py` stays database-free — shock/memory events surface as new `TimestepResult` fields for a later plan to persist.

**Tech Stack:** Pydantic >=2.6 (all new models), no new dependencies.

## Global Constraints

- Python >=3.12, Pydantic >=2.6, SQLAlchemy >=2.0 — no new dependencies without checking with the user first.
- No hardcoded economic constants: `lambda_shock`, `lambda_recover`, `lambda_contagion`, `rolling_window_days` live in `configs/economy/trust_params.yaml`, never hardcoded in `src/economy/trust.py`.
- This is the final data-collection phase per the approved master spec (`docs/superpowers/specs/2026-07-29-phase3-full-scale-simulation-design.md`) and this plan's own design spec (`docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md`) — do not add scope beyond what those two documents describe without checking with the user first.
- `timestep.py` must remain free of any database/session dependency — new shock/memory data surfaces via `TimestepResult` fields only; actual persistence is explicitly Plan 4's job, not this plan's.
- Follow existing patterns: Pydantic models for all config/state, `load_yaml_as`/`CONFIG_ROOT` for config loading (`src/utils/config_loader.py`, `src/utils/constants.py`), `clamp()` for range-safety (`src/utils/helpers.py`).
- The risk-aversion parameter referenced anywhere in this plan (`r_i` in the perceived-trust formula) is the **CARA coefficient `a`**, per the 2026-07-29 correction in the master design spec — never CRRA σ.
- **Task order matters in this plan**: `ShockType`'s new enum members (Task 1) must exist before `TrustLedger` (Task 3) can reference them in tests. Execute tasks strictly in the order below, not the order that might seem more "foundational."

---

## File Structure

- **Create:** `configs/economy/trust_params.yaml`
- **Create:** `src/economy/trust.py` (`TrustParams`, `load_trust_params`, `TrustLedger`)
- **Create:** `src/economy/event_log.py` (`EventLog`)
- **Modify:** `src/economy/shocks.py` (8 new `ShockType` members, extended `ShockEvent` fields, `apply_currency_shock()`, extended `apply_shock()` for `crisis_warning`/`fx_rate_shock`, `bank_failure` gains optional `target_issuer`)
- **Modify:** `src/agents/memory.py` (`AgentMemory.narrative_events`, `record_narrative()`)
- **Modify:** `src/simulation/timestep.py` (`TimestepResult.fired_shocks`/`memory_events` fields; wire shock application into the daily loop)
- **Modify:** `src/blockchain/routing_engine.py` (`generate_candidates` takes an optional `TrustLedger` to read effective peg_error/liquidity_score)
- **Modify:** `src/llm/agent_reasoning.py` (`CurrencyHistory`, `MacroHistory` models; extended `AgentDecisionContext`; new `history_block` prompt field; `build_decision_context` gains optional history params)
- **Modify:** `src/llm/prompts/{buyer,seller,investor,bank}_prompt.txt` (new `# History` section)
- **Test:** new `tests/test_shocks_extended.py`, new `tests/test_trust_ledger.py`, new `tests/test_event_log.py`, extend `tests/test_agents.py`, extend `tests/test_simulation.py`, extend `tests/test_agent_reasoning.py`, extend a routing-engine test file (check for an existing one covering `generate_candidates` first; if none exists, create `tests/test_routing_engine.py`)

---

### Task 1: Extend ShockType, ShockEvent, and add apply_currency_shock

**Files:**
- Modify: `src/economy/shocks.py`
- Test: new `tests/test_shocks_extended.py`

**Interfaces:**
- Produces: 8 new `ShockType` members, `ShockEvent.target_currency: str | None`, `ShockEvent.target_issuer: str | None`, `ShockEvent.decay_days: int | None`, `apply_currency_shock(currencies: dict[str, CurrencyConfig], shock: ShockEvent) -> dict[str, CurrencyConfig]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shocks_extended.py`:

```python
import pytest

from src.currencies.currency import load_currency_universe
from src.economy.shocks import ShockEvent, ShockType, apply_currency_shock


def test_shock_type_has_all_twelve_members():
    expected = {
        "inflation", "bank_failure", "gold_rally", "fee_spike",
        "regulatory_enforcement", "liquidity_crunch", "governance_downgrade",
        "depeg_event", "crisis_warning", "fx_volatility_shock", "fx_rate_shock",
        "capital_controls",
    }
    assert {member.value for member in ShockType} == expected


def test_shock_event_accepts_target_currency_and_issuer():
    shock = ShockEvent(day=5, type=ShockType.GOVERNANCE_DOWNGRADE, magnitude=0.2, target_currency="USDT")
    assert shock.target_currency == "USDT"
    assert shock.target_issuer is None
    assert shock.decay_days is None


def test_governance_downgrade_permanently_lowers_governance_score():
    currencies = load_currency_universe()
    original = currencies["USDT"].governance_score
    shock = ShockEvent(day=0, type=ShockType.GOVERNANCE_DOWNGRADE, magnitude=0.2, target_currency="USDT")

    updated = apply_currency_shock(currencies, shock)

    assert updated["USDT"].governance_score == pytest.approx(max(0.0, original - 0.2))
    assert updated["USDC"].governance_score == currencies["USDC"].governance_score  # untouched
    assert currencies["USDT"].governance_score == original  # original dict/config untouched


def test_governance_downgrade_clamps_at_zero():
    currencies = load_currency_universe()
    shock = ShockEvent(day=0, type=ShockType.GOVERNANCE_DOWNGRADE, magnitude=5.0, target_currency="USDT")

    updated = apply_currency_shock(currencies, shock)

    assert updated["USDT"].governance_score == 0.0


def test_regulatory_enforcement_spikes_issuer_risk_permanently():
    currencies = load_currency_universe()
    original = currencies["USDT"].issuer_risk
    shock = ShockEvent(day=0, type=ShockType.REGULATORY_ENFORCEMENT, magnitude=0.3, target_currency="USDT")

    updated = apply_currency_shock(currencies, shock)

    assert updated["USDT"].issuer_risk == pytest.approx(min(1.0, original + 0.3))


def test_apply_currency_shock_is_a_noop_for_non_currency_shocks():
    currencies = load_currency_universe()
    shock = ShockEvent(day=0, type=ShockType.INFLATION, magnitude=0.05)

    updated = apply_currency_shock(currencies, shock)

    assert updated == currencies
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shocks_extended.py -v`
Expected: FAIL with `AttributeError: GOVERNANCE_DOWNGRADE` (or similar — the new enum members don't exist yet).

- [ ] **Step 3: Extend ShockType and ShockEvent, add apply_currency_shock**

In `src/economy/shocks.py`, replace the `ShockType` class:

```python
class ShockType(str, Enum):
    INFLATION = "inflation"
    BANK_FAILURE = "bank_failure"
    GOLD_RALLY = "gold_rally"
    FEE_SPIKE = "fee_spike"
    REGULATORY_ENFORCEMENT = "regulatory_enforcement"
    LIQUIDITY_CRUNCH = "liquidity_crunch"
    GOVERNANCE_DOWNGRADE = "governance_downgrade"
    DEPEG_EVENT = "depeg_event"
    CRISIS_WARNING = "crisis_warning"
    FX_VOLATILITY_SHOCK = "fx_volatility_shock"
    FX_RATE_SHOCK = "fx_rate_shock"
    CAPITAL_CONTROLS = "capital_controls"
```

Replace the `ShockEvent` class:

```python
class ShockEvent(BaseModel):
    day: int
    type: ShockType
    magnitude: float
    target_currency: str | None = None
    target_issuer: str | None = None
    decay_days: int | None = None
```

Add `apply_currency_shock` after `apply_shock`, and add `from src.utils.helpers import clamp` to the imports:

```python
def apply_currency_shock(
    currencies: dict[str, CurrencyConfig], shock: ShockEvent
) -> dict[str, CurrencyConfig]:
    """Permanent, structural currency-attribute mutations only
    (governance_downgrade, regulatory_enforcement's issuer_risk component).
    Temporary/decaying effects (depeg_event, liquidity_crunch, ...) are
    handled by TrustLedger's offset channels (src/economy/trust.py), not
    here -- see docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md
    Sec 2-3 for why these are two different mechanisms."""
    if shock.target_currency is None or shock.target_currency not in currencies:
        return currencies

    updated = dict(currencies)
    target = updated[shock.target_currency]

    if shock.type == ShockType.GOVERNANCE_DOWNGRADE:
        new_score = clamp(target.governance_score - shock.magnitude, 0.0, 1.0)
        updated[shock.target_currency] = target.model_copy(update={"governance_score": new_score})
    elif shock.type == ShockType.REGULATORY_ENFORCEMENT:
        new_risk = clamp(target.issuer_risk + shock.magnitude, 0.0, 1.0)
        updated[shock.target_currency] = target.model_copy(update={"issuer_risk": new_risk})

    return updated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_shocks_extended.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/economy/shocks.py tests/test_shocks_extended.py
git commit -m "feat: add 8 new shock types and apply_currency_shock for permanent effects"
```

---

### Task 2: Extend apply_shock for crisis_warning, fx_rate_shock, and bank_failure's target_issuer

**Files:**
- Modify: `src/economy/shocks.py`
- Test: `tests/test_shocks_extended.py`

**Interfaces:**
- Produces: `apply_shock` now handles `ShockType.CRISIS_WARNING` and `ShockType.FX_RATE_SHOCK`; `ShockType.BANK_FAILURE` accepts an optional `target_issuer` (already added to `ShockEvent` in Task 1 — this task only adds behavior).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_shocks_extended.py`:

```python
from src.economy.macro_state import MacroState
from src.economy.shocks import apply_shock


def test_crisis_warning_applies_a_small_confidence_dip():
    state = MacroState(confidence_index=1.0)
    shock = ShockEvent(day=0, type=ShockType.CRISIS_WARNING, magnitude=0.05)

    updated = apply_shock(state, shock)

    assert updated.confidence_index == pytest.approx(0.95)
    assert state.confidence_index == 1.0  # original untouched


def test_fx_rate_shock_moves_eur_reference_rate():
    state = MacroState()
    original_eur = state.peg_reference_rates["EUR"]
    shock = ShockEvent(day=0, type=ShockType.FX_RATE_SHOCK, magnitude=0.1)

    updated = apply_shock(state, shock)

    assert updated.peg_reference_rates["EUR"] == pytest.approx(original_eur * 1.1)


def test_bank_failure_still_drops_confidence_with_optional_target_issuer():
    state = MacroState(confidence_index=1.0)
    shock = ShockEvent(day=0, type=ShockType.BANK_FAILURE, magnitude=0.3, target_issuer="Circle")

    updated = apply_shock(state, shock)

    assert updated.confidence_index == pytest.approx(0.7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shocks_extended.py -v`
Expected: FAIL — `apply_shock` doesn't handle `CRISIS_WARNING`/`FX_RATE_SHOCK` yet, so `updated.confidence_index`/`updated.peg_reference_rates["EUR"]` stay unchanged and the assertions fail.

- [ ] **Step 3: Extend apply_shock**

In `src/economy/shocks.py`, replace the `apply_shock` function body's `elif` chain (add two new branches; `BANK_FAILURE`'s existing branch needs no change since `target_issuer` is just an unused-here field on the input, already accepted by `ShockEvent`):

```python
def apply_shock(state: MacroState, shock: ShockEvent) -> MacroState:
    updated = state.model_copy(deep=True)
    if shock.type == ShockType.INFLATION:
        updated.inflation += shock.magnitude
    elif shock.type == ShockType.GOLD_RALLY:
        updated.gold_price *= 1 + shock.magnitude
        updated.peg_reference_rates["XAU"] = updated.gold_price
    elif shock.type == ShockType.BANK_FAILURE:
        updated.confidence_index = max(0.0, updated.confidence_index - shock.magnitude)
    elif shock.type == ShockType.FEE_SPIKE:
        # Fee spikes mutate blockchain gas_fee configs directly (src/blockchain),
        # not macro state -- the caller applies this shock at that layer.
        pass
    elif shock.type == ShockType.CRISIS_WARNING:
        updated.confidence_index = max(0.0, updated.confidence_index - shock.magnitude)
    elif shock.type == ShockType.FX_RATE_SHOCK:
        updated.peg_reference_rates["EUR"] = updated.peg_reference_rates["EUR"] * (1 + shock.magnitude)
    return updated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_shocks_extended.py -v`
Expected: PASS (all 9 tests in the file).

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/economy/shocks.py tests/test_shocks_extended.py
git commit -m "feat: extend apply_shock for crisis_warning and fx_rate_shock"
```

---

### Task 3: Trust params config + TrustLedger core (trust_score dynamics)

**Files:**
- Create: `configs/economy/trust_params.yaml`
- Create: `src/economy/trust.py`
- Test: new `tests/test_trust_ledger.py`

**Interfaces:**
- Consumes: `ShockType`, `ShockEvent` (Task 1 — `ShockType.DEPEG_EVENT` and friends already exist by this point).
- Produces: `TrustParams` (Pydantic model: `lambda_shock: float`, `lambda_recover: float`, `lambda_contagion: float`, `rolling_window_days: int`), `load_trust_params(path: Path = TRUST_PARAMS_PATH) -> TrustParams`, `TrustLedger.__init__(currencies: dict[str, CurrencyConfig], params: TrustParams)`, `TrustLedger.trust_score(symbol: str) -> float`, `TrustLedger.update(fired_shocks: list[ShockEvent]) -> None`, `TrustLedger.history(symbol: str, days: int) -> list[float]`.

- [ ] **Step 1: Create the trust params config**

Create `configs/economy/trust_params.yaml`:

```yaml
lambda_shock: 0.5
lambda_recover: 0.03
lambda_contagion: 0.1
rolling_window_days: 30
```

- [ ] **Step 2: Write the failing test for TrustLedger initialization and quiet-day recovery**

Create `tests/test_trust_ledger.py`:

```python
import pytest

from src.currencies.currency import load_currency_universe
from src.economy.shocks import ShockEvent, ShockType
from src.economy.trust import TrustLedger, TrustParams, load_trust_params


def _params() -> TrustParams:
    return TrustParams(lambda_shock=0.5, lambda_recover=0.03, lambda_contagion=0.1, rolling_window_days=30)


def test_trust_ledger_initializes_at_governance_score():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())

    assert ledger.trust_score("USDC") == pytest.approx(currencies["USDC"].governance_score)
    assert ledger.trust_score("USDT") == pytest.approx(currencies["USDT"].governance_score)


def test_trust_ledger_quiet_day_recovers_toward_baseline():
    currencies = load_currency_universe()
    params = _params()
    ledger = TrustLedger(currencies, params)
    baseline = currencies["USDT"].governance_score

    # Manually depress USDT's trust via an event day, then let quiet days recover it.
    ledger.update([ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.8, target_currency="USDT")])
    depressed = ledger.trust_score("USDT")
    assert depressed < baseline

    for _ in range(5):
        ledger.update([])

    recovered = ledger.trust_score("USDT")
    assert recovered > depressed
    assert recovered < baseline  # partial recovery only, lambda_recover=0.03 is slow


def test_load_trust_params_reads_the_real_config():
    params = load_trust_params()

    assert params.lambda_shock == 0.5
    assert params.lambda_recover == 0.03
    assert params.lambda_contagion == 0.1
    assert params.rolling_window_days == 30
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_trust_ledger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.economy.trust'`.

- [ ] **Step 4: Implement TrustLedger's core trust_score dynamics**

Create `src/economy/trust.py`:

```python
"""Dynamic per-currency trust ledger.

governance_score in configs/currencies/*.yaml is a static structural prior
-- it never changes during a run. trust_score is the missing dynamic
counterpart: it starts at governance_score and moves with lived simulation
experience (decaying fast on a shock, recovering slowly on quiet days),
per docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md
Sec 3.2. peg_error_offset/liquidity_offset (Task 4) reuse the identical
mechanism for temporary shock effects -- see that task for why one ledger
tracks three quantities instead of three separate mechanisms.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from src.currencies.currency import CurrencyConfig
from src.economy.shocks import ShockEvent
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

TRUST_PARAMS_PATH = CONFIG_ROOT / "economy" / "trust_params.yaml"


class TrustParams(BaseModel):
    lambda_shock: float
    lambda_recover: float
    lambda_contagion: float
    rolling_window_days: int


def load_trust_params(path: Path = TRUST_PARAMS_PATH) -> TrustParams:
    return load_yaml_as(path, TrustParams)


class _CurrencyLedgerState(BaseModel):
    trust_score: float
    trust_history: list[float] = Field(default_factory=list)


class TrustLedger:
    def __init__(self, currencies: dict[str, CurrencyConfig], params: TrustParams):
        self._params = params
        self._asset_class_of = {symbol: cfg.asset_class for symbol, cfg in currencies.items()}
        self._baseline_governance = {symbol: cfg.governance_score for symbol, cfg in currencies.items()}
        self._state: dict[str, _CurrencyLedgerState] = {
            symbol: _CurrencyLedgerState(trust_score=cfg.governance_score, trust_history=[cfg.governance_score])
            for symbol, cfg in currencies.items()
        }

    def trust_score(self, symbol: str) -> float:
        return self._state[symbol].trust_score

    def history(self, symbol: str, days: int) -> list[float]:
        return self._state[symbol].trust_history[-days:]

    def update(self, fired_shocks: list[ShockEvent]) -> None:
        severity_by_currency: dict[str, float] = {}
        for shock in fired_shocks:
            if shock.target_currency is None:
                continue
            severity = min(1.0, shock.magnitude)
            severity_by_currency[shock.target_currency] = max(
                severity_by_currency.get(shock.target_currency, 0.0), severity
            )

        for symbol, state in self._state.items():
            severity = severity_by_currency.get(symbol)
            if severity is not None:
                state.trust_score = max(0.0, state.trust_score - self._params.lambda_shock * severity * state.trust_score)
            else:
                contagion_severity = 0.0
                for other, other_severity in severity_by_currency.items():
                    if other != symbol and self._asset_class_of.get(other) == self._asset_class_of.get(symbol):
                        contagion_severity = max(contagion_severity, other_severity)
                if contagion_severity > 0:
                    state.trust_score = max(
                        0.0, state.trust_score - self._params.lambda_contagion * contagion_severity * state.trust_score
                    )
                else:
                    baseline = self._baseline_governance[symbol]
                    state.trust_score = state.trust_score + self._params.lambda_recover * (baseline - state.trust_score)

            state.trust_history.append(state.trust_score)
            max_history = self._params.rolling_window_days * 3
            if len(state.trust_history) > max_history:
                state.trust_history = state.trust_history[-max_history:]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_trust_ledger.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 6: Commit**

```bash
git add configs/economy/trust_params.yaml src/economy/trust.py tests/test_trust_ledger.py
git commit -m "feat: add TrustLedger core trust_score dynamics"
```

---

### Task 4: TrustLedger peg_error/liquidity offset channels

**Context:** `depeg_event`, `fx_volatility_shock` (peg_error), and `liquidity_crunch`, `regulatory_enforcement`, `capital_controls` (liquidity_score) need a *temporary, decaying* effect distinct from `trust_score`'s reputational dynamics. Rather than a second mechanism, this task adds two more decaying channels to the same `TrustLedger`, sharing its constants.

**Files:**
- Modify: `src/economy/trust.py`
- Test: `tests/test_trust_ledger.py`

**Interfaces:**
- Produces: `TrustLedger.peg_error_offset(symbol: str) -> float`, `TrustLedger.liquidity_offset(symbol: str) -> float`, `TrustLedger.effective_peg_error(symbol: str, baseline: float) -> float`, `TrustLedger.effective_liquidity_score(symbol: str, baseline: float) -> float`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trust_ledger.py`:

```python
from src.utils.helpers import clamp


def test_depeg_event_spikes_and_decays_peg_error_offset():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())

    ledger.update([ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")])
    spiked = ledger.peg_error_offset("USDT")
    assert spiked == pytest.approx(0.08)
    assert ledger.effective_peg_error("USDT", currencies["USDT"].peg_error) == pytest.approx(
        currencies["USDT"].peg_error + spiked
    )

    for _ in range(10):
        ledger.update([])

    decayed = ledger.peg_error_offset("USDT")
    assert 0.0 < decayed < spiked


def test_liquidity_crunch_drops_and_recovers_liquidity_offset():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())

    ledger.update([ShockEvent(day=0, type=ShockType.LIQUIDITY_CRUNCH, magnitude=0.3, target_currency="USDC")])
    dropped = ledger.liquidity_offset("USDC")
    assert dropped < 0.0
    effective = ledger.effective_liquidity_score("USDC", currencies["USDC"].liquidity_score)
    assert effective == pytest.approx(clamp(currencies["USDC"].liquidity_score + dropped, 0.0, 1.0))

    for _ in range(10):
        ledger.update([])

    recovered = ledger.liquidity_offset("USDC")
    assert dropped < recovered < 0.0


def test_currencies_untouched_by_offset_shocks_have_zero_offset():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())

    ledger.update([ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")])

    assert ledger.peg_error_offset("USDC") == 0.0
    assert ledger.liquidity_offset("USDC") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trust_ledger.py -v`
Expected: FAIL with `AttributeError: 'TrustLedger' object has no attribute 'peg_error_offset'`.

- [ ] **Step 3: Add the offset channels**

In `src/economy/trust.py`, add `ShockType` to the existing import line (`from src.economy.shocks import ShockEvent, ShockType`), add `clamp` to imports (`from src.utils.helpers import clamp`), extend `_CurrencyLedgerState`:

```python
class _CurrencyLedgerState(BaseModel):
    trust_score: float
    trust_history: list[float] = Field(default_factory=list)
    peg_error_offset: float = 0.0
    liquidity_offset: float = 0.0
```

Add these methods to `TrustLedger` (after `history`):

```python
    def peg_error_offset(self, symbol: str) -> float:
        return self._state[symbol].peg_error_offset

    def liquidity_offset(self, symbol: str) -> float:
        return self._state[symbol].liquidity_offset

    def effective_peg_error(self, symbol: str, baseline: float) -> float:
        return max(0.0, baseline + self.peg_error_offset(symbol))

    def effective_liquidity_score(self, symbol: str, baseline: float) -> float:
        return clamp(baseline + self.liquidity_offset(symbol), 0.0, 1.0)
```

Replace the body of `update` with (adds offset bookkeeping around the existing trust_score logic; the trust_score block itself is unchanged):

```python
    _PEG_OFFSET_SHOCKS = {ShockType.DEPEG_EVENT, ShockType.FX_VOLATILITY_SHOCK}
    _LIQUIDITY_OFFSET_SHOCKS = {ShockType.LIQUIDITY_CRUNCH, ShockType.REGULATORY_ENFORCEMENT, ShockType.CAPITAL_CONTROLS}

    def update(self, fired_shocks: list[ShockEvent]) -> None:
        severity_by_currency: dict[str, float] = {}
        peg_shock_by_currency: dict[str, float] = {}
        liquidity_shock_by_currency: dict[str, float] = {}

        for shock in fired_shocks:
            if shock.target_currency is None:
                continue
            severity = min(1.0, shock.magnitude)
            severity_by_currency[shock.target_currency] = max(
                severity_by_currency.get(shock.target_currency, 0.0), severity
            )
            if shock.type in self._PEG_OFFSET_SHOCKS:
                peg_shock_by_currency[shock.target_currency] = (
                    peg_shock_by_currency.get(shock.target_currency, 0.0) + shock.magnitude
                )
            if shock.type in self._LIQUIDITY_OFFSET_SHOCKS:
                liquidity_shock_by_currency[shock.target_currency] = (
                    liquidity_shock_by_currency.get(shock.target_currency, 0.0) - abs(shock.magnitude)
                )

        for symbol, state in self._state.items():
            severity = severity_by_currency.get(symbol)
            if severity is not None:
                state.trust_score = max(0.0, state.trust_score - self._params.lambda_shock * severity * state.trust_score)
            else:
                contagion_severity = 0.0
                for other, other_severity in severity_by_currency.items():
                    if other != symbol and self._asset_class_of.get(other) == self._asset_class_of.get(symbol):
                        contagion_severity = max(contagion_severity, other_severity)
                if contagion_severity > 0:
                    state.trust_score = max(
                        0.0, state.trust_score - self._params.lambda_contagion * contagion_severity * state.trust_score
                    )
                else:
                    baseline = self._baseline_governance[symbol]
                    state.trust_score = state.trust_score + self._params.lambda_recover * (baseline - state.trust_score)

            state.trust_history.append(state.trust_score)
            max_history = self._params.rolling_window_days * 3
            if len(state.trust_history) > max_history:
                state.trust_history = state.trust_history[-max_history:]

            state.peg_error_offset += (
                -self._params.lambda_recover * state.peg_error_offset + peg_shock_by_currency.get(symbol, 0.0)
            )
            state.liquidity_offset += (
                -self._params.lambda_recover * state.liquidity_offset + liquidity_shock_by_currency.get(symbol, 0.0)
            )
```

(Note: `_PEG_OFFSET_SHOCKS`/`_LIQUIDITY_OFFSET_SHOCKS` are class attributes — place them directly above `__init__`, not inside `update`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trust_ledger.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/economy/trust.py tests/test_trust_ledger.py
git commit -m "feat: add TrustLedger peg_error/liquidity offset channels"
```

---

### Task 5: Event log + TimestepResult.fired_shocks

**Files:**
- Create: `src/economy/event_log.py`
- Modify: `src/simulation/timestep.py`
- Test: new `tests/test_event_log.py`, extend `tests/test_simulation.py`

**Interfaces:**
- Produces: `EventLog.record(shock: ShockEvent) -> None`, `EventLog.all_events() -> list[ShockEvent]`, `TimestepResult.fired_shocks: list[ShockEvent]`.
- Consumes: `env.event_queue.pop_due(day)` (existing, `src/simulation/event_queue.py`), `apply_shock`/`apply_currency_shock` (Tasks 1-2).

- [ ] **Step 1: Write the failing test for EventLog**

Create `tests/test_event_log.py`:

```python
from src.economy.event_log import EventLog
from src.economy.shocks import ShockEvent, ShockType


def test_event_log_records_and_returns_all_events():
    log = EventLog()
    shock_a = ShockEvent(day=5, type=ShockType.INFLATION, magnitude=0.02)
    shock_b = ShockEvent(day=10, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")

    log.record(shock_a)
    log.record(shock_b)

    assert log.all_events() == [shock_a, shock_b]


def test_event_log_starts_empty():
    log = EventLog()

    assert log.all_events() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_log.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.economy.event_log'`.

- [ ] **Step 3: Implement EventLog**

Create `src/economy/event_log.py`:

```python
"""Append-only record of every shock that has fired during a run.

Plain in-memory accumulator -- src/simulation/timestep.py has no database
dependency (see docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md
Sec 3.5), so persisting this to intervention_logs is a later plan's job,
reading TimestepResult.fired_shocks (this task's other change) rather than
this class directly.
"""

from src.economy.shocks import ShockEvent


class EventLog:
    def __init__(self) -> None:
        self._events: list[ShockEvent] = []

    def record(self, shock: ShockEvent) -> None:
        self._events.append(shock)

    def all_events(self) -> list[ShockEvent]:
        return list(self._events)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_event_log.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Write the failing test for TimestepResult.fired_shocks**

Add to `tests/test_simulation.py` (check the file's existing imports first and reuse its established environment-construction helper if one exists; if not, follow the pattern below):

```python
from src.economy.shocks import ShockEvent, ShockType


def test_run_timestep_reports_fired_shocks_on_the_day_they_fire():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    env.event_queue.schedule(ShockEvent(day=0, type=ShockType.INFLATION, magnitude=0.02))
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    assert len(result.fired_shocks) == 1
    assert result.fired_shocks[0].type == ShockType.INFLATION


def test_run_timestep_reports_no_fired_shocks_on_a_quiet_day():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    assert result.fired_shocks == []
```

Check `src/simulation/event_queue.py` for the exact method name used to schedule a shock (the existing `EventQueue.__init__` already takes `scenario.shocks`, a `list[ShockEvent]` — confirm whether it exposes a public `schedule()`/`add()` method, or whether tests should instead construct `Environment` via a `ScenarioConfig` whose `shocks` list already includes the `ShockEvent` at `day=0`; use whichever is idiomatic for this codebase's existing `EventQueue`, and adjust the test above accordingly if `schedule()` isn't the real method name).

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_simulation.py -v`
Expected: FAIL — `TimestepResult` has no `fired_shocks` attribute yet.

- [ ] **Step 7: Add fired_shocks to TimestepResult and wire shock application into run_timestep**

In `src/simulation/timestep.py`, add `ShockEvent` to the existing `from src.economy.shocks import apply_shock` line (`from src.economy.shocks import ShockEvent, apply_currency_shock, apply_shock`), extend `TimestepResult`:

```python
class TimestepResult(BaseModel):
    day: int
    transactions: list[Transaction] = Field(default_factory=list)
    negotiations: list[ConversationLog] = Field(default_factory=list)
    fired_shocks: list[ShockEvent] = Field(default_factory=list)
```

Replace the top of `run_timestep` (the existing steps 1-2 block) with:

```python
def run_timestep(
    env: Environment,
    day: int,
    rng: random.Random,
    max_negotiation_rounds: int = 10,
    agreement_tolerance: float = 0.01,
    concession_rate: float = 0.3,
) -> TimestepResult:
    # Steps 1-2: update macroeconomic state, currency attributes, and prices
    # from any shocks due today.
    due_shocks = env.event_queue.pop_due(day)
    for shock in due_shocks:
        env.macro_state = apply_shock(env.macro_state, shock)
        env.currencies = apply_currency_shock(env.currencies, shock)
    env.refresh_exchange_rates()

    result = TimestepResult(day=day, fired_shocks=due_shocks)
    env.marketplace.clear_listings()
```

(The rest of `run_timestep`'s body, from `sellers = [...]` onward, is unchanged — only the function's opening lines and the `TimestepResult(day=day)` construction change, to `TimestepResult(day=day, fired_shocks=due_shocks)`.)

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_simulation.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass (confirms `env.currencies` being reassigned per-day doesn't break anything reading it elsewhere in the loop).

- [ ] **Step 10: Commit**

```bash
git add src/economy/event_log.py src/simulation/timestep.py tests/test_event_log.py tests/test_simulation.py
git commit -m "feat: add EventLog and TimestepResult.fired_shocks"
```

---

### Task 6: Agent narrative memory + TimestepResult.memory_events

**Files:**
- Modify: `src/agents/memory.py`
- Modify: `src/simulation/timestep.py`
- Test: extend `tests/test_agents.py`, extend `tests/test_simulation.py`

**Interfaces:**
- Produces: `AgentMemory.narrative_events: list[str]`, `AgentMemory.record_narrative(event_text: str, max_events: int = 10) -> None`, `TimestepResult.memory_events: list[tuple[str, str, str]]` (agent_id, memory_type, memory_text).

- [ ] **Step 1: Write the failing test for AgentMemory.record_narrative**

Add to `tests/test_agents.py` (check the file's existing imports first and follow its established style):

```python
from src.agents.memory import AgentMemory


def test_record_narrative_appends_events():
    memory = AgentMemory()

    memory.record_narrative("On day 5 I held USDC through a banking crisis and lost nothing.")

    assert memory.narrative_events == ["On day 5 I held USDC through a banking crisis and lost nothing."]


def test_record_narrative_caps_at_max_events():
    memory = AgentMemory()

    for day in range(15):
        memory.record_narrative(f"Event on day {day}")

    assert len(memory.narrative_events) == 10
    assert memory.narrative_events[0] == "Event on day 5"  # oldest 5 dropped
    assert memory.narrative_events[-1] == "Event on day 14"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py -v`
Expected: FAIL with `AttributeError: 'AgentMemory' object has no attribute 'record_narrative'`.

- [ ] **Step 3: Add narrative_events to AgentMemory**

In `src/agents/memory.py`, replace the class body:

```python
class AgentMemory(BaseModel):
    outcomes: dict[str, dict[str, int]] = Field(default_factory=dict)
    narrative_events: list[str] = Field(default_factory=list)

    def record(self, symbol: str, success: bool) -> None:
        bucket = self.outcomes.setdefault(symbol, {"success": 0, "fail": 0})
        bucket["success" if success else "fail"] += 1

    def success_rate(self, symbol: str) -> float:
        bucket = self.outcomes.get(symbol)
        if not bucket:
            return 1.0
        total = bucket["success"] + bucket["fail"]
        return bucket["success"] / total if total else 1.0

    def record_narrative(self, event_text: str, max_events: int = 10) -> None:
        self.narrative_events.append(event_text)
        if len(self.narrative_events) > max_events:
            self.narrative_events = self.narrative_events[-max_events:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for TimestepResult.memory_events**

Add to `tests/test_simulation.py`:

```python
def test_run_timestep_records_narrative_memory_for_agents_holding_a_shocked_currency():
    env = Environment.build("baseline", {"consumer": 2, "merchant": 2})
    consumer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    consumer.wallet.balances["USDT"] = 100.0
    env.event_queue.schedule(ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT"))
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    matching = [e for e in result.memory_events if e[0] == consumer.agent_id]
    assert len(matching) == 1
    assert matching[0][1] == "Depeg"
    assert "USDT" in matching[0][2]
    assert matching[0][2] in consumer.memory.narrative_events
```

(Uses the same `EventQueue` scheduling mechanism confirmed/adjusted in Task 5 Step 5 — keep it consistent with whatever that task settled on.)

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_simulation.py -v`
Expected: FAIL — `TimestepResult` has no `memory_events` attribute yet.

- [ ] **Step 7: Wire memory_events into run_timestep**

In `src/simulation/timestep.py`, extend `TimestepResult`:

```python
class TimestepResult(BaseModel):
    day: int
    transactions: list[Transaction] = Field(default_factory=list)
    negotiations: list[ConversationLog] = Field(default_factory=list)
    fired_shocks: list[ShockEvent] = Field(default_factory=list)
    memory_events: list[tuple[str, str, str]] = Field(default_factory=list)
```

In `run_timestep`, directly after the `result = TimestepResult(day=day, fired_shocks=due_shocks)` line (from Task 5), add this block (it maps each currency-targeted `ShockType` to a human-readable `memory_type` label, and only records a memory for agents actually holding a positive balance of the targeted currency):

```python
    _SHOCK_MEMORY_LABELS = {
        ShockType.DEPEG_EVENT: "Depeg",
        ShockType.GOVERNANCE_DOWNGRADE: "GovernanceDowngrade",
        ShockType.LIQUIDITY_CRUNCH: "LiquidityCrunch",
        ShockType.REGULATORY_ENFORCEMENT: "RegulatoryEnforcement",
    }
    for shock in due_shocks:
        if shock.target_currency is None or shock.type not in _SHOCK_MEMORY_LABELS:
            continue
        label = _SHOCK_MEMORY_LABELS[shock.type]
        for agent in env.agents.values():
            if agent.wallet.balances.get(shock.target_currency, 0.0) > 0:
                event_text = (
                    f"Day {day}: {shock.target_currency} {label.lower()} "
                    f"(magnitude {shock.magnitude})."
                )
                agent.memory.record_narrative(event_text)
                result.memory_events.append((agent.agent_id, label, event_text))
```

`_SHOCK_MEMORY_LABELS` is a module-level constant — place it above `run_timestep`, not inside it. Add `ShockType` to the existing `from src.economy.shocks import ...` import line.

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_simulation.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/agents/memory.py src/simulation/timestep.py tests/test_agents.py tests/test_simulation.py
git commit -m "feat: add agent narrative memory and TimestepResult.memory_events"
```

---

### Task 7: CurrencyHistory / MacroHistory prompt context

**Files:**
- Modify: `src/llm/agent_reasoning.py`
- Modify: `src/llm/prompts/buyer_prompt.txt`, `src/llm/prompts/seller_prompt.txt`, `src/llm/prompts/investor_prompt.txt`, `src/llm/prompts/bank_prompt.txt`
- Test: extend `tests/test_agent_reasoning.py`

**Interfaces:**
- Consumes: `TrustLedger.trust_score`/`history` (Tasks 3-4), `EventLog.all_events()` (Task 5).
- Produces: `CurrencyHistory`, `MacroHistory` (Pydantic models), `AgentDecisionContext.currency_history: dict[str, CurrencyHistory]`, `AgentDecisionContext.macro_history: MacroHistory | None`, `build_decision_context(...)` gains optional `currency_history`/`macro_history` parameters, `render_prompt` renders a new `{history_block}` field.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_reasoning.py` (check the file's existing imports/fixtures first and follow its established style for constructing an `AgentDecisionContext`):

```python
from src.llm.agent_reasoning import CurrencyHistory, MacroHistory


def test_currency_history_renders_into_the_prompt():
    context = _build_test_context()  # reuse whatever helper this file already uses to build a minimal AgentDecisionContext
    context.currency_history = {
        "USDT": CurrencyHistory(
            trust_now=0.41,
            trust_30d_ago=0.55,
            trust_min_90d=0.38,
            trend="declining",
            depeg_events_90d=2,
            last_event_days_ago=6,
            recent_events=["Day 44: brief 1.8% depeg, recovered in 2 days"],
        )
    }
    context.macro_history = MacroHistory(
        confidence_now=0.9, confidence_30d_ago=0.95, days_since_last_shock=6, last_shock_type="depeg_event"
    )
    schema_json = "{}"

    prompt = render_prompt("buyer", context, schema_json)

    assert "declining" in prompt
    assert "Day 44: brief 1.8% depeg" in prompt
    assert "days_since_last_shock" in prompt or "6" in prompt


def test_currency_history_defaults_to_empty_and_still_renders():
    context = _build_test_context()
    schema_json = "{}"

    prompt = render_prompt("buyer", context, schema_json)

    assert "History" in prompt
```

If this test file has no existing `_build_test_context()` helper, read the file first and adapt these two tests to whatever construction pattern it already uses for `AgentDecisionContext` (e.g. directly calling `build_decision_context(...)` with minimal arguments) rather than inventing a new helper — match the file's existing convention exactly.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: FAIL with `ImportError: cannot import name 'CurrencyHistory'`.

- [ ] **Step 3: Add CurrencyHistory, MacroHistory, and extend AgentDecisionContext**

In `src/llm/agent_reasoning.py`, add after `TransactionContext`:

```python
class CurrencyHistory(BaseModel):
    trust_now: float
    trust_30d_ago: float
    trust_min_90d: float
    trend: str
    depeg_events_90d: int
    last_event_days_ago: int | None = None
    recent_events: list[str] = []


class MacroHistory(BaseModel):
    confidence_now: float
    confidence_30d_ago: float
    days_since_last_shock: int | None = None
    last_shock_type: str | None = None
```

Extend `AgentDecisionContext` (add two fields after `governance_prompt_enabled`):

```python
class AgentDecisionContext(BaseModel):
    agent: AgentUtilityContext
    candidates: list[CurrencyChainOption]
    currency_profiles: dict[str, CurrencyProfile] = {}
    live_price_snapshots: dict[str, LivePriceSnapshot] = {}
    objective_macro_state: MacroState
    perceived_macro_state: MacroState
    transaction_context: TransactionContext
    opponent_offer: NegotiationAction | None = None
    conversation_history: list[str] = []
    governance_prompt_enabled: bool = False
    currency_history: dict[str, CurrencyHistory] = {}
    macro_history: MacroHistory | None = None
```

Extend `build_decision_context`'s signature and body (add two optional parameters, mirroring how `live_price_snapshots` is already handled):

```python
def build_decision_context(
    agent_context: AgentUtilityContext,
    candidates: list[CurrencyChainOption],
    currency_profiles: dict[str, CurrencyProfile],
    objective_macro_state: MacroState,
    perceived_macro_state: MacroState,
    transaction_context: TransactionContext,
    opponent_offer: NegotiationAction | None = None,
    conversation_history: list[str] | None = None,
    governance_prompt_enabled: bool = False,
    live_price_snapshots: dict[str, LivePriceSnapshot] | None = None,
    currency_history: dict[str, CurrencyHistory] | None = None,
    macro_history: MacroHistory | None = None,
) -> AgentDecisionContext:
    candidate_symbols = {candidate.currency_symbol for candidate in candidates}
    relevant_profiles = {
        symbol: profile for symbol, profile in currency_profiles.items() if symbol in candidate_symbols
    }
    relevant_snapshots = {
        symbol: snapshot
        for symbol, snapshot in (live_price_snapshots or {}).items()
        if symbol in candidate_symbols
    }
    relevant_history = {
        symbol: history
        for symbol, history in (currency_history or {}).items()
        if symbol in candidate_symbols
    }
    return AgentDecisionContext(
        agent=agent_context,
        candidates=candidates,
        currency_profiles=relevant_profiles,
        live_price_snapshots=relevant_snapshots,
        objective_macro_state=objective_macro_state,
        perceived_macro_state=perceived_macro_state,
        transaction_context=transaction_context,
        opponent_offer=opponent_offer,
        conversation_history=conversation_history or [],
        governance_prompt_enabled=governance_prompt_enabled,
        currency_history=relevant_history,
        macro_history=macro_history,
    )
```

- [ ] **Step 4: Add the history formatter and wire it into render_prompt**

In `src/llm/agent_reasoning.py`, add after `_format_live_price_block`:

```python
def _format_history_block(currency_history: dict[str, CurrencyHistory], macro_history: MacroHistory | None) -> str:
    lines: list[str] = []
    for symbol, history in currency_history.items():
        events = "; ".join(history.recent_events) if history.recent_events else "no notable recent events"
        lines.append(
            f"- {symbol}: trust_now={history.trust_now:.2f}, trust_30d_ago={history.trust_30d_ago:.2f}, "
            f"trust_min_90d={history.trust_min_90d:.2f}, trend={history.trend}, "
            f"depeg_events_90d={history.depeg_events_90d}, last_event_days_ago={history.last_event_days_ago}. "
            f"Recent: {events}"
        )
    if macro_history is not None:
        lines.append(
            f"- Macro: confidence_now={macro_history.confidence_now:.2f}, "
            f"confidence_30d_ago={macro_history.confidence_30d_ago:.2f}, "
            f"days_since_last_shock={macro_history.days_since_last_shock}, "
            f"last_shock_type={macro_history.last_shock_type}"
        )
    return "\n".join(lines) if lines else "(no historical data available yet)"
```

In `render_prompt`, add `"history_block": _format_history_block(context.currency_history, context.macro_history),` to the `fields` dict, directly after the `"macro_block"` entry.

- [ ] **Step 5: Add the History section to all four prompt templates**

In each of `src/llm/prompts/buyer_prompt.txt`, `src/llm/prompts/seller_prompt.txt`, `src/llm/prompts/investor_prompt.txt`, `src/llm/prompts/bank_prompt.txt`, insert this block directly after the `# Macro-economic conditions\n{macro_block}\n` section and before `# Transaction context`:

```
# History -- how we got here, not just where things stand now
{history_block}

```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass (confirms the 4 prompt template edits didn't break `render_prompt`'s `.format(**fields)` call for any agent class).

- [ ] **Step 8: Commit**

```bash
git add src/llm/agent_reasoning.py src/llm/prompts/buyer_prompt.txt src/llm/prompts/seller_prompt.txt src/llm/prompts/investor_prompt.txt src/llm/prompts/bank_prompt.txt tests/test_agent_reasoning.py
git commit -m "feat: add CurrencyHistory/MacroHistory prompt context"
```

---

### Task 8: Wire TrustLedger's effective peg_error/liquidity_score into candidate generation

**Files:**
- Modify: `src/blockchain/routing_engine.py`
- Test: new `tests/test_routing_engine.py` (check first whether a test file already covers `generate_candidates` under a different name, e.g. inside `tests/test_agents.py` or `tests/test_currency_conversion.py`; if one exists, extend it instead of creating a new file)

**Interfaces:**
- Consumes: `TrustLedger.effective_peg_error`/`effective_liquidity_score` (Task 4).
- Produces: `generate_candidates(..., trust_ledger: TrustLedger | None = None)` — when supplied, candidates reflect shock-adjusted peg_error/liquidity_score instead of the static config values.

- [ ] **Step 1: Write the failing test**

Create `tests/test_routing_engine.py` (or extend the existing file found in Step 1's search):

```python
from src.blockchain.chain import load_chain_universe
from src.blockchain.routing_engine import generate_candidates
from src.currencies.currency import load_currency_universe
from src.economy.shocks import ShockEvent, ShockType
from src.economy.trust import TrustLedger, TrustParams


def _params() -> TrustParams:
    return TrustParams(lambda_shock=0.5, lambda_recover=0.03, lambda_contagion=0.1, rolling_window_days=30)


def test_generate_candidates_uses_static_values_without_a_trust_ledger():
    currencies = load_currency_universe()
    chains = load_chain_universe()

    candidates = generate_candidates({"USDT": 100.0}, currencies, chains)

    assert candidates[0].peg_error == currencies["USDT"].peg_error


def test_generate_candidates_reflects_trust_ledger_effective_peg_error():
    currencies = load_currency_universe()
    chains = load_chain_universe()
    ledger = TrustLedger(currencies, _params())
    ledger.update([ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")])

    candidates = generate_candidates({"USDT": 100.0}, currencies, chains, trust_ledger=ledger)

    assert candidates[0].peg_error == pytest.approx(currencies["USDT"].peg_error + 0.08)
    assert candidates[0].peg_error != currencies["USDT"].peg_error


def test_generate_candidates_reflects_trust_ledger_effective_liquidity_score():
    currencies = load_currency_universe()
    chains = load_chain_universe()
    ledger = TrustLedger(currencies, _params())
    ledger.update([ShockEvent(day=0, type=ShockType.LIQUIDITY_CRUNCH, magnitude=0.3, target_currency="USDC")])

    candidates = generate_candidates({"USDC": 100.0}, currencies, chains, trust_ledger=ledger)

    assert candidates[0].liquidity_score < currencies["USDC"].liquidity_score
```

Add `import pytest` at the top if not already present in whichever file this ends up in.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_routing_engine.py -v` (or the extended existing file)
Expected: FAIL with `TypeError: generate_candidates() got an unexpected keyword argument 'trust_ledger'`.

- [ ] **Step 3: Wire the optional TrustLedger through generate_candidates**

In `src/blockchain/routing_engine.py`, add `from src.economy.trust import TrustLedger` to the imports (this creates a dependency from `src/blockchain` on `src/economy` — confirm this doesn't create a circular import: `src/economy/trust.py` imports from `src.currencies.currency` and `src.economy.shocks`, neither of which imports `src.blockchain`, so this is safe), and change the function signature and body:

Only `peg_error` and the pool-derived `liquidity_score` need to change when a `trust_ledger` is supplied (the liquidity pool's own per-chain value from `liquidity_pools.get_liquidity` is the baseline that gets adjusted — it already varies by chain and is a separate signal from the currency-level trust offset):

```python
def generate_candidates(
    available_balances: dict[str, float],
    currencies: dict[str, CurrencyConfig],
    chains: dict[str, ChainConfig],
    liquidity_pools: LiquidityPoolRegistry | None = None,
    trust_ledger: TrustLedger | None = None,
) -> list[CurrencyChainOption]:
    """One candidate per (currency the agent holds a positive balance of) x (chain)."""
    liquidity_pools = liquidity_pools or LiquidityPoolRegistry()
    options: list[CurrencyChainOption] = []
    for symbol, balance in available_balances.items():
        if balance <= 0 or symbol not in currencies:
            continue
        currency = currencies[symbol]
        if trust_ledger is not None:
            peg_error = trust_ledger.effective_peg_error(symbol, currency.peg_error)
        else:
            peg_error = currency.peg_error
        for chain in chains.values():
            pool_liquidity_score = liquidity_pools.get_liquidity(currency, chain.name)
            if trust_ledger is not None:
                pool_liquidity_score = trust_ledger.effective_liquidity_score(symbol, pool_liquidity_score)
            options.append(
                CurrencyChainOption(
                    currency_symbol=symbol,
                    chain_name=chain.name,
                    governance_score=currency.governance_score,
                    liquidity_score=pool_liquidity_score,
                    peg_error=peg_error,
                    gas_fee=get_gas_fee(chain),
                    finality_seconds=chain.finality_seconds,
                    genius_compliant=currency.genius_compliant,
                )
            )
    return options
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_routing_engine.py -v` (or the extended existing file)
Expected: PASS (all 3 tests).

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: all non-`live`-marked tests pass — confirms every existing caller of `generate_candidates` (which all omit `trust_ledger`, defaulting to `None`) is unaffected.

- [ ] **Step 6: Commit**

```bash
git add src/blockchain/routing_engine.py tests/test_routing_engine.py
git commit -m "feat: wire TrustLedger effective peg_error/liquidity_score into candidate generation"
```

---

## What comes after this plan

1. **Agent population generation** (Plan 3) — the 100-agent population (roles, currency zones, per-agent CARA coefficient `a`, per-agent OpenRouter model assignment with preflight verification).
2. **Matrix runner / experiment orchestration** (Plan 4) — the master simulation + 7 sandboxes + cross-border repeats; this is also where `persist_timestep()` gets extended with a `run_id` parameter to actually write `TimestepResult.fired_shocks`/`memory_events` (this plan's output) into Plan 1's `intervention_logs`/`agent_memory_logs` tables, and where the H4 proximity-sweep scenario YAML (0/5/10/20-day `crisis_warning`→`depeg_event` gaps) gets authored using this plan's shock types.
3. **Econometrics engine** (Plan 5) — H1-H5 regression outputs.
