"""Constructs `CurrencyHistory`/`MacroHistory` (src/llm/agent_reasoning.py)
from `TrustLedger` + `EventLog` data.

This closes the "unconsumed pipe" Phase 3 Plan 2 explicitly left open (its
design spec, docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md
Sec 3.4, defines the CurrencyHistory/MacroHistory *shape* and the
perceived-trust formula that reads trust_now/trust_30d_ago/trust_min_90d, but
leaves the exact trend/depeg_events_90d/last_event_days_ago/recent_events
*construction* as a hand-off to Plan 3/4). Field semantics implemented here,
since no formula for them is spelled out anywhere in the specs (confirmed by
re-reading Sec 3.4 in full):

- `trust_now` is the ledger's current trust_score -- always available,
  matching TrustLedger.trust_score's own docstring ("every currency has
  ongoing reputational dynamics, shocked or not").
- `trust_30d_ago` reads the value 30 entries back in `TrustLedger.history()`
  (index -31, since index -1 is "now"). Early in a run, fewer than 31 daily
  updates exist yet -- `history()` already handles that by returning
  whatever's available, so this falls back to the *oldest* known entry
  (there is no better proxy for "the earliest trust reading we have").
- `trust_min_90d` is the minimum of the last ~90 daily entries.
- `trend` compares `trust_now` to `trust_30d_ago` with a small deadband
  (0.01 trust-score points) so noise doesn't flip the label:
  below the band -> "declining", above it -> "recovering", within it ->
  "stable". This is the natural three-way reading of a "trend" derived from
  exactly two trust readings 30 days apart -- down / flat / up -- and is
  the judgment call this module documents per the design spec's own
  standard for resolving mechanism-level ambiguities the source material
  left open.
- `depeg_events_90d`/`last_event_days_ago`/`recent_events` all read
  `EventLog.all_events()` filtered to `target_currency == symbol` and to the
  trailing 90-day window (`0 <= day - event.day <= 90`), matching the
  90-day rolling-window framing the rest of `CurrencyHistory` already uses
  (`trust_min_90d`). `depeg_events_90d` counts only `ShockType.DEPEG_EVENT`
  entries; `last_event_days_ago`/`recent_events` cover any shock type
  targeting the currency (the source example pairs a non-depeg-labeled
  `last_event_days_ago` with a depeg-flavored recent-events entry, i.e. the
  two fields are not scoped to the same shock type). `recent_events` is
  capped at the 5 most recent, newest first, formatted as
  "Day {day}: {shock_type} (magnitude {magnitude})".

`MacroHistory.confidence_30d_ago` is reconstructed rather than read from a
stored series -- `Environment`/`MacroState` keep only the *current* macro
state, not a history of it (Plan 2 built no such ledger). Since
`BANK_FAILURE`/`CRISIS_WARNING` are the only two shock types that
permanently decrement `confidence_index` (`apply_shock`, src/economy/
shocks.py), the value 30 days ago is recovered by adding back every such
shock's magnitude that fired within the last 30 days. `days_since_last_shock`
/`last_shock_type` read the single most recent event in `EventLog.all_events()`
(any type, any currency -- these are the whole-economy-level twins of the
per-currency fields above).
"""

from typing import TYPE_CHECKING

from src.economy.event_log import EventLog
from src.economy.shocks import ShockEvent, ShockType
from src.economy.trust import TrustLedger
from src.llm.agent_reasoning import CurrencyHistory, MacroHistory

if TYPE_CHECKING:
    # Deferred to a type-checking-only import: src.simulation.environment
    # does not import this module, so a runtime import would be safe, but
    # keeping build_macro_history's dependency on `Environment` type-only
    # avoids growing src/economy/'s runtime import surface for a parameter
    # this function only reads two attributes off of.
    from src.simulation.environment import Environment

_TREND_DEADBAND = 0.01
_MAX_RECENT_EVENTS = 5
_CURRENCY_HISTORY_WINDOW_DAYS = 90
_CONFIDENCE_DECREMENTING_SHOCKS = {ShockType.BANK_FAILURE, ShockType.CRISIS_WARNING}


def _format_event(event: ShockEvent) -> str:
    return f"Day {event.day}: {event.type.value} (magnitude {event.magnitude})"


def _classify_trend(trust_now: float, trust_30d_ago: float) -> str:
    if trust_now < trust_30d_ago - _TREND_DEADBAND:
        return "declining"
    if trust_now > trust_30d_ago + _TREND_DEADBAND:
        return "recovering"
    return "stable"


def build_currency_history(
    ledger: TrustLedger, event_log: EventLog, symbol: str, day: int
) -> CurrencyHistory:
    trust_now = ledger.trust_score(symbol)

    history_31 = ledger.history(symbol, 31)
    trust_30d_ago = history_31[0] if history_31 else trust_now

    history_90 = ledger.history(symbol, _CURRENCY_HISTORY_WINDOW_DAYS)
    trust_min_90d = min(history_90) if history_90 else trust_now

    events_in_window = sorted(
        (
            event
            for event in event_log.all_events()
            if event.target_currency == symbol and 0 <= day - event.day <= _CURRENCY_HISTORY_WINDOW_DAYS
        ),
        key=lambda event: event.day,
        reverse=True,
    )

    depeg_events_90d = sum(1 for event in events_in_window if event.type == ShockType.DEPEG_EVENT)
    last_event_days_ago = day - events_in_window[0].day if events_in_window else None
    recent_events = [_format_event(event) for event in events_in_window[:_MAX_RECENT_EVENTS]]

    return CurrencyHistory(
        trust_now=trust_now,
        trust_30d_ago=trust_30d_ago,
        trust_min_90d=trust_min_90d,
        trend=_classify_trend(trust_now, trust_30d_ago),
        depeg_events_90d=depeg_events_90d,
        last_event_days_ago=last_event_days_ago,
        recent_events=recent_events,
    )


def build_macro_history(env: "Environment", day: int) -> MacroHistory:
    confidence_now = env.macro_state.confidence_index

    all_events = env.event_log.all_events()
    confidence_30d_ago = confidence_now + sum(
        event.magnitude
        for event in all_events
        if event.type in _CONFIDENCE_DECREMENTING_SHOCKS and 0 <= day - event.day <= 30
    )

    past_events = [event for event in all_events if event.day <= day]
    last_shock = max(past_events, key=lambda event: event.day) if past_events else None

    return MacroHistory(
        confidence_now=confidence_now,
        confidence_30d_ago=confidence_30d_ago,
        days_since_last_shock=day - last_shock.day if last_shock is not None else None,
        last_shock_type=last_shock.type.value if last_shock is not None else None,
    )
