import pytest

from src.currencies.currency import load_currency_universe
from src.economy.event_log import EventLog
from src.economy.history_builder import build_currency_history, build_macro_history
from src.economy.shocks import ShockEvent, ShockType
from src.economy.trust import TrustLedger, TrustParams
from src.simulation.environment import Environment


def _params() -> TrustParams:
    return TrustParams(lambda_shock=0.5, lambda_recover=0.03, lambda_contagion=0.1, rolling_window_days=30)


def test_build_currency_history_reports_a_recent_depeg_event():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())
    event_log = EventLog()
    baseline = currencies["USDT"].governance_score

    for day in range(11):
        if day == 3:
            shock = ShockEvent(day=day, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")
            event_log.record(shock)
            ledger.update([shock])
        else:
            ledger.update([])

    history = build_currency_history(ledger, event_log, "USDT", day=10)

    assert history.trust_now == pytest.approx(ledger.trust_score("USDT"))
    assert history.trust_now < baseline
    assert history.depeg_events_90d == 1
    assert history.last_event_days_ago == 7  # day 10 - day 3
    assert history.trend == "declining"
    assert len(history.recent_events) == 1
    assert "depeg_event" in history.recent_events[0]
    assert "3" in history.recent_events[0]


def test_build_currency_history_on_a_quiet_currency_is_stable_with_no_events():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())
    event_log = EventLog()

    for _ in range(10):
        ledger.update([])

    history = build_currency_history(ledger, event_log, "USDC", day=9)

    assert history.depeg_events_90d == 0
    assert history.last_event_days_ago is None
    assert history.recent_events == []
    assert history.trend == "stable"
    assert history.trust_now == pytest.approx(history.trust_30d_ago)


def test_build_currency_history_excludes_events_older_than_90_days():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())
    event_log = EventLog()

    old_shock = ShockEvent(day=0, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")
    event_log.record(old_shock)
    ledger.update([old_shock])
    for _ in range(120):
        ledger.update([])

    history = build_currency_history(ledger, event_log, "USDT", day=120)

    assert history.depeg_events_90d == 0
    assert history.last_event_days_ago is None
    assert history.recent_events == []


def test_build_currency_history_only_counts_events_targeting_the_requested_symbol():
    currencies = load_currency_universe()
    ledger = TrustLedger(currencies, _params())
    event_log = EventLog()

    shock = ShockEvent(day=2, type=ShockType.DEPEG_EVENT, magnitude=0.08, target_currency="USDT")
    event_log.record(shock)
    ledger.update([shock])
    for _ in range(5):
        ledger.update([])

    history = build_currency_history(ledger, event_log, "USDC", day=6)

    assert history.depeg_events_90d == 0
    assert history.last_event_days_ago is None
    assert history.recent_events == []


def test_build_macro_history_reports_days_since_and_type_of_last_shock():
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    original_confidence = env.macro_state.confidence_index

    shock = ShockEvent(day=5, type=ShockType.BANK_FAILURE, magnitude=0.1)
    env.event_log.record(shock)
    env.macro_state.confidence_index = original_confidence - 0.1

    macro_history = build_macro_history(env, day=8)

    assert macro_history.confidence_now == pytest.approx(original_confidence - 0.1)
    assert macro_history.confidence_30d_ago == pytest.approx(original_confidence)
    assert macro_history.days_since_last_shock == 3
    assert macro_history.last_shock_type == "bank_failure"


def test_build_macro_history_with_no_shocks_yet_has_none_fields():
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})

    macro_history = build_macro_history(env, day=0)

    assert macro_history.days_since_last_shock is None
    assert macro_history.last_shock_type is None
    assert macro_history.confidence_now == pytest.approx(macro_history.confidence_30d_ago)
