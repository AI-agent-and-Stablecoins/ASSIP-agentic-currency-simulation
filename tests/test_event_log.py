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
