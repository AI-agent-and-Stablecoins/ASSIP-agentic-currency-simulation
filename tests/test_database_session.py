from sqlalchemy import text

from database.session import get_engine


def test_sqlite_engine_uses_wal_journal_mode():
    engine = get_engine()
    with engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert mode.lower() == "wal"


def test_sqlite_engine_has_a_nonzero_busy_timeout():
    engine = get_engine()
    with engine.connect() as conn:
        timeout_ms = conn.execute(text("PRAGMA busy_timeout")).scalar()
    assert timeout_ms > 0
