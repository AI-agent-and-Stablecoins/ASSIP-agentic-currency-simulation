"""SQLAlchemy engine/session factory.

Reads DATABASE_URL from .env (python-dotenv), defaulting to a local SQLite
file so simulations run with zero setup. Swapping to Postgres later means
changing .env, not code.
"""

import os
import sqlite3
import time

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.utils.constants import DEFAULT_DATABASE_URL, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

_engine = create_engine(DATABASE_URL, echo=False)

_WAL_CONVERSION_MAX_ATTEMPTS = 8
_WAL_CONVERSION_BASE_DELAY_SECONDS = 0.05


def _set_journal_mode_wal_with_retry(cursor) -> None:
    """`PRAGMA journal_mode=WAL`'s one-time conversion of a database file
    that isn't already in WAL mode needs a brief exclusive lock on the
    file itself. Two processes opening their first-ever connection to the
    same freshly-created file at close to the same instant can collide on
    that lock -- and this specific collision surfaces as
    `sqlite3.OperationalError: database is locked` (SQLite's `SQLITE_LOCKED`,
    a schema-level lock), NOT `SQLITE_BUSY`. `busy_timeout` (set on the
    same connection, in the same pragma statement sequence) only retries
    `SQLITE_BUSY` waits -- it does nothing for this error class, confirmed
    by direct reproduction: this exact statement failed near-instantly
    (not after the 30s busy_timeout window) in roughly 2 of 6 runs of a
    two-process test before this retry loop was added. Once any connection
    has successfully converted the file to WAL, every later connection's
    identical PRAGMA is a fast, lock-free no-op (querying the
    already-current mode), so only that first race is ever actually hit.
    """
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(_WAL_CONVERSION_MAX_ATTEMPTS):
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last_error = exc
            time.sleep(_WAL_CONVERSION_BASE_DELAY_SECONDS * (2**attempt))
    raise last_error


@event.listens_for(_engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record) -> None:
    """WAL mode lets concurrent OS processes write to this SQLite file
    without one writer blocking every reader (Plan 6a: separate
    run_matrix(cell_keys=...) processes share one database). busy_timeout
    makes a writer that DOES contend retry for up to 30s instead of
    immediately raising "database is locked" -- both are no-ops for a
    non-SQLite DATABASE_URL (this listener only fires for the sqlite3
    DB-API module, which is what dbapi_connection is when DATABASE_URL
    points at a .db file).

    Pragma ORDER matters: busy_timeout must be set FIRST, so every
    statement after it (including the WAL conversion below) gets that
    protection for ordinary SQLITE_BUSY lock waits. That alone is NOT
    sufficient for the WAL conversion's own one-time race, which raises a
    different error class (SQLITE_LOCKED) busy_timeout doesn't cover --
    see `_set_journal_mode_wal_with_retry`'s docstring for the confirmed
    failure mode and why a manual retry is needed on top of busy_timeout."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA busy_timeout=30000")
    _set_journal_mode_wal_with_retry(cursor)
    cursor.close()


SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_engine():
    return _engine


def create_all_tables() -> None:
    from database.models import Base

    Base.metadata.create_all(_engine)


def new_session() -> Session:
    return SessionLocal()


def assert_schema_current(engine=None) -> None:
    """Raises `RuntimeError` if `engine`'s database has any table that EXISTS
    but is missing a column the current models declare. No-op otherwise.

    This project has no migration machinery on purpose (see
    `database/models.py`'s docstring): the schema comes from
    `Base.metadata.create_all()`, which creates missing TABLES but never
    ALTERs an existing one. So pointing today's code at a `.db` file written
    before a column was added leaves that file's old table shape in place,
    silently -- `create_all()` reports success and the mismatch only surfaces
    on the first write that names the new column.

    For `agents.run_id` (added when agent identity became run-scoped, see
    `AgentRecord`'s docstring) that first write is
    `persist_full_timestep`'s day-0 flush, which happens AFTER a full
    simulated day of real, billable LLM calls for every agent in the
    population -- and `run_matrix`'s per-cell/seed `try/except Exception`
    then swallows the `OperationalError: no such column: agents.run_id` into
    its `failures` list rather than aborting. The cell's `run_id` has by then
    already been committed to `simulation_runs` (that happens before the day
    loop), so a later retry with the same `matrix_run_id`/`checkpoint_dir`
    finds it "already registered, no checkpoint" and SKIPS the cell as
    though it had completed. Checking the columns up front, before any cell
    starts, converts that expensive-and-silent failure into a free-and-loud
    one.

    Deliberately checks only for MISSING columns across every mapped table,
    not for exact equality: extra columns in the database (a rolled-back
    model change, a hand-added scratch column) do not break any read or
    write this code performs, so failing on them would be noise. `engine`
    defaults to this module's own engine.
    """
    from sqlalchemy import inspect

    from database.models import Base

    inspector = inspect(_engine if engine is None else engine)
    stale: list[str] = []
    for table_name, table in Base.metadata.tables.items():
        if not inspector.has_table(table_name):
            continue  # brand-new database -- create_all() will build it correctly
        actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
        missing = sorted({column.name for column in table.columns} - actual_columns)
        if missing:
            stale.append(f"table '{table_name}' is missing column(s): {', '.join(missing)}")
    if not stale:
        return
    raise RuntimeError(
        "Database schema is out of date with database/models.py -- "
        + "; ".join(stale)
        + ". Base.metadata.create_all() only CREATES missing tables, it never ALTERs an existing "
        "one, so a database file written before these columns were added keeps its old shape and "
        "every write naming a new column fails deep inside the simulation day loop (after real "
        "LLM spend). Regenerate the database against a fresh file (recommended -- a partially "
        "written pre-run-scoping database cannot have its per-run split recovered retroactively), "
        "or ALTER the table(s) above to add the missing column(s) before re-running."
    )
