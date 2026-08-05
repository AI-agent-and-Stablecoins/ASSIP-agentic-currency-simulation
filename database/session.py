"""SQLAlchemy engine/session factory.

Reads DATABASE_URL from .env (python-dotenv), defaulting to a local SQLite
file so simulations run with zero setup. Swapping to Postgres later means
changing .env, not code.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.utils.constants import DEFAULT_DATABASE_URL, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

_engine = create_engine(DATABASE_URL, echo=False)


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

    Pragma ORDER matters: busy_timeout must be set FIRST. The
    journal_mode=WAL conversion itself briefly needs exclusive access to the
    database file, so running it as the very first statement on a fresh
    connection left it unprotected -- with two worker processes opening
    connections at the same time, one of them could get an uncaught
    `sqlite3.OperationalError: database is locked` out of the WAL pragma
    before any busy_timeout was in effect. Setting busy_timeout first means
    the WAL conversion gets the same 30s retry protection as every
    subsequent statement."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_engine():
    return _engine


def create_all_tables() -> None:
    from database.models import Base

    Base.metadata.create_all(_engine)


def new_session() -> Session:
    return SessionLocal()
