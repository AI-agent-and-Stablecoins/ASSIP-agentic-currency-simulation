"""SQLAlchemy engine/session factory.

Reads DATABASE_URL from .env (python-dotenv), defaulting to a local SQLite
file so simulations run with zero setup. Swapping to Postgres later means
changing .env, not code.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.utils.constants import DEFAULT_DATABASE_URL, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

_engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_engine():
    return _engine


def create_all_tables() -> None:
    from database.models import Base

    Base.metadata.create_all(_engine)


def new_session() -> Session:
    return SessionLocal()
