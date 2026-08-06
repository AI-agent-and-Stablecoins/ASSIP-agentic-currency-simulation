"""Tests for `database.session.assert_schema_current` -- the fail-fast guard
against running a matrix against a database file whose tables predate the
current model definitions (round-2 review finding I2).

Why the guard exists: this project has no migration machinery; the schema
comes from `Base.metadata.create_all()`, which CREATES missing tables but
never ALTERs an existing one. Pointed at a `.db` file written before
`agents` gained its `run_id` column, `create_all()` silently no-ops and the
mismatch only surfaces deep inside `persist_full_timestep`'s day-0 write as
`OperationalError: no such column: agents.run_id` -- i.e. AFTER a full
simulated day of real (billable) LLM calls for every agent. Worse, that
error is swallowed by `run_matrix`'s per-cell/seed `try/except Exception`
into the `failures` list, while the cell's `run_id` has ALREADY been
committed to `simulation_runs` before the day loop -- so a later retry with
the same `matrix_run_id`/`checkpoint_dir` sees "registered, no checkpoint"
and SKIPS that cell as if it had completed successfully.
"""

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from database.models import Base, SimulationRunRecord
from database.session import assert_schema_current
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]

# The `agents` table exactly as it was defined BEFORE run-scoping: bare `id`
# primary key, no `run_id` column. Written as raw DDL rather than by checking
# out an old revision's models so the fixture is self-contained and cannot
# drift with the current model definitions.
_STALE_AGENTS_DDL = """
CREATE TABLE agents (
    id VARCHAR NOT NULL,
    agent_class VARCHAR NOT NULL,
    profile_name VARCHAR NOT NULL,
    risk_profile VARCHAR NOT NULL,
    currency_zone VARCHAR,
    assigned_model VARCHAR,
    cara_coefficient FLOAT,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id)
)
"""


def _stale_schema_engine(db_path):
    """A database file with the OLD `agents` shape and every other table at
    its CURRENT shape -- exactly what pointing today's code at yesterday's
    `.db` file produces, since `create_all()` skips the table that already
    exists and creates the rest."""
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as connection:
        connection.execute(text(_STALE_AGENTS_DDL))
    Base.metadata.create_all(engine)  # no-ops on `agents`, creates everything else
    # Premise check: create_all really did NOT repair the stale table.
    assert "run_id" not in {col["name"] for col in inspect(engine).get_columns("agents")}
    return engine


def test_assert_schema_current_is_a_noop_on_a_brand_new_empty_database(tmp_path):
    """Nothing exists yet, so nothing is stale -- `create_all()` is about to
    build the current schema. Must not raise."""
    engine = create_engine(f"sqlite:///{tmp_path / 'brand_new.db'}")
    assert_schema_current(engine)


def test_assert_schema_current_is_a_noop_on_a_freshly_created_current_schema(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'current.db'}")
    Base.metadata.create_all(engine)
    assert_schema_current(engine)


def test_assert_schema_current_raises_naming_the_missing_column_on_a_stale_agents_table(tmp_path):
    engine = _stale_schema_engine(tmp_path / "stale.db")

    with pytest.raises(RuntimeError) as excinfo:
        assert_schema_current(engine)

    message = str(excinfo.value)
    assert "agents" in message
    assert "run_id" in message
    # The message must tell the operator what to DO, not just what is wrong.
    assert "regenerate" in message.lower()


def test_run_matrix_fails_fast_on_a_stale_database_before_any_cell_runs(tmp_path):
    """The whole point of the guard: the error arrives BEFORE the first
    cell/seed spends anything and OUTSIDE the per-cell `try/except`, so it
    cannot be swallowed into `failures` and cannot leave a `simulation_runs`
    row that makes a later retry skip the cell.
    """
    engine = _stale_schema_engine(tmp_path / "stale_run_matrix.db")
    session = Session(engine)

    with pytest.raises(RuntimeError, match="run_id"):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=1,
            dry_run=True,
            session=session,
            matrix_run_id="stale-schema-test",
        )

    # No cell/seed got as far as registering itself, so a retry against a
    # repaired database still has all 13 cells left to do.
    assert session.query(SimulationRunRecord).count() == 0


def test_run_cell_group_fails_fast_on_a_stale_database(tmp_path):
    """`distributed_matrix_runner._run_cell_group` builds its own engine
    inside a worker process -- an independent entry point into the same
    database, and the one that would otherwise construct real LLM clients
    from the factories before discovering the schema problem."""
    from src.simulation.distributed_matrix_runner import _run_cell_group

    db_path = tmp_path / "stale_cell_group.db"
    _stale_schema_engine(db_path)

    def _must_not_be_called():
        raise AssertionError("client factory was invoked despite a stale schema")

    with pytest.raises(RuntimeError, match="run_id"):
        _run_cell_group(
            cell_seed_pairs=[("master", 0)],
            model_candidates=MODEL_CANDIDATES,
            num_days=1,
            dry_run=True,
            database_url=f"sqlite:///{db_path}",
            matrix_run_id="stale-schema-group-test",
            llm_max_workers=1,
            checkpoint_dir=None,
            openrouter_client_factory=_must_not_be_called,
            polygon_client_factory=_must_not_be_called,
        )
