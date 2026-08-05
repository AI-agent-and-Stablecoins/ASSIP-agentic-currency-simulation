"""Deterministic dev/test fixture seeding.

Not an Alembic-style migration -- this project uses Base.metadata.create_all()
for schema bootstrap (see database/session.py); this script just seeds a
small, reproducible set of agents for local development and manual testing.
"""

from database.models import Base
from database.repository import AgentRepository
from database.session import get_engine, new_session
from src.simulation.environment import Environment


# `agents` is keyed `(run_id, id)` and `wallets` `(run_id, agent_id,
# currency_symbol)` (see AgentRecord's docstring for why run-scoping is
# load-bearing), so even a fixture needs a run_id. A fixed literal keeps this
# script idempotent: re-running it upserts the same rows rather than
# accumulating a new copy of the fixture population per invocation.
SEED_RUN_ID = "seed-small-baseline"


def seed_small_baseline() -> None:
    Base.metadata.create_all(get_engine())
    env = Environment.build("baseline", {"consumer": 3, "merchant": 2})

    session = new_session()
    try:
        repo = AgentRepository(session)
        for agent in env.agents.values():
            repo.upsert_agent(agent, SEED_RUN_ID)
        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    seed_small_baseline()
