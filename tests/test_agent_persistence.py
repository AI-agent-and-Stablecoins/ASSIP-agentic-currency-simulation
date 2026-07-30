from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import AgentRecord, Base
from database.repository import AgentRepository
from src.agents.agent_factory import build_agent, load_agent_profiles


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_upsert_agent_persists_population_fields():
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)
    agent.currency_zone = "EUR"
    agent.assigned_model = "anthropic/claude-sonnet-5"
    agent.cara_coefficient = 2.0

    repo.upsert_agent(agent)
    session.commit()

    row = session.get(AgentRecord, agent.agent_id)
    assert row.currency_zone == "EUR"
    assert row.assigned_model == "anthropic/claude-sonnet-5"
    assert row.cara_coefficient == 2.0


def test_upsert_agent_allows_none_population_fields():
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)  # currency_zone/assigned_model/cara_coefficient all None

    repo.upsert_agent(agent)
    session.commit()

    row = session.get(AgentRecord, agent.agent_id)
    assert row.currency_zone is None
    assert row.assigned_model is None
    assert row.cara_coefficient is None
