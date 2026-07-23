import json

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, LLMDecisionRecord
from database.repository import LLMDecisionRepository
from experiments.experiment_007_governance_prompting import run_cell


def _decision_json() -> str:
    return json.dumps(
        {
            "action": "OFFER",
            "proposed_currency": "USDC",
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 100.0,
            "reasoning": "USDC has stronger governance",
        }
    )


def test_run_cell_persists_a_decision_record_when_given_a_repository():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": _decision_json()}}]})

    client = httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    repository = LLMDecisionRepository(session)

    result = run_cell("anthropic/claude-sonnet-5", governance_prompt_enabled=True, client=client, repository=repository)
    session.commit()

    assert result["excluded"] is False
    rows = session.query(LLMDecisionRecord).all()
    assert len(rows) == 1
    assert rows[0].actual_model == "anthropic/claude-sonnet-5"
    assert rows[0].governance_prompt_enabled is True
    assert rows[0].currency == "USDC"


def test_run_cell_without_a_repository_still_works_and_persists_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": _decision_json()}}]})

    client = httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))

    result = run_cell("anthropic/claude-sonnet-5", governance_prompt_enabled=False, client=client)

    assert result["excluded"] is False
