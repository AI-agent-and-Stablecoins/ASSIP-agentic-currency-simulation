import os

import pytest

from experiments.experiment_007_governance_prompting import run_cell
from src.llm.llm_router import build_openrouter_client, load_model_roster


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "1", reason="Set RUN_LIVE_LLM_TESTS=1 to run live OpenRouter calls"
)
def test_governance_prompting_cell_runs_against_the_real_api():
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY must be set in .env to run this live test"

    roster = load_model_roster()
    client = build_openrouter_client(api_key)
    primary_model = roster.resolve(roster.routing_policies.default_reliability_chain.primary)

    result = run_cell(primary_model, governance_prompt_enabled=True, client=client)

    assert result["model_id"] == primary_model
    if not result["excluded"]:
        assert result["selected_currency"] in ("USDC", "USDT")
