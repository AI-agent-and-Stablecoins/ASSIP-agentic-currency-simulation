# Equivalence Framework Implementation Plan (Sub-Project C)

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Use subagent-driven execution with review checkpoints.

**Goal:** Build the equivalence/indifference-search framework per `docs/superpowers/specs/2026-08-14-equivalence-framework-design.md`: a new end-of-run elicitation question ("would you switch?"), and a per-agent binary search over one currency/chain characteristic that uses it, aggregated into a cohort mean compensation value.

**Architecture:** 3 tasks. Task 1 builds the new elicitation LLM path (schema + prompt + call wrapper, added to `src/llm/llm_router.py` since it needs that module's private retry/repair helpers — reaching into another module's underscore-prefixed internals would be worse than adding a sibling function where they're already in scope). Task 2 builds the pure `EquivalenceComparison` data table. Task 3 builds the binary search + cohort aggregation, consuming both.

**Tech Stack:** Python 3.12, pydantic 2.x, pytest, httpx (existing dependency). No new dependencies.

## Global Constraints

- Follow the spec exactly: `docs/superpowers/specs/2026-08-14-equivalence-framework-design.md`.
- The search is a cheap post-hoc question sequence against an already-completed sim's final agent state — never a full 365-day re-run per candidate value.
- Binary search is a fixed 7 rounds per agent, per comparison.
- The cohort's reported value is the mean of its 14 agents' individual indifference points (X−Y), matching sub-project B's per-agent-then-cohort-average pattern.
- No comments beyond what the codebase already uses at each touched call site; new tests follow existing style (plain `assert`, no docstrings on trivial tests).
- Test runs are capped at ~5 minutes — run only the targeted test files named in each task, never the full suite.

---

### Task 1: The elicitation question — schema, prompt, and call wrapper

**Files:**
- Create: `src/llm/switch_elicitation.py`
- Create: `src/llm/prompts/switch_question_prompt.txt`
- Modify: `src/llm/llm_router.py` (add `call_model_for_switch`, alongside the existing `call_model`)
- Test: `tests/test_switch_elicitation.py`

**Interfaces:**
- Consumes: `RetryConfig`, `_post_chat_completion`, `_record_usage`, `AuthenticationError`, `ModelCallFailedError`, `_TECHNICAL_RETRY_STATUS_CODES` (all already exist in `src/llm/llm_router.py`); `AgentUtilityContext` (`src/llm/agent_reasoning.py`, already exists — has `risk_profile`, `utility_type`, `risk_aversion`, `wallet_balances` fields).
- Produces: `SwitchDecision(BaseModel)` with `will_switch: bool`, `reasoning: str` (`src/llm/switch_elicitation.py`); `render_switch_prompt(agent_context, fixed_symbol, fixed_field, fixed_value, varied_symbol, varied_field, varied_value) -> str`; `call_model_for_switch(prompt: str, model_id: str, client: httpx.Client, retry_config: RetryConfig | None = None) -> SwitchDecision` (`src/llm/llm_router.py`). Task 3 calls both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_switch_elicitation.py`:

```python
from src.llm.agent_reasoning import AgentUtilityContext
from src.llm.llm_router import call_model_for_switch
from src.llm.switch_elicitation import SwitchDecision, render_switch_prompt
from tests.llm_test_helpers import mock_openrouter_client


def _context():
    return AgentUtilityContext(
        agent_id="consumer-seed0-000",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=2.0,
        eis=None,
        multi_attribute_weights=None,
        wallet_balances={"USDT": 100.0, "TDUSD": 50.0},
        currency_zone="USD",
        assigned_model="vendor/model",
        cara_coefficient=None,
    )


def test_render_switch_prompt_includes_both_coins_and_the_values():
    prompt = render_switch_prompt(
        _context(),
        fixed_symbol="USDT",
        fixed_field="liquidity_score",
        fixed_value=0.98,
        varied_symbol="TDUSD",
        varied_field="liquidity_score",
        varied_value=0.50,
    )

    assert "USDT" in prompt
    assert "TDUSD" in prompt
    assert "0.98" in prompt
    assert "0.5" in prompt
    assert "will_switch" in prompt


def test_call_model_for_switch_parses_a_valid_response():
    client = mock_openrouter_client({"vendor/model": {"will_switch": True, "reasoning": "better liquidity"}})

    result = call_model_for_switch("some prompt", "vendor/model", client)

    assert isinstance(result, SwitchDecision)
    assert result.will_switch is True
    assert result.reasoning == "better liquidity"


def test_call_model_for_switch_raises_after_exhausting_retries_on_repeated_bad_json():
    import httpx
    import pytest

    from src.llm.llm_router import ModelCallFailedError, RetryConfig

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    retry_config = RetryConfig(max_retries=2, backoff_base_seconds=0.0, sleep_fn=lambda _: None)

    with pytest.raises(ModelCallFailedError):
        call_model_for_switch("some prompt", "vendor/model", client, retry_config=retry_config)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_switch_elicitation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.switch_elicitation'`.

- [ ] **Step 3: Create the prompt template**

Create `src/llm/prompts/switch_question_prompt.txt`:

```
You are an AI agent in a simulated digital-currency economy. You have spent 365 days transacting and have learned about the currencies available to you. Consider the following hypothetical and answer honestly based on what you have learned.

# Your risk and utility profile
{utility_context_block}

# The comparison
{comparison_block}

# Your task
Respond with a single JSON object matching this schema exactly -- no prose before or after the JSON:
{schema_block}
```

- [ ] **Step 4: Implement `src/llm/switch_elicitation.py`**

```python
"""Schema and prompt-rendering for the end-of-run switch-elicitation
question (docs/superpowers/specs/2026-08-14-equivalence-framework-design.md
§4): a simpler, parallel path alongside src/llm/decision_schema.py's
negotiation Decision -- a yes/no switch question has no candidates, live
prices, or conversation history to describe, and no wallet/currency/chain
constraint to validate the way a negotiation Decision does.
"""

from pydantic import BaseModel

from src.llm.agent_reasoning import PROMPTS_DIR, AgentUtilityContext

SWITCH_PROMPT_PATH = PROMPTS_DIR / "switch_question_prompt.txt"


class SwitchDecision(BaseModel):
    will_switch: bool
    reasoning: str


def _format_utility_context(agent: AgentUtilityContext) -> str:
    parts = [f"Risk profile: {agent.risk_profile}", f"Utility type: {agent.utility_type}"]
    if agent.risk_aversion is not None:
        parts.append(f"Risk aversion (CRRA/CARA-style gamma): {agent.risk_aversion}")
    return "\n".join(parts)


def render_switch_prompt(
    agent_context: AgentUtilityContext,
    fixed_symbol: str,
    fixed_field: str,
    fixed_value: float,
    varied_symbol: str,
    varied_field: str,
    varied_value: float,
) -> str:
    template = SWITCH_PROMPT_PATH.read_text(encoding="utf-8")
    comparison_block = (
        f"Coin A ({fixed_symbol}): {fixed_field} = {fixed_value}\n"
        f"Coin B ({varied_symbol}): {varied_field} = {varied_value}\n"
        f"Would you switch your holdings from {fixed_symbol} to {varied_symbol} given this?"
    )
    schema_block = '{"will_switch": true or false, "reasoning": "one sentence"}'
    return template.format(
        utility_context_block=_format_utility_context(agent_context),
        comparison_block=comparison_block,
        schema_block=schema_block,
    )
```

- [ ] **Step 5: Add `call_model_for_switch` to `src/llm/llm_router.py`**

In `src/llm/llm_router.py`, add this import near the existing `from src.llm.decision_schema import Decision` line:

```python
from src.llm.switch_elicitation import SwitchDecision
```

Then add, immediately after the existing `call_model` function's closing (`raise ModelCallFailedError(model_id, last_error)` at the end of that function):

```python
def _parse_switch_decision(response: httpx.Response) -> SwitchDecision:
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return SwitchDecision.model_validate_json(content)


def call_model_for_switch(
    prompt: str,
    model_id: str,
    client: httpx.Client,
    retry_config: RetryConfig | None = None,
) -> SwitchDecision:
    """Same 3-tier failure handling as call_model (technical-failure retry,
    one repair reprompt), targeting SwitchDecision instead of Decision --
    a yes/no switch question has no economic-validity tier to check."""
    retry_config = retry_config or RetryConfig()
    messages = [{"role": "user", "content": prompt}]
    last_error = "unknown error"

    for attempt in range(retry_config.max_retries):
        try:
            response = _post_chat_completion(client, model_id, messages)
        except httpx.TimeoutException as exc:
            last_error = f"timeout: {exc}"
            retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
            continue

        if response.status_code in (401, 403):
            raise AuthenticationError(
                f"OpenRouter rejected the API key for model {model_id}: HTTP {response.status_code}"
            )

        if response.status_code in _TECHNICAL_RETRY_STATUS_CODES:
            last_error = f"HTTP {response.status_code}"
            retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
            continue

        if response.status_code != 200:
            raise ModelCallFailedError(model_id, f"unexpected HTTP {response.status_code}")

        try:
            decision = _parse_switch_decision(response)
            _record_usage(response)
            return decision
        except (KeyError, IndexError, ValueError) as exc:
            repair_messages = messages + [
                {"role": "assistant", "content": response.text},
                {
                    "role": "user",
                    "content": (
                        f"Your last response was not valid JSON matching the required schema: {exc}. "
                        "Respond again with valid JSON only."
                    ),
                },
            ]
            try:
                repair_response = _post_chat_completion(client, model_id, repair_messages)
                repaired_decision = _parse_switch_decision(repair_response)
                _record_usage(repair_response)
                return repaired_decision
            except (KeyError, IndexError, ValueError) as repair_exc:
                last_error = f"malformed output, repair failed: {repair_exc}"
                retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
                continue

    raise ModelCallFailedError(model_id, last_error)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_switch_elicitation.py -q`
Expected: PASS — all 3 tests.

- [ ] **Step 7: Run the targeted test suite**

Run: `.venv/bin/python -m pytest tests/test_switch_elicitation.py tests/test_llm_router.py -q`
Expected: PASS, all green (confirms `call_model` itself is unaffected by the new import/function added alongside it).

- [ ] **Step 8: Commit**

```bash
git add src/llm/switch_elicitation.py src/llm/prompts/switch_question_prompt.txt src/llm/llm_router.py tests/test_switch_elicitation.py
git commit -m "feat: add the switch-elicitation LLM call path"
```

---

### Task 2: `EquivalenceComparison` data table

**Files:**
- Create: `src/economy/equivalence_framework.py` (this task only adds the dataclass + data; Task 3 adds the search function to the same file)
- Test: `tests/test_equivalence_framework.py` (this task only adds the data tests; Task 3 adds the search tests to the same file)

**Interfaces:**
- Produces: `EquivalenceComparison` (frozen dataclass: `hypothesis: str`, `fixed_currency: str`, `varied_currency: str`, `varied_field: str`, `bounds: tuple[float, float]`), `EQUIVALENCE_COMPARISONS: dict[str, list[EquivalenceComparison]]` keyed by hypothesis (`"H2"` maps to a 2-item list, every other key `"H3"`-`"H11"` to a 1-item list).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_equivalence_framework.py`:

```python
from src.economy.equivalence_framework import EQUIVALENCE_COMPARISONS, EquivalenceComparison


def test_h2_has_exactly_two_comparisons():
    assert len(EQUIVALENCE_COMPARISONS["H2"]) == 2
    varied = {c.varied_currency for c in EQUIVALENCE_COMPARISONS["H2"]}
    assert varied == {"EURC", "PAXG"}


def test_every_other_hypothesis_has_exactly_one_comparison():
    for hypothesis in ("H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"):
        assert len(EQUIVALENCE_COMPARISONS[hypothesis]) == 1


def test_h1_has_no_comparisons():
    assert "H1" not in EQUIVALENCE_COMPARISONS


def test_h3_fixes_usdt_liquidity_and_varies_tdusd_liquidity():
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    assert comparison.fixed_currency == "USDT"
    assert comparison.varied_currency == "TDUSD"
    assert comparison.varied_field == "liquidity_score"
    assert comparison.bounds == (0.0, 1.0)


def test_gas_fee_comparisons_have_the_gas_fee_bounds():
    for hypothesis in ("H5", "H8", "H10", "H11"):
        comparison = EQUIVALENCE_COMPARISONS[hypothesis][0]
        assert comparison.varied_field == "gas_fee"
        assert comparison.bounds == (0.0, 5.0)


def test_h2_comparisons_vary_governance_score():
    for comparison in EQUIVALENCE_COMPARISONS["H2"]:
        assert comparison.varied_field == "governance_score"
        assert comparison.fixed_currency == "USDT"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_equivalence_framework.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.economy.equivalence_framework'`.

- [ ] **Step 3: Implement the dataclass and data table**

Create `src/economy/equivalence_framework.py`:

```python
"""The equivalence/indifference-search framework, per
docs/superpowers/specs/2026-08-14-equivalence-framework-design.md. Every
comparison fixes one real currency's (or its assigned chain's) value at its
real config value (Y) and searches the other's corresponding value (X)
via binary search against the switch-elicitation question -- X-Y is the
reported compensation. When varied_field == "gas_fee", the modification
target is varied_currency's assigned chain (its
src.economy.hypothesis_scenarios.HYPOTHESIS_CHAIN_PINS entry), not the
currency itself -- gas_fee lives on ChainConfig, not CurrencyConfig.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EquivalenceComparison:
    hypothesis: str
    fixed_currency: str
    varied_currency: str
    varied_field: str
    bounds: tuple[float, float]


EQUIVALENCE_COMPARISONS: dict[str, list[EquivalenceComparison]] = {
    "H3": [EquivalenceComparison("H3", "USDT", "TDUSD", "liquidity_score", (0.0, 1.0))],
    "H4": [EquivalenceComparison("H4", "USDT", "DAI", "peg_error", (0.0, 0.05))],
    "H5": [EquivalenceComparison("H5", "USDT", "USDC", "gas_fee", (0.0, 5.0))],
    "H6": [EquivalenceComparison("H6", "EURC", "USDC", "liquidity_score", (0.0, 1.0))],
    "H7": [EquivalenceComparison("H7", "EURT", "USDC", "peg_error", (0.0, 0.05))],
    "H8": [EquivalenceComparison("H8", "EURC", "USDC", "gas_fee", (0.0, 5.0))],
    "H9": [EquivalenceComparison("H9", "TDUSD", "USDT", "peg_error", (0.0, 0.05))],
    "H10": [EquivalenceComparison("H10", "TDUSD", "USDT", "gas_fee", (0.0, 5.0))],
    "H11": [EquivalenceComparison("H11", "DAI", "TDUSD", "gas_fee", (0.0, 5.0))],
    "H2": [
        EquivalenceComparison("H2", "USDT", "EURC", "governance_score", (0.0, 1.0)),
        EquivalenceComparison("H2", "USDT", "PAXG", "governance_score", (0.0, 1.0)),
    ],
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_equivalence_framework.py -q`
Expected: PASS — all 6 tests.

- [ ] **Step 5: Run the targeted test suite**

Run: `.venv/bin/python -m pytest tests/test_equivalence_framework.py tests/test_hypothesis_scenarios.py -q`
Expected: PASS, all green.

- [ ] **Step 6: Commit**

```bash
git add src/economy/equivalence_framework.py tests/test_equivalence_framework.py
git commit -m "feat: add EquivalenceComparison data table for the 10 in-scope hypotheses"
```

---

### Task 3: Per-agent binary search and cohort aggregation

**Files:**
- Modify: `src/economy/equivalence_framework.py` (add the search function, alongside Task 2's dataclass/data)
- Test: `tests/test_equivalence_framework.py` (add the search tests, alongside Task 2's data tests)

**Interfaces:**
- Consumes: `EquivalenceComparison`, `EQUIVALENCE_COMPARISONS` (Task 2); `SwitchDecision`, `render_switch_prompt` (`src/llm/switch_elicitation.py`, Task 1); `call_model_for_switch` (`src/llm/llm_router.py`, Task 1); `RISK_AVERSION_COHORTS`, `CARA_ELIGIBLE_ROLES` (`src/agents/population.py`, already exist); `HYPOTHESIS_CHAIN_PINS` (`src/economy/hypothesis_scenarios.py`, already exists — needed to resolve which chain a `varied_field == "gas_fee"` comparison's currency is assigned to); `Environment.currencies`, `Environment.chains` (already exist).
- Produces: `cohort_indifference_points(env: Environment, comparison: EquivalenceComparison, model_id: str, client: httpx.Client) -> dict[float, float]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_equivalence_framework.py`:

```python
from src.agents.population import generate_hypothesis_population
from src.currencies.currency import load_currency_universe
from src.economy.equivalence_framework import cohort_indifference_points
from src.economy.hypothesis_scenarios import HYPOTHESIS_CURRENCIES
from src.economy.macro_state import MacroState
from src.economy.wallet_seeding import seed_restricted_wallets
from src.simulation.environment import Environment
from tests.llm_test_helpers import mock_openrouter_client


def _h3_env():
    real_currencies = load_currency_universe()
    restricted = {symbol: real_currencies[symbol] for symbol in HYPOTHESIS_CURRENCIES["H3"]}
    population = generate_hypothesis_population(0, ["vendor/model"], "crra")
    env = Environment.build_from_population("baseline", population, currencies=restricted)
    seed_restricted_wallets(env.agents, restricted, real_currencies, MacroState().peg_reference_rates)
    return env


def test_binary_search_converges_toward_a_known_threshold():
    env = _h3_env()
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    # Every agent always answers "will_switch: True" -- each round narrows
    # `high` toward `low`, so the search converges near bounds[0].
    client = mock_openrouter_client({"vendor/model": {"will_switch": True, "reasoning": "test"}})

    result = cohort_indifference_points(env, comparison, "vendor/model", client)

    assert set(result.keys()) <= {0.0, 2.0, 4.0, 6.0}
    # cohort_indifference_points reports (indifference_point - fixed_value);
    # fixed_value here is USDT's real liquidity_score (0.98), and the
    # indifference point converges near bounds[0]=0.0, so the result is
    # a large negative number.
    for value in result.values():
        assert value < -0.9


def test_cohort_mean_is_the_average_of_individual_agent_indifference_points():
    from src.currencies.currency import load_currency_universe

    env = _h3_env()
    comparison = EQUIVALENCE_COMPARISONS["H3"][0]
    fixed_value = load_currency_universe()[comparison.fixed_currency].liquidity_score
    client = mock_openrouter_client({"vendor/model": {"will_switch": False, "reasoning": "test"}})

    result = cohort_indifference_points(env, comparison, "vendor/model", client)

    # Every agent answering "no" the whole search pushes every trial toward
    # the upper bound -- the final per-agent indifference point converges
    # near bounds[1], so the reported (indifference_point - fixed_value)
    # converges near (bounds[1] - fixed_value).
    expected = comparison.bounds[1] - fixed_value
    for value in result.values():
        assert value == pytest.approx(expected, abs=0.05)
```

Add `import pytest` to the top of `tests/test_equivalence_framework.py` if not already present from Task 2.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_equivalence_framework.py -k "binary_search or cohort_mean" -q`
Expected: FAIL — `ImportError: cannot import name 'cohort_indifference_points'`.

- [ ] **Step 3: Implement the search function**

In `src/economy/equivalence_framework.py`, add these imports at the top:

```python
import httpx

from src.agents.population import CARA_ELIGIBLE_ROLES, RISK_AVERSION_COHORTS
from src.economy.hypothesis_scenarios import HYPOTHESIS_CHAIN_PINS
from src.llm.llm_router import call_model_for_switch
from src.llm.switch_elicitation import render_switch_prompt
from src.simulation.environment import Environment

_SEARCH_ROUNDS = 7
```

Then add, after `EQUIVALENCE_COMPARISONS`:

```python
def _fixed_value(env: Environment, comparison: EquivalenceComparison) -> float:
    currency = env.currencies[comparison.fixed_currency]
    if comparison.varied_field == "gas_fee":
        chain_name = HYPOTHESIS_CHAIN_PINS[comparison.hypothesis][comparison.fixed_currency]
        return env.chains[chain_name].gas_fee
    return getattr(currency, comparison.varied_field)


def _agent_indifference_point(
    agent, env: Environment, comparison: EquivalenceComparison, fixed_value: float, model_id: str, client: httpx.Client
) -> float:
    low, high = comparison.bounds
    agent_context = agent.build_llm_context()

    for _ in range(_SEARCH_ROUNDS):
        midpoint = (low + high) / 2
        prompt = render_switch_prompt(
            agent_context,
            fixed_symbol=comparison.fixed_currency,
            fixed_field=comparison.varied_field,
            fixed_value=fixed_value,
            varied_symbol=comparison.varied_currency,
            varied_field=comparison.varied_field,
            varied_value=midpoint,
        )
        decision = call_model_for_switch(prompt, model_id, client)
        if decision.will_switch:
            high = midpoint
        else:
            low = midpoint

    return (low + high) / 2


def cohort_indifference_points(
    env: Environment, comparison: EquivalenceComparison, model_id: str, client: httpx.Client
) -> dict[float, float]:
    fixed_value = _fixed_value(env, comparison)

    cohort_sums: dict[float, float] = {cohort: 0.0 for cohort in RISK_AVERSION_COHORTS}
    cohort_counts: dict[float, int] = {cohort: 0 for cohort in RISK_AVERSION_COHORTS}

    for agent in env.agents.values():
        if agent.profile_name not in CARA_ELIGIBLE_ROLES:
            continue
        cohort = min(RISK_AVERSION_COHORTS, key=lambda c: abs(c - agent.risk_aversion))
        indifference_point = _agent_indifference_point(agent, env, comparison, fixed_value, model_id, client)
        compensation = indifference_point - fixed_value
        cohort_sums[cohort] += compensation
        cohort_counts[cohort] += 1

    return {
        cohort: cohort_sums[cohort] / count
        for cohort, count in cohort_counts.items()
        if count > 0
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_equivalence_framework.py -q`
Expected: PASS — all 8 tests (6 from Task 2, 2 new).

- [ ] **Step 5: Run the targeted test suite**

Run: `.venv/bin/python -m pytest tests/test_equivalence_framework.py tests/test_switch_elicitation.py tests/test_population.py tests/test_hypothesis_scenarios.py tests/test_equilibrium_holdings.py -q`
Expected: PASS, all green.

- [ ] **Step 6: Commit**

```bash
git add src/economy/equivalence_framework.py tests/test_equivalence_framework.py
git commit -m "feat: add per-agent binary search and cohort aggregation for the equivalence framework"
```
