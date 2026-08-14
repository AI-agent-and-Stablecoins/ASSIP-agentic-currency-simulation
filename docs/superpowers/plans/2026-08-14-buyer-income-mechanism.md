# Buyer Income Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give buyer-class agents (currently only the `consumer` profile) a weekly income deposit so they stop permanently exhausting their wallet ~5 days into any run, which is what caused 99.1% transaction failure in the Aug 13 2026 real matrix run.

**Architecture:** A new config-driven `pay_income(agent, day)` function in `src/economy/income.py`, called once per simulated day for every buyer inside `run_timestep` (`src/simulation/timestep.py`), regardless of the `use_llm` path. Two new optional fields (`income_per_period`, `income_period_days`) carry the amount/schedule from `configs/agent_profiles/consumer.yaml` through `AgentProfileConfig` and `build_agent` onto `BaseAgent`, so non-buyer profiles (bank/institution/investor/merchant) are untouched (fields stay `None`, function no-ops).

**Tech Stack:** Python 3.12, pydantic 2.x, pytest. No new dependencies.

## Global Constraints

- Follow the spec exactly: `docs/superpowers/specs/2026-08-14-buyer-income-mechanism-design.md`.
- No income payment on day 0 (the initial `initial_wallet` already covers the first period) — first payment lands on `day == income_period_days`.
- Income currency matches `agent.currency_zone` via `HOME_CURRENCY_BY_ZONE = {"USD": "USDC", "EUR": "EURC"}`; an unrecognized/`None` zone is a silent no-op (matches `compute_fx_tax`'s existing treatment of an unset zone), not an error.
- No new database table, column, or ledger entry for income — it is a wallet-only mechanic (see spec §3).
- No comments beyond what the codebase already uses at each touched call site — do not add explanatory comments to lines that don't already have them, and do not remove existing comments.
- All new tests follow the existing style already used in `tests/test_agents.py` / `tests/test_simulation.py` (plain `assert`, no docstrings on trivial tests, one behavior per test).

---

### Task 1: Thread `income_per_period`/`income_period_days` from config onto agents

**Files:**
- Modify: `src/agents/agent_factory.py` (the `AgentProfileConfig` class and `build_agent` function)
- Modify: `src/agents/base_agent.py` (the `BaseAgent` class)
- Modify: `configs/agent_profiles/consumer.yaml`
- Test: `tests/test_agents.py`

**Interfaces:**
- Produces: `AgentProfileConfig.income_per_period: float | None`, `AgentProfileConfig.income_period_days: int | None`; `BaseAgent.income_per_period: float | None`, `BaseAgent.income_period_days: int | None`. Task 2 and Task 3 read these two `BaseAgent` fields.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_agents.py`:

```python
def test_consumer_profile_has_income_fields():
    profile = load_agent_profiles()["consumer"]

    assert profile.income_per_period == 250.0
    assert profile.income_period_days == 7


def test_non_buyer_profile_has_no_income_fields():
    profile = load_agent_profiles()["merchant"]

    assert profile.income_per_period is None
    assert profile.income_period_days is None


def test_build_agent_carries_income_fields_onto_agent():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)

    assert agent.income_per_period == 250.0
    assert agent.income_period_days == 7


def test_build_agent_leaves_income_fields_none_for_a_profile_without_them():
    profile = load_agent_profiles()["merchant"]
    agent = build_agent(profile)

    assert agent.income_per_period is None
    assert agent.income_period_days is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agents.py -k income -v`
Expected: FAIL — `AttributeError: 'AgentProfileConfig' object has no attribute 'income_per_period'` (pydantic models don't have undeclared attributes).

- [ ] **Step 3: Add the fields to `AgentProfileConfig`**

In `src/agents/agent_factory.py`, change:

```python
class AgentProfileConfig(BaseModel):
    name: str
    agent_class: AgentClass
    risk_tolerance: Literal["low", "medium", "high"]
    utility_type: Literal["crra", "cara", "multi_attribute", "risk_neutral", "epstein_zin_proxy"]
    risk_aversion: float | None = None
    eis: float | None = None
    weights: MultiAttributeWeights | None = None
    initial_wallet: dict[str, float] = {}
```

to:

```python
class AgentProfileConfig(BaseModel):
    name: str
    agent_class: AgentClass
    risk_tolerance: Literal["low", "medium", "high"]
    utility_type: Literal["crra", "cara", "multi_attribute", "risk_neutral", "epstein_zin_proxy"]
    risk_aversion: float | None = None
    eis: float | None = None
    weights: MultiAttributeWeights | None = None
    income_per_period: float | None = None
    income_period_days: int | None = None
    initial_wallet: dict[str, float] = {}
```

- [ ] **Step 4: Add the fields to `BaseAgent`**

In `src/agents/base_agent.py`, change:

```python
    currency_zone: str | None = None
    assigned_model: str | None = None
    cara_coefficient: float | None = None
    memory: AgentMemory = Field(default_factory=AgentMemory)
```

to:

```python
    currency_zone: str | None = None
    assigned_model: str | None = None
    cara_coefficient: float | None = None
    income_per_period: float | None = None
    income_period_days: int | None = None
    memory: AgentMemory = Field(default_factory=AgentMemory)
```

- [ ] **Step 5: Pass the fields through `build_agent`**

In `src/agents/agent_factory.py`, change the end of `build_agent`'s return statement from:

```python
        currency_zone=currency_zone,
        assigned_model=assigned_model,
        cara_coefficient=nominal_cara,
    )
```

to:

```python
        currency_zone=currency_zone,
        assigned_model=assigned_model,
        cara_coefficient=nominal_cara,
        income_per_period=profile.income_per_period,
        income_period_days=profile.income_period_days,
    )
```

- [ ] **Step 6: Add the income fields to the consumer profile**

In `configs/agent_profiles/consumer.yaml`, change:

```yaml
name: consumer
agent_class: buyer
risk_tolerance: low
utility_type: crra
risk_aversion: 3.0
initial_wallet:
  USDC: 1000.0
  EURC: 300.0
```

to:

```yaml
name: consumer
agent_class: buyer
risk_tolerance: low
utility_type: crra
risk_aversion: 3.0
income_per_period: 250.0
income_period_days: 7
initial_wallet:
  USDC: 1000.0
  EURC: 300.0
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agents.py -v`
Expected: PASS — all tests in the file, including the 4 new ones (this also guards against breaking the pre-existing tests in this file with the new fields).

- [ ] **Step 8: Run the full test suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, same pass count as before this task (no regressions from adding two new optional fields).

- [ ] **Step 9: Commit**

```bash
git add src/agents/agent_factory.py src/agents/base_agent.py configs/agent_profiles/consumer.yaml tests/test_agents.py
git commit -m "feat: add income_per_period/income_period_days fields to buyer agents"
```

---

### Task 2: Implement `pay_income`

**Files:**
- Create: `src/economy/income.py`
- Test: `tests/test_income.py`

**Interfaces:**
- Consumes: `BaseAgent.income_per_period`, `BaseAgent.income_period_days`, `BaseAgent.currency_zone` (from Task 1); `Wallet.deposit(symbol: str, amount: float) -> None` (`src/agents/wallet.py`, already exists); `AgentMemory.record_narrative(event_text: str, max_events: int = 10) -> None` (`src/agents/memory.py`, already exists).
- Produces: `HOME_CURRENCY_BY_ZONE: dict[str, str]` and `pay_income(agent: BaseAgent, day: int) -> tuple[str, float] | None`. Task 3 calls `pay_income(buyer, day)` for every buyer once per simulated day.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_income.py`:

```python
import pytest

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.economy.income import HOME_CURRENCY_BY_ZONE, pay_income


def _consumer(currency_zone="USD"):
    agent = build_agent(load_agent_profiles()["consumer"])
    agent.currency_zone = currency_zone
    return agent


def test_home_currency_by_zone_maps_usd_and_eur():
    assert HOME_CURRENCY_BY_ZONE == {"USD": "USDC", "EUR": "EURC"}


def test_pay_income_deposits_into_usd_zone_buyers_usdc_on_payday():
    agent = _consumer("USD")
    before = agent.wallet.balances["USDC"]

    result = pay_income(agent, day=7)

    assert result == ("USDC", 250.0)
    assert agent.wallet.balances["USDC"] == pytest.approx(before + 250.0)


def test_pay_income_deposits_into_eur_zone_buyers_eurc_on_payday():
    agent = _consumer("EUR")
    before = agent.wallet.balances["EURC"]

    result = pay_income(agent, day=7)

    assert result == ("EURC", 250.0)
    assert agent.wallet.balances["EURC"] == pytest.approx(before + 250.0)


def test_pay_income_is_a_no_op_on_day_zero():
    agent = _consumer("USD")
    before = dict(agent.wallet.balances)

    result = pay_income(agent, day=0)

    assert result is None
    assert agent.wallet.balances == before


def test_pay_income_is_a_no_op_between_paydays():
    agent = _consumer("USD")
    before = dict(agent.wallet.balances)

    result = pay_income(agent, day=3)

    assert result is None
    assert agent.wallet.balances == before


def test_pay_income_fires_again_on_the_second_payday():
    agent = _consumer("USD")

    result = pay_income(agent, day=14)

    assert result == ("USDC", 250.0)


def test_pay_income_is_a_no_op_for_a_profile_without_income_configured():
    agent = build_agent(load_agent_profiles()["merchant"])

    result = pay_income(agent, day=7)

    assert result is None


def test_pay_income_is_a_no_op_for_an_unrecognized_currency_zone():
    agent = _consumer(currency_zone=None)

    result = pay_income(agent, day=7)

    assert result is None


def test_pay_income_appends_a_narrative_memory_event():
    agent = _consumer("USD")

    pay_income(agent, day=7)

    assert agent.memory.narrative_events == ["Day 7: received 250.0 USDC income."]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_income.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.economy.income'`.

- [ ] **Step 3: Implement `src/economy/income.py`**

```python
"""Weekly income for buyer agents: without this, buyers only ever spend from
their fixed initial_wallet and the economy permanently runs dry a few days
into any multi-week run (see docs/superpowers/specs/2026-08-14-buyer-income-
mechanism-design.md). Only buyer profiles that opt in via
income_per_period/income_period_days in configs/agent_profiles/*.yaml are
paid; every other role's fields stay None, so this is a no-op for them.
"""

from src.agents.base_agent import BaseAgent

HOME_CURRENCY_BY_ZONE = {"USD": "USDC", "EUR": "EURC"}


def pay_income(agent: BaseAgent, day: int) -> tuple[str, float] | None:
    if agent.income_per_period is None or agent.income_period_days is None:
        return None
    if day == 0 or day % agent.income_period_days != 0:
        return None
    currency = HOME_CURRENCY_BY_ZONE.get(agent.currency_zone)
    if currency is None:
        return None

    agent.wallet.deposit(currency, agent.income_per_period)
    agent.memory.record_narrative(f"Day {day}: received {agent.income_per_period} {currency} income.")
    return (currency, agent.income_per_period)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_income.py -v`
Expected: PASS — all 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/economy/income.py tests/test_income.py
git commit -m "feat: add pay_income for buyer agents"
```

---

### Task 3: Wire `pay_income` into the day loop

**Files:**
- Modify: `src/simulation/timestep.py`
- Test: `tests/test_simulation.py`

**Interfaces:**
- Consumes: `pay_income(agent: BaseAgent, day: int) -> tuple[str, float] | None` (Task 2).

- [ ] **Step 1: Write the failing regression test**

Add to the end of `tests/test_simulation.py`:

```python
def test_run_timestep_settlement_resumes_after_a_buyer_receives_income():
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    consumer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    consumer.currency_zone = "USD"
    consumer.wallet.balances = {"USDC": 0.01}
    rng = random.Random(0)

    settled_before_payday = []
    for day in range(1, 7):
        result = run_timestep(env, day, rng)
        settled_before_payday.extend(tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED)

    assert settled_before_payday == []

    result = run_timestep(env, day=7, rng=rng)

    settled_on_payday = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert len(settled_on_payday) > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_simulation.py -k settlement_resumes -v`
Expected: FAIL — `assert len(settled_on_payday) > 0` fails (`settled_on_payday == []`) because no income has been paid yet, so the buyer is still broke on day 7.

- [ ] **Step 3: Add the import**

In `src/simulation/timestep.py`, change:

```python
from src.economy.fx_dynamics import advance_eur_usd_rate
from src.economy.fx_tax import compute_fx_tax, load_fx_params
from src.economy.shocks import ShockEvent, ShockType, apply_currency_shock, apply_shock
```

to:

```python
from src.economy.fx_dynamics import advance_eur_usd_rate
from src.economy.fx_tax import compute_fx_tax, load_fx_params
from src.economy.income import pay_income
from src.economy.shocks import ShockEvent, ShockType, apply_currency_shock, apply_shock
```

- [ ] **Step 4: Call `pay_income` for every buyer, once per day**

In `src/simulation/timestep.py`, inside `run_timestep`, change:

```python
    env.marketplace.clear_listings()

    sellers = [a for a in env.agents.values() if isinstance(a, SellerAgent)]
    buyers = {a.agent_id: a for a in env.agents.values() if isinstance(a, BuyerAgent)}

    for seller in sellers:
```

to:

```python
    env.marketplace.clear_listings()

    sellers = [a for a in env.agents.values() if isinstance(a, SellerAgent)]
    buyers = {a.agent_id: a for a in env.agents.values() if isinstance(a, BuyerAgent)}

    for buyer in buyers.values():
        pay_income(buyer, day)

    for seller in sellers:
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_simulation.py -k settlement_resumes -v`
Expected: PASS.

- [ ] **Step 6: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, full suite green, including `test_simulation_conserves_total_currency_balances` (that test only runs 5 days, so `day % 7 == 0` never fires within it and it stays unaffected) and `test_income.py`/`test_agents.py` from Tasks 1-2.

- [ ] **Step 7: Commit**

```bash
git add src/simulation/timestep.py tests/test_simulation.py
git commit -m "feat: pay buyer income once per day in run_timestep"
```
