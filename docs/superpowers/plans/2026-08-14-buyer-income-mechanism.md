# Buyer Income Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give buyer-class agents (currently only the `consumer` profile) a weekly income deposit so they stop permanently exhausting their wallet ~5 days into any run, which is what caused 99.1% transaction failure in the Aug 13 2026 real matrix run.

**Architecture:** A new config-driven `pay_income(agent, day, currencies, exchange_rates)` function in `src/economy/income.py`, called once per simulated day for every buyer inside `run_timestep` (`src/simulation/timestep.py`), regardless of the `use_llm` path. Two new optional fields (`income_per_period`, `income_period_days`) carry the amount/schedule from `configs/agent_profiles/consumer.yaml` through `AgentProfileConfig` and `build_agent` onto `BaseAgent`, so non-buyer profiles (bank/institution/investor/merchant) are untouched (fields stay `None`, function no-ops).

**Tech Stack:** Python 3.12, pydantic 2.x, pytest. No new dependencies.

## Global Constraints

- Follow the spec exactly: `docs/superpowers/specs/2026-08-14-buyer-income-mechanism-design.md`.
- No income payment on day 0 (the initial `initial_wallet` already covers the first period) — first payment lands on `day == income_period_days`.
- Income currency matches `agent.currency_zone`: the exact home symbol (`HOME_CURRENCY_BY_ZONE = {"USD": "USDC", "EUR": "EURC"}`) when it exists in the environment's currency universe; otherwise (sandbox cells) split evenly by USD value across every zone-matching currency in that universe, or the whole universe if none match (Task 4, spec §7). An unrecognized/`None` zone is always a silent no-op (matches `compute_fx_tax`'s existing treatment of an unset zone), not an error.
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

---

### Task 4: Fix sandbox-cell currency resolution (post-review amendment)

**Why this task exists:** the whole-branch review after Tasks 1-3 found a Critical bug: `pay_income`'s hardcoded `USDC`/`EURC` symbols don't exist in the 12 sandbox cells' currency universe (`src/currencies/sandbox_currencies.py`'s synthetic `SBX*` symbols — see `src/simulation/matrix_runner.py`'s `_seed_sandbox_wallets`), so the first payday crashes every sandbox cell with `KeyError: 'USDC'`. See `docs/superpowers/specs/2026-08-14-buyer-income-mechanism-design.md` §7 for the full amendment writeup and the user's decision on the fix approach. Read that section before starting — it explains why a plain "match the buyer's zone" fix isn't enough (sandbox pairs often share the same peg, so zone-matching alone is ambiguous or impossible).

**Files:**
- Modify: `src/economy/income.py` (full rewrite of `pay_income`'s body and signature)
- Modify: `src/simulation/timestep.py` (the `pay_income` call site — one line)
- Modify: `tests/test_income.py` (updated signature/return type on every existing test, plus new sandbox-split tests)
- Modify: `tests/test_matrix_runner.py` (one new regression test)

**Interfaces:**
- Consumes: `ExchangeRateTable.convert(amount: float, from_symbol: str, to_symbol: str) -> float` (`src/currencies/exchange_rates.py`, already exists); `currency_zone_of(currency: CurrencyConfig) -> str | None` (`src/economy/fx_tax.py`, already exists); `env.currencies: dict[str, CurrencyConfig]` and `env.exchange_rates: ExchangeRateTable` (`src/simulation/environment.py`, already exist and are current by the time `run_timestep`'s income loop runs).
- Produces: `pay_income(agent: BaseAgent, day: int, currencies: dict[str, CurrencyConfig], exchange_rates: ExchangeRateTable) -> dict[str, float] | None` — this REPLACES the old `pay_income(agent, day) -> tuple[str, float] | None` signature from Task 2. Every existing caller/test of the old signature must be updated in this task.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_income.py` with:

```python
import pytest

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.currencies.currency import load_currency_universe
from src.currencies.exchange_rates import ExchangeRateTable
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.economy.income import HOME_CURRENCY_BY_ZONE, pay_income
from src.economy.macro_state import MacroState


def _consumer(currency_zone="USD"):
    agent = build_agent(load_agent_profiles()["consumer"])
    agent.currency_zone = currency_zone
    return agent


def _real_currencies_and_rates():
    currencies = load_currency_universe()
    return currencies, ExchangeRateTable(currencies, MacroState().peg_reference_rates)


def _sandbox_currencies_and_rates(pair_key):
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[pair_key]
    currencies = {option_a.symbol: option_a, option_b.symbol: option_b}
    return currencies, ExchangeRateTable(currencies, MacroState().peg_reference_rates)


def test_home_currency_by_zone_maps_usd_and_eur():
    assert HOME_CURRENCY_BY_ZONE == {"USD": "USDC", "EUR": "EURC"}


def test_pay_income_deposits_into_usd_zone_buyers_usdc_on_payday():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()
    before = agent.wallet.balances["USDC"]

    result = pay_income(agent, 7, currencies, rates)

    assert result == {"USDC": 250.0}
    assert agent.wallet.balances["USDC"] == pytest.approx(before + 250.0)


def test_pay_income_deposits_into_eur_zone_buyers_eurc_on_payday():
    agent = _consumer("EUR")
    currencies, rates = _real_currencies_and_rates()
    before = agent.wallet.balances["EURC"]
    expected_amount = 250.0 / 1.08  # EUR peg_reference_rate, MacroState() default

    result = pay_income(agent, 7, currencies, rates)

    assert result == pytest.approx({"EURC": expected_amount})
    assert agent.wallet.balances["EURC"] == pytest.approx(before + expected_amount)


def test_pay_income_is_a_no_op_on_day_zero():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()
    before = dict(agent.wallet.balances)

    result = pay_income(agent, 0, currencies, rates)

    assert result is None
    assert agent.wallet.balances == before


def test_pay_income_is_a_no_op_between_paydays():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()
    before = dict(agent.wallet.balances)

    result = pay_income(agent, 3, currencies, rates)

    assert result is None
    assert agent.wallet.balances == before


def test_pay_income_fires_again_on_the_second_payday():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()

    result = pay_income(agent, 14, currencies, rates)

    assert result == {"USDC": 250.0}


def test_pay_income_is_a_no_op_for_a_profile_without_income_configured():
    agent = build_agent(load_agent_profiles()["merchant"])
    currencies, rates = _real_currencies_and_rates()

    result = pay_income(agent, 7, currencies, rates)

    assert result is None


def test_pay_income_is_a_no_op_for_an_unrecognized_currency_zone():
    agent = _consumer(currency_zone=None)
    currencies, rates = _real_currencies_and_rates()

    result = pay_income(agent, 7, currencies, rates)

    assert result is None


def test_pay_income_appends_a_narrative_memory_event():
    agent = _consumer("USD")
    currencies, rates = _real_currencies_and_rates()

    pay_income(agent, 7, currencies, rates)

    assert agent.memory.narrative_events == ["Day 7: received 250.0 USDC income."]


def test_pay_income_splits_evenly_across_a_same_peg_sandbox_pair_matching_the_buyers_zone():
    agent = _consumer("USD")
    currencies, rates = _sandbox_currencies_and_rates("liquidity_vs_governance")

    result = pay_income(agent, 7, currencies, rates)

    assert set(result.keys()) == {"SBX1_HILIQ_LOGOV", "SBX1_HIGOV_LOLIQ"}
    assert result["SBX1_HILIQ_LOGOV"] == pytest.approx(125.0)
    assert result["SBX1_HIGOV_LOLIQ"] == pytest.approx(125.0)
    assert agent.wallet.balances["SBX1_HILIQ_LOGOV"] == pytest.approx(125.0)
    assert agent.wallet.balances["SBX1_HIGOV_LOLIQ"] == pytest.approx(125.0)


def test_pay_income_splits_across_the_whole_pair_when_no_currency_matches_the_buyers_zone():
    agent = _consumer("EUR")
    currencies, rates = _sandbox_currencies_and_rates("liquidity_vs_governance")

    result = pay_income(agent, 7, currencies, rates)

    assert set(result.keys()) == {"SBX1_HILIQ_LOGOV", "SBX1_HIGOV_LOLIQ"}
    assert result["SBX1_HILIQ_LOGOV"] == pytest.approx(125.0)
    assert result["SBX1_HIGOV_LOLIQ"] == pytest.approx(125.0)


def test_pay_income_pays_only_the_zone_matching_side_of_an_asset_backing_pair():
    agent = _consumer("USD")
    currencies, rates = _sandbox_currencies_and_rates("asset_backing_vs_liquidity")

    result = pay_income(agent, 7, currencies, rates)

    assert result == {"SBX4_STABLE_HILIQ": pytest.approx(250.0)}
    assert "SBX4_GOLD_LOLIQ" not in agent.wallet.balances


def test_pay_income_narrative_describes_a_multi_currency_split():
    agent = _consumer("USD")
    currencies, rates = _sandbox_currencies_and_rates("liquidity_vs_governance")

    pay_income(agent, 7, currencies, rates)

    narrative = agent.memory.narrative_events[0]
    assert "SBX1_HILIQ_LOGOV" in narrative
    assert "SBX1_HIGOV_LOLIQ" in narrative
    assert " + " in narrative
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_income.py -v`
Expected: FAIL — `TypeError: pay_income() takes 2 positional arguments but 4 were given` (the old Task 2 signature only accepts `(agent, day)`).

- [ ] **Step 3: Rewrite `src/economy/income.py`**

Replace its entire contents with:

```python
"""Weekly income for buyer agents: without this, buyers only ever spend from
their fixed initial_wallet and the economy permanently runs dry a few days
into any multi-week run (see docs/superpowers/specs/2026-08-14-buyer-income-
mechanism-design.md). Only buyer profiles that opt in via
income_per_period/income_period_days in configs/agent_profiles/*.yaml are
paid; every other role's fields stay None, so this is a no-op for them.

Payment currency resolution (see the spec's §7 amendment): the master/real
cell always pays into the buyer's exact home currency (USDC/EURC). Sandbox
cells restrict the environment's currency universe to 2 synthetic symbols
that often share the same peg (see src/currencies/sandbox_currencies.py) --
paying only one side of such a pair would bias the very comparison the
sandbox exists to run, so income there splits evenly by USD value across
every zone-matching currency, or the whole universe if none match, mirroring
src/simulation/matrix_runner.py's _seed_sandbox_wallets.
"""

from src.agents.base_agent import BaseAgent
from src.currencies.currency import CurrencyConfig
from src.currencies.exchange_rates import ExchangeRateTable
from src.economy.fx_tax import currency_zone_of

HOME_CURRENCY_BY_ZONE = {"USD": "USDC", "EUR": "EURC"}


def pay_income(
    agent: BaseAgent,
    day: int,
    currencies: dict[str, CurrencyConfig],
    exchange_rates: ExchangeRateTable,
) -> dict[str, float] | None:
    if agent.income_per_period is None or agent.income_period_days is None:
        return None
    if day == 0 or day % agent.income_period_days != 0:
        return None

    home_symbol = HOME_CURRENCY_BY_ZONE.get(agent.currency_zone)
    if home_symbol is None:
        return None

    if home_symbol in currencies:
        targets = [home_symbol]
    else:
        targets = [
            symbol for symbol, currency in currencies.items() if currency_zone_of(currency) == agent.currency_zone
        ] or list(currencies.keys())

    share_usd = agent.income_per_period / len(targets)
    paid: dict[str, float] = {}
    for symbol in targets:
        amount = exchange_rates.convert(share_usd, "USD", symbol)
        agent.wallet.deposit(symbol, amount)
        paid[symbol] = amount

    paid_desc = " + ".join(f"{amount} {symbol}" for symbol, amount in paid.items())
    agent.memory.record_narrative(f"Day {day}: received {paid_desc} income.")
    return paid
```

- [ ] **Step 4: Update the call site in `run_timestep`**

In `src/simulation/timestep.py`, change:

```python
    for buyer in buyers.values():
        pay_income(buyer, day)
```

to:

```python
    for buyer in buyers.values():
        pay_income(buyer, day, env.currencies, env.exchange_rates)
```

- [ ] **Step 5: Run the income tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_income.py -v`
Expected: PASS — all 13 tests.

- [ ] **Step 6: Write the failing matrix-runner regression test**

In `tests/test_matrix_runner.py`, change:

```python
def test_cell_keys_restricts_which_cells_run():
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=2,
        dry_run=True,
        session=_session(),
        cell_keys=["master", "liquidity_vs_governance_domestic"],
    )
    assert failures == []
    assert {r.cell_key for r in results} == {"master", "liquidity_vs_governance_domestic"}


def test_run_matrix_refuses_dry_run_false_without_any_real_clients():
```

to:

```python
def test_cell_keys_restricts_which_cells_run():
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=2,
        dry_run=True,
        session=_session(),
        cell_keys=["master", "liquidity_vs_governance_domestic"],
    )
    assert failures == []
    assert {r.cell_key for r in results} == {"master", "liquidity_vs_governance_domestic"}


def test_income_does_not_crash_a_sandbox_cell_across_a_payday():
    """Regression test for the whole-branch review's Critical finding: a
    sandbox cell's restricted 2-symbol currency universe (SBX*) doesn't
    contain the real USDC/EURC symbols pay_income used to hardcode, so the
    first payday (day 7, per consumer.yaml's income_period_days) crashed
    every sandbox cell with KeyError('USDC') before this fix."""
    results, failures = run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=8,
        dry_run=True,
        session=_session(),
        cell_keys=["master", "liquidity_vs_governance_domestic"],
    )
    assert failures == []
    assert {r.cell_key for r in results} == {"master", "liquidity_vs_governance_domestic"}


def test_run_matrix_refuses_dry_run_false_without_any_real_clients():
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_matrix_runner.py -k income_does_not_crash -v`
Expected: FAIL only if Steps 3-4 above were somehow skipped — since this task's Step 3-4 already landed by the time you reach this step, this test should actually PASS immediately. If it does, that's fine: this step exists to confirm the test is a real, currently-meaningful regression guard, not a tautology. Confirm it would have failed against the pre-Task-4 code by checking: does `git stash` (stashing only this task's changes to `src/economy/income.py` and `src/simulation/timestep.py`, keeping the new test) followed by running this one test reproduce `KeyError: 'USDC'`? If so, `git stash pop` to restore your changes and proceed. If you're unsure how to isolate this safely, skip the stash check and just note in your report that the test passes with the fix in place — do not risk losing work.

- [ ] **Step 8: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, full suite green (one pre-existing, unrelated failure is a known exception: `tests/test_dashboard_process_control.py::test_start_marks_failed_when_the_child_exits_immediately` reproduces identically on unmodified `main` and is not caused by this plan — if you see ONLY that one failure, the suite is green for the purposes of this task; any other failure must be investigated).

- [ ] **Step 9: Commit**

```bash
git add src/economy/income.py src/simulation/timestep.py tests/test_income.py tests/test_matrix_runner.py
git commit -m "fix: resolve sandbox-cell income currency instead of hardcoding USDC/EURC"
```
