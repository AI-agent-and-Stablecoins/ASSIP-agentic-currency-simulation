# Equilibrium Holdings Measurement Implementation Plan (Sub-Project B)

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Use subagent-driven execution with review checkpoints if orchestrating this via multiple agents.

**Goal:** Build `holdings_by_cohort(env)`, per `docs/superpowers/specs/2026-08-14-equilibrium-holdings-design.md`: given a completed hypothesis-sim's live `Environment`, compute the mean %-of-wealth held in each currency, grouped by risk-aversion cohort (0.0/2.0/4.0/6.0) among the cohorted roles (consumer/bank/investor) — the doc's H1 "equilibrium holdings" table type.

**Architecture:** One new module, `src/economy/equilibrium_holdings.py`, one function, no new database/persistence, no changes to any existing file. Reuses `Wallet.total_value_usd`/`ExchangeRateTable.convert` (currency-value math) and `RISK_AVERSION_COHORTS`/`CARA_ELIGIBLE_ROLES` (cohort identification), both already built by sub-project A.

**Tech Stack:** Python 3.12, pydantic 2.x, pytest. No new dependencies.

## Global Constraints

- Follow the spec exactly: `docs/superpowers/specs/2026-08-14-equilibrium-holdings-design.md`.
- No convergence/stabilization detection — read whatever the wallet balances are at the moment this function is called.
- Cohort bucketing is by NEAREST value in `RISK_AVERSION_COHORTS`, not exact equality — required so a CARA run's `HYPOTHESIS_CARA_ZERO_SUBSTITUTE` (`1e-4`)-valued cohort lands in the same `0.0` output key a CRRA/Epstein-Zin run's exact `0.0` cohort does.
- An agent with `total_value_usd() <= 0` is excluded from its cohort's average, not treated as a `0.0` contribution.
- No comments beyond what the codebase already uses at each touched call site; new tests follow existing style (plain `assert`, no docstrings on trivial tests).

---

### Task 1: `holdings_by_cohort`

**Files:**
- Create: `src/economy/equilibrium_holdings.py`
- Test: `tests/test_equilibrium_holdings.py`

**Interfaces:**
- Consumes: `Environment.agents: dict[str, BaseAgent]`, `Environment.currencies: dict[str, CurrencyConfig]`, `Environment.exchange_rates: ExchangeRateTable` (`src/simulation/environment.py`, already exist); `Wallet.total_value_usd(rates) -> float` (`src/agents/wallet.py`, already exists); `ExchangeRateTable.convert(amount, from_symbol, to_symbol) -> float` (`src/currencies/exchange_rates.py`, already exists); `RISK_AVERSION_COHORTS: list[float]`, `CARA_ELIGIBLE_ROLES: set[str]` (`src/agents/population.py`, already exist); `generate_hypothesis_population`, `seed_restricted_wallets`, `HYPOTHESIS_CURRENCIES` (already exist, used only in tests here).
- Produces: `holdings_by_cohort(env: Environment) -> dict[float, dict[str, float]]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_equilibrium_holdings.py`:

```python
import random

import pytest

from src.agents.population import generate_hypothesis_population
from src.currencies.currency import load_currency_universe
from src.economy.equilibrium_holdings import holdings_by_cohort
from src.economy.hypothesis_scenarios import HYPOTHESIS_CURRENCIES
from src.economy.macro_state import MacroState
from src.economy.wallet_seeding import seed_restricted_wallets
from src.simulation.environment import Environment
from src.simulation.timestep import run_timestep


def _h1_env(utility_type="crra", seed=0):
    real_currencies = load_currency_universe()
    restricted = {symbol: real_currencies[symbol] for symbol in HYPOTHESIS_CURRENCIES["H1"]}
    population = generate_hypothesis_population(seed, ["vendor/model"], utility_type)
    env = Environment.build_from_population("baseline", population, currencies=restricted)
    seed_restricted_wallets(env.agents, restricted, real_currencies, MacroState().peg_reference_rates)
    return env


def test_freshly_seeded_wallets_split_evenly_across_h1s_three_currencies():
    env = _h1_env()

    result = holdings_by_cohort(env)

    assert set(result.keys()) == {0.0, 2.0, 4.0, 6.0}
    for cohort_pcts in result.values():
        assert set(cohort_pcts.keys()) == {"USDC", "EURC", "PAXG"}
        for pct in cohort_pcts.values():
            assert pct == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_computes_the_correct_arithmetic_mean_across_a_cohort():
    env = _h1_env()
    cohort_agents = [
        a for a in env.agents.values()
        if a.profile_name in ("consumer", "bank", "investor") and a.risk_aversion == 0.0
    ]
    assert len(cohort_agents) >= 2
    cohort_agents[0].wallet.balances = {"USDC": 100.0, "EURC": 0.0, "PAXG": 0.0}
    cohort_agents[1].wallet.balances = {"USDC": 0.0, "EURC": 100.0, "PAXG": 0.0}
    for agent in cohort_agents[2:]:
        agent.wallet.balances = {"USDC": 0.0, "EURC": 100.0, "PAXG": 0.0}

    result = holdings_by_cohort(env)

    n = len(cohort_agents)
    assert result[0.0]["USDC"] == pytest.approx(1.0 / n)
    assert result[0.0]["EURC"] == pytest.approx((n - 1) / n)
    assert result[0.0]["PAXG"] == pytest.approx(0.0)


def test_cara_zero_substitute_buckets_into_the_0_0_cohort_key():
    env = _h1_env(utility_type="cara")

    result = holdings_by_cohort(env)

    assert 0.0 in result
    assert 1e-4 not in result


def test_a_bankrupt_agent_is_excluded_from_its_cohorts_average():
    env = _h1_env()
    cohort_agents = [
        a for a in env.agents.values()
        if a.profile_name in ("consumer", "bank", "investor") and a.risk_aversion == 0.0
    ]
    bankrupt = cohort_agents[0]
    bankrupt.wallet.balances = {"USDC": 0.0, "EURC": 0.0, "PAXG": 0.0}
    for agent in cohort_agents[1:]:
        agent.wallet.balances = {"USDC": 100.0, "EURC": 0.0, "PAXG": 0.0}

    result = holdings_by_cohort(env)

    assert result[0.0]["USDC"] == pytest.approx(1.0)


def test_h1_end_to_end_percentages_sum_to_one_per_cohort():
    """Plumbing test: proves the real 3-currency H1 restriction, real
    population, real wallet seeding, and a real (deterministic-path,
    fast) run_timestep loop all compose correctly through
    holdings_by_cohort. Uses the deterministic path purely for test
    speed -- per docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-
    pivot-design.md's binding decision, a REAL hypothesis-sim measuring
    genuine cohort differentiation must use use_llm=True; this test
    only proves the measurement math/plumbing, not cohort behavior."""
    env = _h1_env()
    rng = random.Random(0)
    for day in range(5):
        run_timestep(env, day, rng)

    result = holdings_by_cohort(env)

    assert len(result) > 0
    for cohort_pcts in result.values():
        assert sum(cohort_pcts.values()) == pytest.approx(1.0, rel=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_equilibrium_holdings.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.economy.equilibrium_holdings'`.

- [ ] **Step 3: Implement `src/economy/equilibrium_holdings.py`**

```python
"""Reads a completed hypothesis-sim's final agent wallets and computes
mean %-of-wealth-per-currency by risk-aversion cohort -- the "equilibrium
holdings" table type from docs/superpowers/specs/2026-08-14-equilibrium-
holdings-design.md (e.g. H1's "50% USD, 30% Euro, 20% gold" example).
Operates on a live Environment object, not persisted database records --
no runner/persistence layer exists yet for hypothesis-sims (see that
spec's §3).
"""

from src.agents.population import CARA_ELIGIBLE_ROLES, RISK_AVERSION_COHORTS
from src.simulation.environment import Environment


def holdings_by_cohort(env: Environment) -> dict[float, dict[str, float]]:
    cohort_pct_sums: dict[float, dict[str, float]] = {cohort: {} for cohort in RISK_AVERSION_COHORTS}
    cohort_agent_counts: dict[float, int] = {cohort: 0 for cohort in RISK_AVERSION_COHORTS}

    for agent in env.agents.values():
        if agent.profile_name not in CARA_ELIGIBLE_ROLES:
            continue
        total_usd = agent.wallet.total_value_usd(env.exchange_rates)
        if total_usd <= 0:
            continue

        cohort = min(RISK_AVERSION_COHORTS, key=lambda c: abs(c - agent.risk_aversion))
        cohort_agent_counts[cohort] += 1
        for symbol in env.currencies:
            balance = agent.wallet.balances.get(symbol, 0.0)
            pct = env.exchange_rates.convert(balance, symbol, "USD") / total_usd
            cohort_pct_sums[cohort][symbol] = cohort_pct_sums[cohort].get(symbol, 0.0) + pct

    result: dict[float, dict[str, float]] = {}
    for cohort, count in cohort_agent_counts.items():
        if count == 0:
            continue
        result[cohort] = {symbol: total / count for symbol, total in cohort_pct_sums[cohort].items()}
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_equilibrium_holdings.py -q`
Expected: PASS — all 5 tests.

- [ ] **Step 5: Run the targeted test suite (not the full suite — cap at ~5 minutes)**

Run: `.venv/bin/python -m pytest tests/test_equilibrium_holdings.py tests/test_population.py tests/test_hypothesis_scenarios.py tests/test_wallet_seeding.py tests/test_hypothesis_sim_integration.py -q`
Expected: PASS, all green.

- [ ] **Step 6: Commit**

```bash
git add src/economy/equilibrium_holdings.py tests/test_equilibrium_holdings.py
git commit -m "feat: add holdings_by_cohort equilibrium-holdings measurement"
```
