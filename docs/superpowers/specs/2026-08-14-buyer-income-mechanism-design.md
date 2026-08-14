# Buyer Income Mechanism — Design Spec

## 0. Why this spec exists

The Aug 13 2026 real matrix run (`assip.db`, 100 agents, 13 cells, 365 days, 99 OpenRouter models) collapsed economically: buyer (`consumer`) agents start with a fixed 1000 USDC / 300 EURC wallet, have no income anywhere in `src/`, and goods cost $20-200 with multiple purchases/day. By day ~5 of every 365-day cell, every consumer's USDC is exhausted; 0 of the remaining ~360 days settle any transaction (574,286 failed vs. 5,135 settled transactions overall, 99.1% failure). Negotiations kept "succeeding" (LLMs kept agreeing to deals) but deterministic settlement rejected nearly all of them for insufficient funds, meaning most of that run's OpenRouter spend produced no usable data.

User decisions made during brainstorming (2026-08-14):
1. Fix mechanism: recurring income/wage for buyer agents (not a bigger one-time endowment, not lower goods prices).
2. Income amount: an explicit config value per profile (not auto-derived from the goods basket).
3. Pay schedule: weekly (every 7 days).
4. Income currency: matches the agent's `currency_zone` (USD-zone → USDC, EUR-zone → EURC) — preserves the existing cross-border FX-tax design (`src/economy/fx_tax.py`) exactly as-is.

## 1. Config changes

Two new optional fields on `AgentProfileConfig` (`src/agents/agent_factory.py`):

```python
income_per_period: float | None = None
income_period_days: int | None = None
```

Set on `configs/agent_profiles/consumer.yaml` only:

```yaml
income_per_period: 250.0
income_period_days: 7
```

Left `None` (i.e. absent) on `bank.yaml`, `institution.yaml`, `investor.yaml`, `merchant.yaml` — none of these are buyer-class profiles, and none currently starve (banks/investors/institutions hold large static piles relative to trade size; merchants are sellers and accumulate). No income logic runs for them.

**Sizing rationale**: settled-transaction data from the broken run showed most consumers spending ~$1-4/day when unconstrained, with occasional single-day purchases of $300-950. $250/week (~$36/day) comfortably covers routine spending while still leaving an agent that blows a large chunk of its balance in one day genuinely broke until the next payday — preserving budget-constrained negotiation dynamics (relevant to the CARA/risk-aversion hypotheses, H1-H3) rather than making money a non-issue. This is a starting value, not a calibrated constant — easy to retune later since it's config, not code.

Because this is a config change, it changes `config_hash` in `simulation_runs` automatically — any new run is distinguishable in the database from the broken Aug 13 run without any extra provenance code.

## 2. Schema/model changes

`BaseAgent` (`src/agents/base_agent.py`) gets two new optional fields, mirroring the profile config:

```python
income_per_period: float | None = None
income_period_days: int | None = None
```

`build_agent` (`src/agents/agent_factory.py`) passes `profile.income_per_period` / `profile.income_period_days` straight through into the constructed agent, alongside the existing fields. No change to `cara_override`/zone/model-assignment logic.

## 3. Income module

New file `src/economy/income.py`, following the existing `src/economy/*.py` convention (small, focused, config-driven modules like `fx_tax.py`):

```python
HOME_CURRENCY_BY_ZONE = {"USD": "USDC", "EUR": "EURC"}

def pay_income(agent: BaseAgent, day: int) -> tuple[str, float] | None:
    """Deposit this agent's periodic income if due today. Returns (currency, amount)
    paid, or None if the agent has no income configured or today isn't payday."""
```

Logic:
- Returns `None` if `agent.income_per_period` or `agent.income_period_days` is `None` (non-buyer profiles, or any buyer profile that opts out by omitting the fields).
- Returns `None` if `day == 0` (the initial `initial_wallet` endowment already covers the first period — day 0 is not additionally topped up) or `day % agent.income_period_days != 0`.
- Returns `None` if `agent.currency_zone` is not in `HOME_CURRENCY_BY_ZONE` (defensive — matches how `compute_fx_tax` treats an unset/unrecognized zone as a no-op rather than an error).
- Otherwise: `agent.wallet.deposit(currency, agent.income_per_period)`, appends a narrative note to `agent.memory` (`f"Day {day}: received {amount} {currency} income."`) — the same mechanism `run_timestep` already uses to surface shock effects into agent memory/LLM context (`src/simulation/timestep.py`'s shock-memory block) — and returns `(currency, amount)`.

No new database table/column. No ledger entry (income isn't a marketplace transaction between two agents; it doesn't belong in `transactions`/`negotiations`). This keeps the fix scoped to "agents can afford to keep transacting," not a parallel bookkeeping system.

## 4. Integration point

`run_timestep` (`src/simulation/timestep.py`), inside the day-level block that already runs unconditionally regardless of `use_llm` (alongside the shock/`trust_ledger.update`/`price_index` advancement, before `env.marketplace.clear_listings()` and the buyer loop):

```python
from src.economy.income import pay_income

for buyer in buyers.values():
    pay_income(buyer, day)
```

Runs for every buyer (`buyers` dict, already built via `isinstance(a, BuyerAgent)` a few lines below where this is inserted — the insertion point is moved to right after that dict is constructed), not just `active_buyers` — income is a wallet-level daily tick independent of whether the agent transacts that day, consistent with shocks/trust dynamics applying "regardless of whether anything else happens today."

## 5. Testing

New `tests/test_income.py`:
- `pay_income` deposits the correct amount into USDC for a USD-zone buyer and EURC for a EUR-zone buyer.
- `pay_income` returns `None` and makes no wallet change on `day=0` and on any non-multiple-of-`income_period_days`.
- `pay_income` returns `None` for an agent with `income_per_period=None` (e.g. a `merchant`/`bank`/`investor`/`institution` built from their current profiles).
- `pay_income` appends the expected narrative to `agent.memory`.

Extend `tests/test_simulation.py` (the existing home for `run_timestep`-level tests) with one regression test: run a short simulation where a buyer's wallet is driven to ~0 before `income_period_days` elapses, assert it cannot settle, then advance to the payday and assert a subsequent transaction can settle again. This is the direct regression test for the bug that motivated this spec.

## 6. Out of scope

- Re-running the actual 13-cell/365-day/99-model matrix to regenerate valid research data — real OpenRouter spend, a separate decision for the user once this fix is merged.
- Any change to `bank.yaml`/`institution.yaml`/`investor.yaml`/`merchant.yaml` — none of them are starving.
- Any change to goods prices, `calibrate_currency_configs.py`, or the hallucination-detection "true price" baseline.
- Indexing income to inflation/macro state (`src/economy/monetary_policy.py`, `inflation.py`) — flat per-period income is sufficient to fix the starvation bug; tying it to inflation is a separate, un-asked-for feature.
- A ledger/transaction record for income payments — deliberately kept out of `transactions`/`negotiations` (see §3).
