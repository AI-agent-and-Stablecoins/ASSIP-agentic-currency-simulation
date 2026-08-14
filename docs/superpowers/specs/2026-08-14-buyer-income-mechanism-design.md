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

## 7. Amendment (2026-08-14, post-implementation whole-branch review): sandbox-cell currency resolution

The final whole-branch review found a Critical bug not anticipated by §1-§5: `HOME_CURRENCY_BY_ZONE`'s hardcoded `USDC`/`EURC` symbols do not exist in the 12 sandbox cells' currency universe (`src/currencies/sandbox_currencies.py` — each sandbox cell restricts `env.currencies` to 2 synthetic `SBX*` symbols via `matrix_runner.py`'s `_seed_sandbox_wallets`). `pay_income` depositing into a symbol absent from that universe crashed every sandbox cell with `KeyError: 'USDC'` on the first payday (reproduced end-to-end: `run_matrix` recorded `failures` for every sandbox cell it touched, while `main` completed cleanly).

Compounding the fix: sandbox currency pairs often **share the same peg** (e.g. all 4 currencies across sandboxes 1/2/3/6 peg to USD; no sandbox currency pegs to EUR at all) — so "pick the currency matching the buyer's zone" is either ambiguous (two USD-pegged options in one pair) or impossible (no EUR-pegged option exists). Depositing into only one side of a pair would inject extra liquidity into that option specifically, biasing the very H7-H11 comparison the sandbox exists to run cleanly.

**User decision (2026-08-14):** `pay_income` splits the income evenly **by USD value** across every currency in the environment's universe that matches the buyer's `currency_zone` — or, if none match, across the *entire* currency universe — mirroring `_seed_sandbox_wallets`'s own existing "split evenly by USD value across both sandbox symbols" logic for the initial endowment. This never favors one side of a sandbox pair. Concretely:

- If the agent's exact home symbol (`HOME_CURRENCY_BY_ZONE[zone]`, i.e. `USDC`/`EURC`) exists in the environment's currency universe (the master/real-currency cell), pay 100% into it — unchanged behavior from §3, byte-for-byte.
- Otherwise (a sandbox cell), split the income by USD value across every currency in the universe whose `currency_zone_of(...)` (`src/economy/fx_tax.py`, already used for the FX tax) matches the buyer's zone — or, if none match, across every currency in the universe.
- Each share is converted from USD to that currency's native units via the environment's own `ExchangeRateTable.convert(amount, "USD", symbol)` — the same mechanism `_seed_sandbox_wallets` and `run_timestep`'s settlement path already use, and a fix (not a regression) to a latent unit bug in the original §3 implementation: depositing the raw USD-scale `income_per_period` number directly as native EURC units, unconverted, is off by the EUR/USD peg-reference ratio (`configs/scenarios/*.yaml`'s `peg_reference_rates: {USD: 1.0, EUR: 1.08, ...}` — a real, if small, ~8% overpayment for every EUR-zone consumer that nobody's test caught because the original tests asserted the implementation's own output, not an independently-computed expected value).

This requires widening `pay_income`'s signature from `pay_income(agent, day)` to `pay_income(agent, day, currencies, exchange_rates)` — narrow, explicit params (matching `compute_fx_tax`'s existing style), not the whole `Environment` (importing `Environment` into `src/economy/income.py` would be circular, since `src/simulation/environment.py` already imports from `src.economy.*`). The return type widens from `tuple[str, float] | None` to `dict[str, float] | None` (one entry per currency actually paid into — a singleton dict in the common master-cell case, multiple entries only in a sandbox split) to represent a genuine multi-currency payment.

**User decision, deferred (2026-08-14):** the review also found that `Environment.build` (used by `SimulationRunner`, the dashboard, and `experiments/*.py` — already documented elsewhere in this codebase as "the legacy count-based path") never assigns `currency_zone`, so income remains a permanent no-op for every run built through it — those callers still exhibit the original starvation bug, unfixed by this plan. This is accepted as a known, documented limitation rather than fixed here: the actual production run this plan exists to unblock is the `matrix_runner`-based master/sandbox matrix, not the lighter-weight `SimulationRunner` path. Revisit separately if `SimulationRunner`/dashboard/`experiments/*.py` runs need the same fix.

## 8. Out of scope

- Re-running the actual 13-cell/365-day/99-model matrix to regenerate valid research data — real OpenRouter spend, a separate decision for the user once this fix is merged.
- Any change to `bank.yaml`/`institution.yaml`/`investor.yaml`/`merchant.yaml` — none of them are starving.
- Any change to goods prices, `calibrate_currency_configs.py`, or the hallucination-detection "true price" baseline.
- Indexing income to inflation/macro state (`src/economy/monetary_policy.py`, `inflation.py`) — flat per-period income is sufficient to fix the starvation bug; tying it to inflation is a separate, un-asked-for feature.
- A ledger/transaction record for income payments — deliberately kept out of `transactions`/`negotiations` (see §3).
