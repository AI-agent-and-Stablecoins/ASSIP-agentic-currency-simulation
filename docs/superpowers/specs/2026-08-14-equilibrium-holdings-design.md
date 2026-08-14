# Equilibrium Holdings Measurement — Design Spec (Pivot Sub-Project B)

## 0. Why this spec exists

Continuing the research-methodology pivot from `New info.pdf` (see `docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-pivot-design.md` §0 for the full A/B/C/D/E decomposition — sub-project A, the hypothesis-sandbox/population mechanism, is already built and merged). This spec covers **B: equilibrium-holdings measurement** — the doc's H1 "Baseline model" table type: run agents to day 365 and tabulate final portfolio composition (e.g. "50% USD, 30% Euro, 20% gold" at risk-neutral) by risk-aversion cohort × utility function.

**Out of scope for this spec** (separate, later pieces): the equivalence/indifference-search framework (C), the end-of-run switch-elicitation question (D), the existing econometrics engine's fate (E), and wiring hypothesis-sims into `run_matrix`'s persisted/checkpointed batch machinery (deferred from sub-project A).

## 1. Scope

**User decision (2026-08-14):** the doc's equilibrium-holdings table is genuinely meaningful only for H1 (medium of exchange alone — a real 3-way portfolio-composition question) and its cross-border variant, both already defined in `HYPOTHESIS_CURRENCIES["H1"] = ("USDC", "EURC", "PAXG")` (`src/economy/hypothesis_scenarios.py`). H2-H11 are 2-currency indifference/compensation questions (sub-project C's territory), not portfolio-composition ones. This spec nonetheless builds ONE generic function parametrized by whichever currencies a given hypothesis-sim restricts to — reusable for any hypothesis's currency set, not hardcoded to H1's 3 symbols — since the mechanism (holdings % by cohort) is the same regardless of which/how-many currencies are in play; H1 (+ cross-border) is simply the hypothesis whose table actually gets featured in the paper.

## 2. "Equilibrium" = the final day's snapshot

**User decision (2026-08-14):** no convergence detection (day-over-day stabilization threshold, etc.) — "equilibrium" means reading whatever the holdings-% happens to be on the last simulated day (365) of a completed run. This matches how every other measurement in this pivot already reads state at day 365 (H1's own risk-aversion preference test, the equivalence framework's end-of-run elicitation), and avoids designing a convergence criterion (threshold, window, non-convergence handling) for a benefit the doc itself doesn't ask for — it just says agents "continue trading" for the full period.

## 3. Data source: a live `Environment` object, not the database

**User decision (2026-08-14):** operates directly on an already-completed, in-memory `Environment` (the same `Environment.build_from_population` + manual day-loop pattern sub-project A's own tests already establish — see `tests/test_hypothesis_sim_integration.py`), not a persisted `run_matrix` database session. No runner/persistence layer exists yet for these hypothesis-sims (deliberately deferred from sub-project A), so requiring database-backed data would block this entire piece on that deferred task. Whenever the runner is eventually built, it can persist the same `Environment` state; this function works unchanged on the live object either before or in place of that persistence step.

## 4. The function

New module `src/economy/equilibrium_holdings.py` (same layer as `src/economy/hypothesis_scenarios.py`/`wallet_seeding.py` — both already operate on live `Environment`/agent state rather than persisted database records, unlike `src/econometrics/`'s DB-session-based regression pipeline):

```python
def holdings_by_cohort(env: Environment) -> dict[float, dict[str, float]]:
    """Returns {risk_aversion_cohort: {currency_symbol: mean_pct_of_wealth}}
    for env's cohorted agents (consumer/bank/investor), reading each
    agent's final env.agents[...].wallet.balances as of whenever this is
    called (the "equilibrium" snapshot -- typically after a completed
    365-day run)."""
```

**Cohort bucketing**: groups by nearest value in `RISK_AVERSION_COHORTS` (`src/agents/population.py`, already `[0.0, 2.0, 4.0, 6.0]`) rather than exact equality — necessary because a CARA run's a=0 cohort is stored as `HYPOTHESIS_CARA_ZERO_SUBSTITUTE` (`1e-4`), not exactly `0.0` (per sub-project A's amendment). Bucketing by nearest canonical value means a CRRA run's `0.0` and a CARA run's `1e-4` both land in the same output key (`0.0`), so a caller combining 3 utility-function runs into one table gets 3 columns that align on the same 4 row-keys. Cohort membership is otherwise identified by role (`consumer`/`bank`/`investor`), matching `CARA_ELIGIBLE_ROLES` (`src/agents/population.py`) — merchant/institution agents are never included, since they're never cohorted in the first place.

**Per-agent %-of-wealth**: for each currency symbol in `env.currencies`, `env.exchange_rates.convert(balance, symbol, "USD") / agent.wallet.total_value_usd(env.exchange_rates)`. Reuses the existing `Wallet.total_value_usd`/`ExchangeRateTable.convert` methods (`src/agents/wallet.py`, `src/currencies/exchange_rates.py`) — no new currency-conversion logic.

**Per-cohort aggregation**: the mean of each agent's %-of-wealth-in-symbol-X across all agents in that cohort. An agent with `total_value_usd == 0` (fully bankrupt) is excluded from its cohort's average entirely (dividing its zero wealth into a %-split is undefined, not "0% of everything") rather than raising or silently contributing a `0.0`/`0.0` NaN.

**What this function does NOT do**: combine multiple `Environment`s (one per utility function) into the doc's full 3-column table — that's the caller's job (call `holdings_by_cohort` once per already-existing `generate_hypothesis_population(..., utility_type=...)`-built environment, then zip the three cohort dicts together), trivial glue not worth a dedicated abstraction in this spec's scope.

## 5. Testing

- A cohort with all agents holding an identical, known wallet split produces exactly that split as the "mean" (sanity check on the aggregation math).
- A cohort with genuinely different agents' wallets produces the correct arithmetic mean, not e.g. a sum or a naive unweighted symbol-count.
- A CARA-run environment's `HYPOTHESIS_CARA_ZERO_SUBSTITUTE`-valued cohort buckets into the `0.0` output key, not its own separate `1e-4` key.
- An agent with zero total wealth is excluded from its cohort's average without raising.
- Works correctly against H1's real 3-currency (USDC/EURC/PAXG) restriction end-to-end: build a real hypothesis population + `Environment.build_from_population` + `seed_restricted_wallets` + a short real day-loop run, then call `holdings_by_cohort` and confirm the percentages sum to ~1.0 per agent-averaged cohort and are plausible (no currency at exactly 0% or negative).

## 6. Out of scope (this spec)

- Combining 3 utility-function environments into one paper-ready table/DataFrame/markdown — caller-level glue per §4.
- Any convergence/stabilization detection — §2.
- Database persistence of holdings results — §3.
- The equivalence/indifference-search framework (C), the end-of-run elicitation (D), the econometrics engine's fate (E), and runner-wiring — all separate, already-deferred pieces.
