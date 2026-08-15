# Synthetic-Coin Track — Design Spec (Dual-Method Robustness Check)

## 0. Why this spec exists

The hypothesis-sandbox mechanism (sub-projects A/B/C) uses real stablecoins, chosen per hypothesis because a real pair happens to differ mostly along the one dimension that hypothesis tests — an approximation forced by the earlier decision to avoid synthetic currencies. The user now wants a **second, fully controlled currency track** built alongside the real-coin one: coins with exactly the attribute values specified below, not tied to any real stablecoin's actual data, so both methods can be run and their results compared for convergence.

**User decisions (2026-08-15):**
- **Run both tracks.** The real-coin track (already built, already reviewed, already mid-run) is untouched. This spec adds a new, parallel synthetic-coin track. Nothing here modifies `HYPOTHESIS_CURRENCIES`, `EQUIVALENCE_COMPARISONS`, or any already-shipped real-coin code.
- **Full cross-product per hypothesis, not a 2-coin binary choice.** Every hypothesis's sandbox offers agents the complete grid of coins spanning its two tested dimensions (e.g. 2 governance levels × 3 spread levels = 6 coins), holding the other three dimensions at one fixed, neutral value. Agents freely trade among all of them for the full 365 days, same as H1's existing mechanism — generalized to N coins instead of 3.
- **Search mechanism: whatever gives the best results** — resolved below (§6) as a discrete-level elicitation replacing the real-coin track's continuous binary search, since there are only 3 possible values per dimension here, not a continuous range.

## 1. The five synthetic attribute dimensions

| Dimension | Values | Codebase field |
|---|---|---|
| Governance | High (1) = Genius Act compliant / Low (0) = not | `governance_score` (1.0 / 0.0), `genius_compliant` (True/False) |
| Medium of exchange | USD / EUR / gold (XAU) | `peg` |
| Liquidity (bid-ask spread) | 0.01% / 0.05% / 0.10% | new `bid_ask_spread` field (see §2) |
| Volatility | 0.1% / 0.4% / 0.8% | `peg_error` (0.001 / 0.004 / 0.008) |
| Gas fees | 1¢ / 5¢ / 10¢ per transaction | `ChainConfig.gas_fee` (0.01 / 0.05 / 0.10 — already a literal per-transaction dollar cost, confirmed against `configs/blockchains/ethereum.yaml`'s `gas_fee: 2.50` / `solana.yaml`'s `gas_fee: 0.002`) |

## 2. `bid_ask_spread` is a new field, not reused `liquidity_score`

`liquidity_score` (real coins) is an abstract 0-1 "goodness" score (USDT=0.98, TDUSD=0.35) — higher is better, but it isn't literally a spread percentage; the existing `_HIGHER_IS_BETTER["liquidity_score"] = True` convention (`src/economy/equivalence_framework.py`) depends on that "higher = better" direction. The user's spec gives literal spread percentages where **lower is better** (0.01% beats 0.10%) — the opposite sign convention, and a literal quantity, not an abstract score.

Rather than overload `liquidity_score` with an inverted meaning for synthetic coins only, add a new field to the synthetic-coin path: `bid_ask_spread: float` (a literal fraction, matching `peg_error`'s own literal-fraction convention — 0.0001/0.0005/0.0010), lower-is-better like `peg_error`/`gas_fee`. This keeps the real-coin track's `liquidity_score` semantics completely untouched.

## 3. H1-H11 → dimension-pair mapping (the doc's own literal structure)

With five dimensions, C(5,2) = 10 pairs — exactly H2-H11, plus H1 (medium of exchange alone, already built). This is the *New info.pdf* text's own literal H-numbering (verbatim, from the user's earlier paste), now finally buildable without the real-currency approximation:

| Hypothesis | Dimension pair | Coins in the grid |
|---|---|---|
| H1 | Medium of exchange alone | 3 (USD / EUR / gold) |
| H2 | Governance × Medium | 2 × 3 = 6 |
| H3 | Governance × Liquidity | 2 × 3 = 6 |
| H4 | Governance × Volatility | 2 × 3 = 6 |
| H5 | Governance × Gas fees | 2 × 3 = 6 |
| H6 | Medium × Liquidity | 3 × 3 = 9 |
| H7 | Medium × Volatility | 3 × 3 = 9 |
| H8 | Medium × Gas fees | 3 × 3 = 9 |
| H9 | Liquidity × Volatility | 3 × 3 = 9 |
| H10 | Liquidity × Gas fees | 3 × 3 = 9 |
| H11 | Volatility × Gas fees | 3 × 3 = 9 |

**This differs from the real-coin track's H2-H11 mapping** (which tests liquidity/peg_error/gas_fee pairs constrained by what real coin pairs actually isolate). That's expected, not a bug to reconcile: two independent methods estimating the same underlying research questions, not required to line up 1:1 on mechanism — the whole point is checking whether their conclusions converge.

**Neutral fixed values for the three untested dimensions** in each hypothesis's grid: governance defaults to High (1) when not tested, medium defaults to USD, liquidity defaults to the middle level (0.05%), volatility defaults to the middle level (0.4%), gas fees default to the middle level (5¢). (Reasoning: a defensible, symmetric midpoint/compliant-default choice, not itself a research variable — flagged here for review, easy to change during planning if you'd rather anchor differently.)

## 4. Sandbox construction: reusing `sandbox_currencies.py`'s exact pattern

`src/currencies/sandbox_currencies.py` already builds synthetic `StablecoinConfig`/`GoldBackedConfig` instances with fully controlled attributes, isolating exactly the named dimensions per pair, holding everything else constant — precisely this task's mechanism, just for pairs instead of full grids. New module `src/currencies/synthetic_hypothesis_currencies.py` extends that same pattern to build, per hypothesis, every coin in its cross-product (§3's table), each a `StablecoinConfig` (peg=USD/EUR) or `GoldBackedConfig` (peg=gold) with:
- `governance_score`/`genius_compliant` per the grid cell's governance level
- `bid_ask_spread` per the grid cell's liquidity level (§2)
- `peg_error` per the grid cell's volatility level
- `peg` per the grid cell's medium level
- `issuer_risk`/other untested `CurrencyConfig` fields held at one shared neutral value across every coin in a given hypothesis's grid (matching `sandbox_currencies.py`'s "hold everything else constant" convention)

Gas fees: each coin chain-pinned (reusing the existing `currency_chain_pins`/`HYPOTHESIS_CHAIN_PINS` mechanism) to one of three new synthetic `ChainConfig` entries (`synthetic_gas_low`/`mid`/`high`, gas_fee=0.01/0.05/0.10) when gas fee is the tested (or fixed-at-a-specific-level) dimension.

## 5. Population, day-loop, and persistence: reuse `run_hypothesis_matrix` unchanged

Nothing about population generation, the day-loop, checkpointing, or `persist_full_timestep` differs between real and synthetic coins — `Environment.build_from_population(currencies=...)` already takes any `dict[str, CurrencyConfig]`, real or synthetic. The synthetic track needs only a different *source* of `(hypothesis, currencies, chain_pins)` triples feeding the same runner — either a `currency_source` parameter on `run_hypothesis_matrix` or a thin wrapper delegating to the same internals (an implementation-time choice, not a design fork).

`run_id`s need a track marker so synthetic and real runs against the same `matrix_run_id` don't collide or get conflated in reports: `f"{matrix_run_id}-{track}-{spec.key}-{utility_type}-seed{seed}"` (`track` = `"real"` / `"synthetic"`).

## 6. Measurement: equilibrium holdings (always) + simplified discrete compensation search

**Equilibrium holdings, every hypothesis, not just H1:** the N-coin free-choice design directly produces exactly what `holdings_by_cohort` already measures (it's already generic over any currency set) — %-of-wealth per coin per cohort, generalized from 3 coins to N. Report this for every synthetic hypothesis (H1-H11), not only H1.

**Compensation search, simplified (this spec's resolution of "whatever gives the best results"):** the real-coin track's continuous 7-10-round binary search doesn't fit a space with only 3 possible values per dimension. Replace it with a direct discrete-level elicitation: for the two coins at the grid's extreme corners (e.g. High-gov+0.01%-spread vs. Low-gov+0.01%-spread), ask the agent the existing switch question at each of the varied dimension's 3 levels directly (at most 3 `call_model_for_switch` calls per agent per comparison, down from 7-10) — find the transition point between the 3 discrete levels rather than searching a continuous range. This is both a better fit for the discrete design and meaningfully cheaper (3 LLM calls instead of 7-10, ~60% fewer calls for this phase), which also helps with the real-money cost this study has already run into.

## 7. Reporting

`src/reporting/hypothesis_tables.py` generalizes cleanly: `build_equilibrium_holdings_table` already takes an arbitrary `cell_key`/currency set (just needs the zone-label map extended beyond H1's fixed USDC/EURC/PAXG to whatever coins a given synthetic hypothesis's grid contains). A new side-by-side view — real-track table next to synthetic-track table for the same hypothesis — is the natural place to actually show "both methods converged," per the user's stated goal.

## 8. Out of scope (this spec)

- Modifying the real-coin track's `HYPOTHESIS_CURRENCIES`/`EQUIVALENCE_COMPARISONS`/search mechanism in any way.
- Cross-border/event-based variants for the synthetic track (start with baseline-only; extend later the same way the real track already has cross-border/event variants, if wanted).
- Automated statistical convergence testing between the two tracks' results (this spec produces the data side-by-side; deciding "did they converge" is a research judgment call for the paper, not code).
- Actually running the synthetic-track study at scale — this spec builds the mechanism.
