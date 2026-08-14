# Hypothesis Sandboxes — Design Spec (Pivot Sub-Project A: Sandbox/Population Definitions)

## 0. Why this spec exists

`New info.pdf` (repo root, committed 2026-08-14) describes a pivot in the white paper's methodology, away from the currently-implemented regression-based H1-H11 hypothesis testing (`src/econometrics/`, run against the existing 13-cell master/6-sandbox-pair matrix) and toward: (a) equilibrium-holdings tables, (b) an "equivalence framework" that finds indifference points by varying one currency characteristic, and (c) an explicit end-of-run question asked directly to each agent ("would you switch coins under this specified change"). The new methodology defines its own 11 hypotheses (H1-H11), unrelated in numbering and content to the current codebase's H1-H11.

Per user decision (2026-08-14), this pivot is being brainstormed and built as **separate specs, one piece at a time**, in this order:

- **A (this spec):** what each of the 11 hypotheses isolates, and how each one's currency/chain/population universe is constructed — the "world-building" layer every other piece depends on.
- **B (future spec):** the equilibrium-holdings measurement (run to a stable portfolio, tabulate by risk-aversion × utility function).
- **C (future spec):** the equivalence/indifference-search mechanism (vary one characteristic until the agent is indifferent).
- **D (future spec):** the end-of-run "would you switch" elicitation asked directly to each agent once, at day 365.
- **E (future spec):** the fate of the existing regression-based econometrics engine (`src/econometrics/`) — kept alongside, retired, or something else.

**This spec covers A only.** It defines, for each of the 11 hypotheses plus the cross-border and event-based sections: which 1-2 characteristics are isolated, which **real** currencies (and, for gas-fee hypotheses, real chains) realize that isolation, and the population/cohort structure — everything needed so `run_timestep`'s existing machinery (unchanged) can run each hypothesis-sim and persist real transaction/wallet data. It does NOT cover how that data gets turned into equilibrium-holdings tables, indifference points, or elicited switch answers — those are B/C/D.

**User decision (2026-08-14): use real stablecoins, not synthetic ones.** The existing `src/currencies/sandbox_currencies.py` was built specifically because the real 9-currency universe is "internally consistent" (its own docstring: better governance correlates with better peg stability throughout), so no real pair isolates exactly one dimension the way two synthetic currencies can. The user explicitly chose real coins anyway, accepting the residual confound as a real, disclosed limitation of using real data rather than hiding it behind synthetic constructs — for each hypothesis, §3 picks whichever real pair has the largest contrast on that hypothesis's target dimension(s) relative to its other dimensions, not a perfectly clean isolation.

## 1. The 11 hypotheses (verbatim from `New info.pdf`)

| # | Hypothesis (paraphrased from the doc) |
|---|---|
| H1 | More risk-averse agents prefer USD over Euro over gold |
| H2 | More risk-averse agents prefer High-governance coins over concerns of medium of exchange (USD v Euro v gold) |
| H3 | More risk-averse agents prefer High-governance coins over Highly Liquid coins |
| H4 | More risk-averse agents prefer High-governance coins over low-volatility coins |
| H5 | More risk-averse agents prefer High-governance coins over low gas fees |
| H6 | More risk-averse agents care about medium of exchange (USD v Euro) over liquidity |
| H7 | More risk-averse agents care about medium of exchange (USD v Euro) over volatility |
| H8 | More risk-averse agents care about medium of exchange (USD v Euro) over gas fees |
| H9 | More risk-averse agents care about liquidity over volatility |
| H10 | More risk-averse agents care about liquidity over gas fees |
| H11 | More risk-averse agents care about volatility over gas fees |

Per user decision, each hypothesis's isolated characteristics follow the doc's literal wording exactly — no forced consistency across hypotheses (e.g. H2 is genuinely 3-way USD/Euro/gold per its own text, while H6-H8 are genuinely 2-way USD/Euro only, because that's what each hypothesis's own sentence says).

## 2. Common structure across every hypothesis-sim

- **3 sims per hypothesis, one per utility function** (CRRA, CARA, Epstein-Zin) — every agent in a given sim uses the same utility function, so cross-utility-function comparison isn't confounded by which agents happened to get which function (user decision).
- **4 fixed risk-aversion cohorts**: a = 0 (risk neutral), 2 (moderate), 4 (more), 6 (most) — replacing the current 8-value random CARA sample, so every table row is a real, evenly-sized cohort (user decision).
- **Cohorted/reported roles: consumer, bank, and investor** (all current `CARA_ELIGIBLE_ROLES`) — all three get a risk-aversion cohort assignment and appear in the reported results; merchant and institution keep their current fixed `multi_attribute` behavior and exist only to provide market counterparties (user decision).
- **Population size**: 100 agents per hypothesis-sim, restructured from the current `ROLE_COUNTS` so the cohorted roles divide evenly by 4:

  ```python
  HYPOTHESIS_ROLE_COUNTS = {
      "consumer": 40,   # 10 per risk-aversion cohort
      "bank": 8,        # 2 per risk-aversion cohort
      "investor": 8,    # 2 per risk-aversion cohort
      "merchant": 35,   # unchanged from ROLE_COUNTS, uncohorted
      "institution": 9, # ROLE_COUNTS' 10, minus 1, to keep the total at 100
  }
  ```

  This is a new constant alongside the existing `ROLE_COUNTS` in `src/agents/population.py`, not a replacement — the current 13-cell matrix keeps using `ROLE_COUNTS` unchanged; only the new hypothesis-sims use `HYPOTHESIS_ROLE_COUNTS`. Within each cohorted role, agents are assigned round-robin across the 4 risk-aversion values so each role's own sub-population splits evenly (10 consumers per cohort, 2 banks per cohort, 2 investors per cohort — 14 cohorted agents per risk-aversion level in total).
- **365-day timescale**, matching the doc's explicit "one-year, 365-day" statement and the existing master-cell convention.

## 3. Per-hypothesis sandbox construction

No new currency configs are created. Every hypothesis-sim restricts `env.currencies` to a subset of the **existing 9 real currencies** already loaded by `load_currency_universe()` (`configs/currencies/*.yaml`) — the same mechanism the current sandbox cells use to restrict to 2 symbols, just pointed at real symbols instead of synthetic ones. Their real, as-configured values:

| Symbol | Peg | Governance | Liquidity | Volatility (peg_error) | Issuer risk | Genius compliant |
|---|---|---|---|---|---|---|
| USDC | USD | 0.95 | 0.97 | 0.0003 | 0.05 | true |
| USDT | USD | 0.55 | 0.98 | 0.0008 | 0.25 | false |
| DAI | USD | 0.70 | 0.75 | 0.0030 | 0.20 | false |
| FDUSD | USD | 0.60 | 0.55 | 0.0010 | 0.30 | true |
| TDUSD | USD | 0.90 | 0.35 | 0.0001 | 0.05 | true |
| EURC | EUR | 0.93 | 0.60 | 0.0005 | 0.08 | true |
| EURT | EUR | 0.50 | 0.45 | 0.0012 | 0.28 | false |
| PAXG | XAU | 0.85 | 0.50 | 0.0020 | 0.10 | true |
| XAUT | XAU | 0.65 | 0.40 | 0.0040 | 0.22 | false |

Gas-fee hypotheses (H5, H8, H10, H11) realize "high vs. low gas fees" as **the same real currency-selection principle, offered on two different real chains** (user decision): the cheap reference is **Solana** (`gas_fee=0.002`), the expensive reference is **Ethereum** (`gas_fee=2.50`) — both already defined in `configs/blockchains/`, no new chain configs needed. The "better" trait (higher governance/liquidity, or lower volatility) is always paired with the *expensive* chain and the "worse" trait with the *cheap* chain, so the hypothesis genuinely tests whether a risk-averse agent pays more for the better characteristic — not an arbitrary pairing.

**User decision (2026-08-14): reuse across hypotheses is fine** — several hypotheses land on the same real coin pair because it happens to be the best (or only) available crossover on those dimensions; this is disclosed, not hidden.

| # | Isolates | Real pair | Rationale |
|---|---|---|---|
| H1 | medium of exchange alone | USDC / EURC / PAXG | flagship (most reputable) real coin per peg |
| H2 | governance × medium of exchange (3-way) | USDC/USDT (USD), EURC/EURT (EUR), PAXG/XAUT (gold) | each peg's two real coins split cleanly into a high-governance and low-governance option |
| H3 | governance × liquidity | TDUSD (gov 0.90/liq 0.35) vs USDT (gov 0.55/liq 0.98) | largest real governance/liquidity crossover among USD coins |
| H4 | governance × volatility | DAI (gov 0.70/vol 0.0030) vs USDT (gov 0.55/vol 0.0008) | best available crossover — imperfect, DAI's governance edge over USDT is modest |
| H5 | governance × gas fees | USDC (gov 0.95) on Ethereum vs USDT (gov 0.55) on Solana | |
| H6 | medium of exchange (USD/EUR) × liquidity | USDC (liq 0.97) vs EURC (liq 0.60) | flagship pair already has a strong liquidity gap |
| H7 | medium of exchange (USD/EUR) × volatility | USDC (vol 0.0003) vs EURT (vol 0.0012) | EURC's volatility (0.0005) is too close to USDC's to be a meaningful test; EURT gives a real gap. **This is the one medium-of-exchange hypothesis that does NOT use the USDC/EURC flagship pair.** |
| H8 | medium of exchange (USD/EUR) × gas fees | USDC on Solana vs EURC on Ethereum | flagship pair |
| H9 | liquidity × volatility | TDUSD (liq 0.35/vol 0.0001) vs USDT (liq 0.98/vol 0.0008) | same pair as H3 — TDUSD/USDT are the two most "opposite" real USD coins on several axes at once |
| H10 | liquidity × gas fees | USDT (liq 0.98) on Ethereum vs TDUSD (liq 0.35) on Solana | |
| H11 | volatility × gas fees | TDUSD (vol 0.0001) on Ethereum vs DAI (vol 0.0030) on Solana | |

Unlike the other 9 hypotheses (each a 2-currency `env.currencies` restriction), H1's sandbox restricts `env.currencies` to all 3 of `{USDC, EURC, PAXG}` at once, and H2's restricts to all 6 of `{USDC, USDT, EURC, EURT, PAXG, XAUT}` at once — every hypothesis-sim is still exactly one sandbox with one fixed currency set, just sized to how many real coins that hypothesis's comparison needs.

"Offered on chain X" means: when that hypothesis's `Environment` is constructed, `env.chains` is restricted to just the two chains its two currencies are each assigned to (mirroring how sandbox cells today restrict `env.currencies` to 2 symbols) — a buyer choosing a given currency has no other chain option for it, so the chain's gas fee is effectively that currency's gas fee for this hypothesis-sim.

Wallet seeding for every hypothesis-sim reuses the existing `_seed_sandbox_wallets` pattern (split each agent's pre-existing USD value evenly across the hypothesis's real symbols) unchanged — no new wallet-seeding logic needed.

## 4. Cross-border section

Re-runs **H1, H2, H6, H7, H8 only** (per the doc's explicit priority list), using each hypothesis's sandbox construction from §3 unchanged, but with the existing cross-border mechanism already in the codebase: half the cohorted population assigned `currency_zone="USD"`, half `currency_zone="EUR"` (reusing `generate_agent_population`'s existing zone-splitting logic), and the existing FX conversion tax (`src/economy/fx_tax.py`) applied whenever a buyer settles in a currency outside their zone. This produces 5 hypotheses × 3 utility functions = 15 sims.

## 5. Event-based section

Re-runs **H1, H2, H4, H9** (controller's recommendation, accepted by user): H1 and H2 for their relevance to a banking-crisis flight-to-safety narrative, H4 and H9 for their direct relevance to a depeg event (a depeg is fundamentally a volatility shock). Each of these 4 hypotheses runs under two separate shock scenarios — a depeg event and a banking-crisis event — using the existing shock mechanism (`src/economy/shocks.py`, `ShockType.DEPEG_EVENT` / a banking-crisis-classed shock already defined there) applied to whichever of that hypothesis's two real currencies is the "worse" side (e.g. H4's DAI, H9's USDT — mirroring how shocks already target a specific `target_currency`). This produces 4 hypotheses × 2 events × 3 utility functions = 24 sims.

## 6. Total sim count

11 baseline × 3 + 5 cross-border × 3 + 4 event-based × 2 × 3 = 33 + 15 + 24 = **72 sims total**, each a full 365-day, 100-agent run — a substantial increase from the current 13-cell matrix. This is a direct, necessary consequence of the design decisions above (3 utility functions × 11+5+8 hypothesis-instances), not a padding choice; flagging the scale here since it's a real cost/time consideration for the actual runs, even though the user said not to worry about the doc's computational-limitation section for design purposes.

## 7. Out of scope (this spec)

- Equilibrium-holdings measurement/reporting (sub-project B).
- The equivalence/indifference-search mechanism — i.e., actually varying a currency's characteristic across trials to find where an agent becomes indifferent (sub-project C).
- The end-of-run "would you switch" elicitation question (sub-project D).
- Any change to the existing regression-based econometrics engine (sub-project E).
- Orchestration/runner code that actually launches all 72 sims (a `matrix_runner`-style extension) — this spec defines the sandbox/population *inputs* each sim needs; wiring them into a callable runner is implementation-plan work, not a new design decision, since it follows the existing `_build_cell_specs`/`run_matrix` pattern directly.
- Deleting the old 6 synthetic asset-backing sandbox pairs in `src/currencies/sandbox_currencies.py` — left in place, unused (no hypothesis in the new 11 needs synthetic currencies at all, since §3 uses only real ones).
