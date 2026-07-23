# Phase 2: LLM Negotiation Layer — Design

Status: approved, pending final user sign-off on this document
Date: 2026-07-22
Source specs: `phase_2_instructions_v2.md`, `Agentic AI and Preferences for Medium of Exchange.pdf`, `deep-research-report.md`

## 1. Objective

Phase 1 is a fully deterministic, rule-based multi-agent simulation of AI preferences for
stablecoin media of exchange (utility functions, negotiation, settlement, metrics — all
complete and tested). Phase 2 adds an **LLM-driven decision and negotiation path that runs
alongside Phase 1's deterministic path**, without modifying or weakening it. This lets the
project compare rule-based agents against real LLM-driven agents (via OpenRouter, 5 real
models) under identical economic conditions, and gives a live, model-attributed results
readout, not just a mocked test suite.

The five target hypotheses (risk aversion vs. USD/EUR, liquidity vs. gas fees, governance vs.
liquidity, macro-crisis proximity vs. gold backing, cross-border volatility vs. currency
choice) are the yardstick for every context field and metric added below — nothing is added to
the LLM's context that doesn't serve one of these. Concretely:

| Hypothesis | Independent variable | Dependent variable |
|---|---|---|
| 1. Risk aversion → USD over EUR | risk profile / `risk_aversion` (γ) | USD-pegged vs. EUR-pegged stablecoin selection rate |
| 2. Liquidity vs. gas fees | bid-ask-spread proxy (`liquidity_score`) vs. `gas_fee` | currency/chain choice under a liquidity/fee trade-off |
| 3. Governance vs. liquidity | governance-prompt condition (§12) | USDC vs. USDT selection; governance-attribute trade-off (see tiered outcomes, §12) |
| 4. Crisis/depeg proximity → gold | agent's *perceived* proximity to banking crisis/depeg (vs. objective macro state) | PAXG/XAUT selection rate |
| 5. Cross-border volatility → USD | cross-border flag + FX volatility | USD-pegged vs. EUR-pegged selection rate in cross-border transactions |

Every new context field or metric introduced below should be traceable to one of these five
rows; if it isn't, it doesn't belong in this phase.

**Non-negotiable invariant** (unchanged from `phase_2_instructions_v2.md`):

```
LLM  = reasoning / decision only
Backend = state transition
Ledger = source of truth
```

An LLM call can never mutate a wallet, ledger, or transaction directly. Every LLM output
passes through deterministic validation and settlement exactly like a rule-based agent's
decision does.

## 2. Scope

**Approved exception to the general `experiments/`/`dashboard/` exclusion**: the user has
explicitly authorized editing `experiments/` and `dashboard/` for this phase (previously
off-limits per `phase_2_instructions_v2.md` §2). Phase 2 uses this narrowly: it implements
exactly one experiment end-to-end (`experiment_007_governance_prompting.py`) as the live
integration deliverable. `experiments/008`–`011` and `dashboard/` are untouched unless
requested later.

New/modified files:

```
src/utility/
  risk_neutral.py                (new)
  epstein_zin.py                 (new)
  utility_factory.py             (extended: risk_neutral, epstein_zin_proxy)

src/llm/
  llm_router.py                  (implemented — was NotImplementedError stub)
  agent_reasoning.py             (implemented — was NotImplementedError stub)
  hallucination_detector.py      (implemented — overpayment_pct kept test-compatible)
  decision_schema.py             (new — Decision model + economic validation)
  decision_adapter.py            (new — Decision -> negotiation engine action)
  market_intelligence.py         (new — static profile loader + optional live Polygon price)
  prompts/bank_prompt.txt        (filled in)
  prompts/buyer_prompt.txt       (filled in)
  prompts/investor_prompt.txt    (filled in)
  prompts/seller_prompt.txt      (filled in)

src/negotiation/
  llm_negotiation_engine.py      (new — additive; existing negotiate() untouched)
  llm_offer.py                   (new — immutable offer records w/ previous_offer_id)

src/agents/base_agent.py         (extended: build_llm_context() hook)

configs/llm/models.yaml           (new — model roster + routing policies)
configs/currencies/profiles/*.yaml (new — 9 files, one per existing currency, from
                                     deep-research-report.md)

database/models.py               (extended: LLMDecisionRecord, MarketSnapshotRecord)
database/repository.py           (extended: matching repositories)
metrics/wandb_logger.py          (extended: log_llm_metrics(), additive method)

experiments/experiment_007_governance_prompting.py   (implemented for real)

tests/test_llm_router.py                  (new)
tests/test_agent_reasoning.py             (new)
tests/test_llm_negotiation_engine.py      (new)
tests/test_market_intelligence.py         (new)
tests/test_decision_adapter.py            (new)
tests/test_utility_risk_neutral.py        (new)
tests/test_utility_epstein_zin.py         (new)
tests/test_hallucinations.py              (extended, existing tests untouched)
tests/test_llm_wallet_invariant.py        (new)

pyproject.toml                    (add httpx under a new `llm` optional-dependency group)
.env.example                      (document RUN_LIVE_LLM_TESTS)
```

Explicitly **not** touched: `src/utility/crra.py`, `src/utility/cara.py`,
`src/utility/multi_attribute.py`, `src/negotiation/negotiation_engine.py`,
`src/negotiation/offer.py`, `src/negotiation/counter_offer.py`, `src/simulation/timestep.py`'s
default (rule-based) path, any existing test.

## 3. Utility functions

Two new utility functions join the existing `CRRAUtility` / `CARAUtility` /
`MultiAttributeUtility`, registered in `utility_factory.build_utility_function()`.

### RiskNeutralUtility (`utility_type: risk_neutral`)

A clean, deliberately unshaped baseline:

```
U(option, wealth) = wealth * safety_multiplier - gas_fee
```

where `safety_multiplier = governance_score * liquidity_score * (1 - peg_error)` — the same
multiplier CRRA/CARA already use, so the three functions remain comparable on the same input
surface. Critically, this multiplier is *linear*, not concave — no curvature is smuggled in.
This is the "net economic payoff, no risk preference" baseline the other three are measured
against.

### EpsteinZinProxyUtility (`utility_type: epstein_zin_proxy`)

Named explicitly as a **proxy**, not literal Epstein-Zin. True EZ is a recursive utility over
a consumption stream — it needs a continuation value from future timesteps, which this
simulation's single-period per-transaction decision doesn't have. Presenting it as literal EZ
would misrepresent the math in any resulting paper.

**Mandatory framing, verbatim, in the class docstring and any paper/report text that cites
this function:**

> This is not an Epstein-Zin utility function and should not be interpreted as one in
> empirical results. It is an EZ-inspired static proxy designed to test whether separately
> parameterizing risk aversion and an EIS-inspired fee-sensitivity parameter changes choice
> behavior relative to CRRA.

What it preserves from EZ: **separating risk aversion from elasticity of intertemporal
substitution (EIS)**, which CRRA collapses into one parameter (`γ`). Formula:

```
safety = governance_score * liquidity_score * (1 - peg_error)     # same safety multiplier
fee_sensitivity = 1 / eis                                # eis_inspired_fee_sensitivity; ψ = eis
effective_wealth = max(wealth * safety - gas_fee * fee_sensitivity, 1e-9)

U = log(effective_wealth)                        if risk_aversion == 1
U = effective_wealth ** (1 - risk_aversion) / (1 - risk_aversion)   otherwise
```

`risk_aversion` (γ) shapes curvature over the safety-adjusted payoff — same role as in CRRA.
The external constructor parameter stays named `eis` (spec compatibility with
`phase_2_instructions_v2.md`'s CRRA/CARA/EZ framing), but internally the derived quantity
`1 / eis` is named `eis_inspired_fee_sensitivity`, not `fee_sensitivity` alone — using `ψ = EIS`
to scale a gas-fee penalty is a **behavioral modeling choice**, not a derivation from formal EZ
preferences over consumption, and the name must not imply otherwise. Low `eis` (reluctant to
substitute) amplifies the fee penalty; high `eis` dampens it. These two parameters move
different axes of the same option, which is the behaviorally meaningful part of EZ for this
simulation.

**Independent-testability requirement**: unit tests must show (a) varying γ with ψ fixed
changes curvature over safety differences but not the fee-sensitivity ordering, and (b)
varying ψ with γ fixed changes the utility gap between low-fee and high-fee options but not
the safety-driven ordering. If a future change makes γ and ψ empirically indistinguishable,
that's a regression.

## 4. LLM Router (`src/llm/llm_router.py`)

`httpx`-based OpenRouter client (the one new dependency this phase adds — under a new
`llm` extras group in `pyproject.toml`, mirroring the existing `observability`/`market-data`
pattern). Auth via `OPENROUTER_API_KEY`; the key is never logged (headers are scrubbed before
any log statement).

### Model roster vs. routing policy (`configs/llm/models.yaml`)

Two separate concepts, per point #4 of the design review — the roster is *what's available*,
policy is *how it's used*, so model identity is never implicitly read as "better" than another:

```yaml
models:
  - id: anthropic/claude-sonnet-5
    label: claude-sonnet-5
  - id: openai/gpt-5.6-luna
    label: gpt-5.6-luna
  - id: deepseek/deepseek-v4-pro
    label: deepseek-v4-pro
  - id: google/gemini-3.5-flash-lite
    label: gemini-3.5-flash-lite
  - id: perplexity/sonar
    label: perplexity-sonar

routing_policies:
  default_reliability_chain:
    primary: claude-sonnet-5
    fallbacks: [gpt-5.6-luna, deepseek-v4-pro, gemini-3.5-flash-lite, perplexity-sonar]
  model_comparison:
    pinned_models: [claude-sonnet-5, gpt-5.6-luna, deepseek-v4-pro, gemini-3.5-flash-lite, perplexity-sonar]
```

All 5 slugs are now **verified** against OpenRouter's own model pages (checked 2026-07-22, see
sources below) rather than left as guesses:

| Label | Verified slug | Source |
|---|---|---|
| claude-sonnet-5 | `anthropic/claude-sonnet-5` | openrouter.ai/anthropic/claude-sonnet-5 |
| gpt-5.6-luna | `openai/gpt-5.6-luna` | openrouter.ai/openai/gpt-5.6-luna |
| deepseek-v4-pro | `deepseek/deepseek-v4-pro` | openrouter.ai/deepseek/deepseek-v4-pro |
| gemini-3.5-flash-lite | `google/gemini-3.5-flash-lite` | openrouter.ai/google/gemini-3.5-flash-lite |
| perplexity-sonar | `perplexity/sonar` | openrouter.ai/perplexity/sonar |

Because OpenRouter's catalog can still change after this document is written (models are
deprecated/renamed), verification at design time is not a substitute for a runtime preflight —
the implementation must still validate every configured model ID against OpenRouter at startup
and fail with a clear, labeled error (which model, what OpenRouter said) rather than silently
skipping it or falling through to a different model unannounced.

`default_reliability_chain` is used by ordinary simulation runs: optimizes for getting *a*
decision. `model_comparison` is used by `experiment_007`: each of the 5 models is called on its
own pinned run with **no cross-model substitution** — if a pinned model fails even after
transient retries, that run is marked failed/excluded, not silently completed by a different
model. This keeps "model" a clean experimental factor instead of confounding it with
reliability.

### Three-tier failure handling

Distinct code paths, not one blended retry loop:

| Failure class | Examples | Handling |
|---|---|---|
| Technical | timeout, 429, 500, 502, 503 | exponential backoff retry (configurable attempts) → next model in the active policy's chain. 401/403 abort immediately (a bad key won't fix itself by trying 4 more models). |
| Malformed output | bad JSON, missing required field | one repair re-prompt (send the parse/validation error back) → next model in chain if still malformed |
| Economically invalid | insufficient funds, unsupported currency/chain, non-positive amount/price | **bounded correction loop**: reject, re-prompt the *same* model with the specific validation failure, up to `max_correction_attempts` (default 2) — not unlimited, so a model can't loop forever proposing impossible trades. After the cap, that turn falls to the deterministic fallback (existing `choose_best()`/rule-based `negotiate()`), logged as a decision failure for that turn only. |

`LLMCallResult` (returned by the router) always records what actually happened, never just the
final answer:

```python
class LLMCallResult(BaseModel):
    requested_model: str
    actual_model: str
    fallback_used: bool
    fallback_reason: str | None
    model_attempts: list[str]
    decision: Decision
```

## 5. Agent reasoning & context (`src/llm/agent_reasoning.py`)

`AgentDecisionContext` assembles, per `phase_2_instructions_v2.md` §4B–§4D:

- Agent identity, risk profile, and utility context — `utility_type` + whichever of
  `risk_aversion` / `eis` / multi-attribute weights apply, **always presented in the CRRA/CARA
  numeric framing regardless of which utility function the agent actually runs** (per explicit
  user request), so every model sees a consistent risk-preference vocabulary.
- Currency governance/regulatory attributes (governance score, reserve model, transparency,
  issuer risk, GENIUS Act compliance) — already in `configs/currencies/*.yaml`.
- Market parameters (liquidity, bid-ask spread proxy, gas fee, chain, settlement time, peg
  error, exchange rate).
- Macro state **and** the agent's separately-tracked *perception* of it (objective vs.
  perceived proximity to banking crisis / depegging — required for hypothesis 4; perception
  starts equal to objective state with optional configurable noise/staleness).
- Transaction context: domestic vs. cross-border, and for cross-border, origin/destination
  currency, exchange rate, and exchange-rate volatility (hypothesis 5).
- Opponent's current offer + conversation history so far.
- **Market intelligence** (new, §6 below): static curated profile + optional live price
  snapshot per candidate currency.

`build_llm_context()` on `BaseAgent` assembles the agent-side fields (risk profile, utility
params, wallet balances, preferences/memory) into a plain context object — `src/llm` depends on
`src/agents` for this hook (the instructions explicitly authorize adding hooks to
`base_agent.py`), but `src/llm/agent_reasoning.py`'s core functions take plain data, not
`BaseAgent` instances, matching this codebase's existing pattern of layers depending on plain
values rather than on each other's classes (e.g. `routing_engine.generate_candidates` takes
balances, not a `Wallet`).

Prompt templates (`src/llm/prompts/*.txt`, one per agent class) are rendered with Python's
built-in `str.format`-style substitution — no new templating dependency.

## 6. Market intelligence (`src/llm/market_intelligence.py`)

Two clearly separate sources, addressing the reproducibility gap in the design review:

**Static profile corpus** (primary source, reproducible): `deep-research-report.md`'s 9
sections are pre-compiled into `configs/currencies/profiles/{SYMBOL}.yaml` — one file per
existing currency config (`DAI`, `EURC`, `EURT`, `FDUSD`, `PAXG`, `Tokenized_Deposits`, `USDC`,
`USDT`, `XAUT`). Each holds: `executive_summary`, `timeline` (list of `{date, event}`),
`reserves_and_transparency`, `governance`, `price_and_market_cap`, `crra_cara_note`,
`use_cases`, `regulatory_and_controversies`, `source`, `report_date` ("2026-07"). This is
git-versioned and identical on every run — no cross-run drift from "the news changed since
yesterday."

**Live price snapshot** (optional, supplementary): `Polygon_API_KEY`-backed fetch of current
crypto aggregate price for a ticker (e.g. `X:USDCUSD`), cached per process run to avoid
redundant calls. Every snapshot is a `MarketSnapshotRecord`: `retrieval_timestamp`, `source`,
`ticker`, `price`, `data_window`, persisted alongside the run it was used in — so a later re-run
can see exactly what price data the LLM was shown, rather than silently re-fetching different
numbers. If Polygon errors or has no data for a symbol, the context explicitly says
"live price unavailable" (never silently substitutes zero) and the decision pipeline proceeds
on the static profile alone — a market-data outage must never crash a negotiation.

The static profile is injected into the prompt under an explicit **"Background / historical
information — not current market state"** heading, distinct from the live price snapshot and
the simulation's own current governance/liquidity/peg-error fields. Without this label an LLM
could treat a mid-2026 reserve-composition snapshot or a stale news event as describing the
simulation's present moment, which would quietly corrupt the very governance/liquidity
comparisons hypotheses 1–3 depend on.

## 7. Structured decision & adapter

`src/llm/decision_schema.py` — `Decision` (action ∈ {OFFER, COUNTER_OFFER, ACCEPT, REJECT,
WALK_AWAY}, proposed_currency, proposed_chain, amount, price, reasoning, plus optional
confidence/utility_estimate/risk_assessment/preferred_alternative_currency/chain). `reasoning`
is the spec-mandated wire field name; downstream, `LLMDecisionRecord` persists it as
`reported_reasoning` to keep it epistemically honest — it's the model's self-report, not
verified causal evidence, and quantitative hypothesis tests must key off observed choices
(currency/chain/price), not this text.

`src/llm/decision_adapter.py` sits between the raw `Decision` and the negotiation engine:
validates currency/chain compatibility, amount/price positivity, and available funds (the
"economically invalid" tier from §4), and converts a valid `Decision` into the negotiation
engine's internal action type. This keeps the LLM-specific schema out of the negotiation state
machine's internals.

## 8. LLM-driven negotiation engine (`src/negotiation/llm_negotiation_engine.py`)

Additive — `src/negotiation/negotiation_engine.py`'s rule-based `negotiate()` is untouched and
remains the deterministic baseline/comparison path.

`NegotiationSession` tracks the full state from `phase_2_instructions_v2.md` §5:
`negotiation_id`, `buyer_id`, `seller_id`, `current_round`, `max_rounds`, `status` (IN_PROGRESS
/ ACCEPTED / REJECTED / WALKED_AWAY / MAX_ROUNDS_REACHED / FAILED_VALIDATION), `initial_offer`,
`current_offer`, `current_currency`, `current_blockchain`, `conversation_history`,
`created_at`, `completed_at`. New `LLMOffer` (in `llm_offer.py`) is immutable and carries
`offer_id` / `previous_offer_id` / `agent_id` / `action` / `reasoning` / `timestamp`, distinct
from the existing rule-based `Offer` so nothing here can break Phase 1's tested negotiation
path. A hard `max_rounds` cap guarantees termination, same guarantee the rule-based engine
already provides.

## 9. Hallucination detection (`src/llm/hallucination_detector.py`)

`overpayment_pct(expected, paid)` is implemented **exactly** as the existing (already
committed) test in `tests/test_hallucinations.py` requires: signed percentage
`(paid - expected) / expected * 100`, raising `ValueError` when `expected <= 0`. This is a
pre-existing test contract and is not renegotiable.

`HallucinationResult` adds the derived fields the design review asked for, on top of (not
instead of) that signed value: `absolute_error` (`abs(paid - expected)`), `percentage_error`
(`abs(overpayment_pct)`), and `direction` (`OVERPAYMENT` / `UNDERPAYMENT` / `ACCURATE`, via a
configurable `hallucination_threshold`, default 0.20, never hardcoded). Correlated fields:
`currency_symbol`, `chain_name`, `requested_model`, `actual_model`, `agent_type`,
`risk_profile`, `economic_scenario`.

## 10. Persistence & W&B

`LLMDecisionRecord` (new table) — for every LLM decision: `decision_id`, `simulation_id`,
`timestep`, `agent_id`, `agent_type`, `requested_model`, `actual_model`, `fallback_used`,
`fallback_reason`, `model_attempts` (JSON), `prompt_version`, `rendered_prompt_hash`, `action`,
`currency`, `chain`, `amount`, `price`, `reported_reasoning`, `negotiation_id`, `round`,
`risk_profile`, `utility_type`, `utility_parameters` (JSON), `scenario`,
`domestic_or_cross_border`, `governance_prompt_enabled`.

**Prompt versioning is two fields, not one**, because "which template" and "what exact text was
sent" are different reproducibility questions:

- `prompt_version` — identifies the *template*: `{prompt_name}@{semantic_version}` (e.g.
  `buyer_prompt@v1`), bumped manually whenever a `src/llm/prompts/*.txt` file changes. Answers
  "which version of the prompt design was this?"
- `rendered_prompt_hash` — a `sha256` of the *exact rendered prompt text* sent for this specific
  call (template + that call's context values substituted in). Answers "what did the model
  literally receive?" A model comparison is only reproducible if both are known — two calls
  with the same `prompt_version` can still have different `rendered_prompt_hash`es because the
  context (wallet balance, macro state, opponent offer) differs call to call, and that's
  expected, not a bug.

`MarketSnapshotRecord` (new table) — `retrieval_timestamp`, `source`, `ticker`, `price`,
`data_window`, linked to the simulation/negotiation it informed.

The existing `HallucinationRecord` (already in `database/models.py`, already has a
`model_name` field) is populated for the first time — extended additively only if a field is
genuinely missing after implementation, not assumed in advance.

`metrics/wandb_logger.py`'s `WandbRunLogger` gets one additive method, `log_llm_metrics()`,
called only by the LLM path; the existing `on_timestep()` used by Phase 1 runs is untouched.

## 11. Testing strategy

All OpenRouter/Polygon HTTP calls are mocked in the default test suite (`httpx.MockTransport`
or equivalent — no real network in a normal `pytest` run). Covers: success, bad JSON, missing
fields, timeout, rate limit, retry, fallback (both technical and economic-invalidity tiers),
auth failure, and the wallet-mutation invariant (an LLM decision can never touch a `Wallet`
before deterministic settlement runs). One `@pytest.mark.live` test, skipped unless
`RUN_LIVE_LLM_TESTS=1`, makes a single real OpenRouter call as an opt-in smoke check —
documented in `.env.example`.

New tests for the utility functions explicitly verify γ/ψ independence (§3) and that
`RiskNeutralUtility` contains no hidden curvature (§3).

## 12. Live-results deliverable: `experiment_007_governance_prompting.py`

A controlled comparison, not just a demo:

- **Independent variables**: governance-prompting condition (baseline vs. governance-emphasized
  prompt) × model (5, pinned via the `model_comparison` routing policy — no substitution).
- **Held constant**: agent profile, risk parameters, market state, available currencies,
  transaction opportunities, random seed, opponent behavior.
- **Dependent variables, explicitly tiered by evidential strength** (per the design review —
  self-reported reasoning is not proof the model weighted governance, only choice behavior is):
  - *Primary outcome* (observed choice, strong evidence): USDC vs. USDT selection rate.
  - *Secondary behavioral outcome* (observed choice under a controlled trade-off scenario,
    strong evidence): liquidity sacrificed — i.e. cases where the model picks the
    better-governed/lower-liquidity option over the more-liquid/worse-governed one — and
    negotiation outcome/hallucination rate.
  - *Exploratory outcome* (self-report, weak evidence, reported separately and never merged
    into the primary/secondary numbers): rate of `reported_reasoning` mentioning
    governance/compliance terms. Used only to generate qualitative discussion, never as
    quantitative proof of *why* a choice was made.
- Runs real negotiations through the LLM layer end-to-end, settles deterministically, logs to
  the DB (`LLMDecisionRecord`, `MarketSnapshotRecord`) and W&B (`log_llm_metrics()`), and prints
  a plain-English results table keyed by (model × condition).
- `experiments/008`–`011` remain stubs; only `007` is implemented this phase.

## 13. Open risk / assumptions carried into the implementation plan

- All 5 OpenRouter model slugs are now verified (§4 table, checked 2026-07-22) — this closes
  the design review's mandatory item #1. The residual risk is only that OpenRouter's catalog
  can change *after* this document is written, which is why the runtime preflight check (§4)
  is still mandatory, not optional hardening.
- Polygon.io's crypto ticker coverage for the specific stablecoins in scope (esp.
  gold-backed/tokenized-deposit tickers) is unverified; the graceful-degradation path is not
  optional polish, it's load-bearing.

## 14. Sources for model-slug verification (2026-07-22)

- [Claude Sonnet 5 - API Pricing & Benchmarks | OpenRouter](https://openrouter.ai/anthropic/claude-sonnet-5)
- [GPT-5.6 Luna - API Pricing & Benchmarks | OpenRouter](https://openrouter.ai/openai/gpt-5.6-luna)
- [DeepSeek V4 Pro - API Pricing & Benchmarks | OpenRouter](https://openrouter.ai/deepseek/deepseek-v4-pro)
- [Gemini 3.5 Flash Lite - API Pricing & Benchmarks | OpenRouter](https://openrouter.ai/google/gemini-3.5-flash-lite)
- [Sonar - API Pricing & Benchmarks | OpenRouter](https://openrouter.ai/perplexity/sonar)
