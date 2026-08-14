# Equivalence Framework — Design Spec (Pivot Sub-Project C, incorporating D's elicitation primitive)

## 0. Why this spec exists

Continuing the research-methodology pivot from `New info.pdf` (see `docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-pivot-design.md` §0 for the full A/B/C/D/E decomposition). Sub-project A (hypothesis-sandbox mechanism) and B (equilibrium-holdings measurement, H1's table type) are already built and merged.

This spec covers **C: the equivalence/indifference-search framework** — the doc's H3-H11 "compensation" table type (e.g. "a risk-neutral agent needs a 0.01% better bid-ask spread to switch from a high-governance coin to a low-governance coin"). **User decision (2026-08-14): building C requires building D's elicitation question as its foundational primitive** — the search IS repeated calls to "would you switch under this specified change," so this spec builds both together; D as a separately-named future piece would only mean "also expose the single-question version standalone," which isn't needed yet.

**Out of scope for this spec:** the existing econometrics engine's fate (E), and wiring hypothesis-sims into `run_matrix`'s persisted/checkpointed batch machinery (deferred from sub-project A — this framework runs against a live, already-completed `Environment`, same as sub-project B).

## 1. Scope: H2-H11, not H1

**User decision (2026-08-14):** H1 (medium of exchange alone) is excluded — H1's actual result is the risk-aversion-driven holdings shift (more USD as risk aversion increases), already fully captured by sub-project B's `holdings_by_cohort`. H1 has no doc-given compensation example and needs no equivalence search of its own.

## 2. Mechanism: one completed sim, then a cheap post-hoc question sequence

**User decision (2026-08-14), following the doc's own framing** ("this question could be asked at the end of the 365-day simulation rather than after every transaction"): the expensive part (365 days of transacting/learning, already built by sub-project A) happens once per (hypothesis × utility function). The search over compensation values is a cheap follow-up against each agent's final, already-learned state — NOT a full separate 365-day re-run per candidate value. Each search trial costs one extra LLM call per agent, not a new simulation.

**Binding prerequisite (already recorded, sub-project A):** the underlying 365-day sim MUST have run with `use_llm=True` — the deterministic path cannot differentiate risk-aversion cohorts at all, so a search built on a deterministic-path sim would search over meaningless, identical-across-cohorts responses.

## 3. Which characteristic is held fixed vs. searched, per hypothesis

Every hypothesis's real currencies (already chosen in sub-project A, `HYPOTHESIS_CURRENCIES` in `src/economy/hypothesis_scenarios.py`) supply the coin identities. The search always **fixes the "low-trait" coin's second-dimension value at its real config value** (Y, the doc's own convention) and **varies the "high-trait" coin's second-dimension value** (X) across a bounded range until the agent's answer flips — X−Y is the reported compensation.

Because H2 needs two separate searches (vs.-EUR and vs.-gold — see below) while every other hypothesis needs exactly one, this is modeled as data, one row per *comparison* (not one row per hypothesis), mirroring sub-project A's `HypothesisCellSpec`/`build_hypothesis_cell_specs()` pattern:

| Hypothesis | Fixed coin (Y, real value) | Varied coin (X, searched) | Varied field | Bounds |
|---|---|---|---|---|
| H3 | USDT `liquidity_score` (0.98) | TDUSD `liquidity_score` | `liquidity_score` | `[0.0, 1.0]` |
| H4 | USDT `peg_error` (0.0008) | DAI `peg_error` | `peg_error` | `[0.0, 0.05]` |
| H5 | USDT's chain (Solana) `gas_fee` (0.002) | USDC's chain (Ethereum) `gas_fee` | `gas_fee` | `[0.0, 5.0]` |
| H6 | EURC `liquidity_score` (0.60) | USDC `liquidity_score` | `liquidity_score` | `[0.0, 1.0]` |
| H7 | EURT `peg_error` (0.0012) | USDC `peg_error` | `peg_error` | `[0.0, 0.05]` |
| H8 | EURC's chain (Ethereum) `gas_fee` (2.50) | USDC's chain (Solana) `gas_fee` | `gas_fee` | `[0.0, 5.0]` |
| H9 | TDUSD `peg_error` (0.0001) | USDT `peg_error` | `peg_error` | `[0.0, 0.05]` |
| H10 | TDUSD's chain (Solana) `gas_fee` (0.002) | USDT's chain (Ethereum) `gas_fee` | `gas_fee` | `[0.0, 5.0]` |
| H11 | DAI's chain (Solana) `gas_fee` (0.002) | TDUSD's chain (Ethereum) `gas_fee` | `gas_fee` | `[0.0, 5.0]` |
| H2 (vs. EUR) | USDT `governance_score` (0.55) | EURC `governance_score` | `governance_score` | `[0.0, 1.0]` |
| H2 (vs. gold) | USDT `governance_score` (0.55) | PAXG `governance_score` | `governance_score` | `[0.0, 1.0]` |

For H3/H4/H9 (governance/liquidity roles), "high-trait" is the coin favored by that hypothesis's named preference (H3/H4: governance → TDUSD/DAI; H9: liquidity → USDT), and its OWN second dimension is what gets searched, while the other coin's corresponding value is the fixed reference (Y) — consistently "vary the valued trait's own coin on its other dimension," per the doc's own worked example (§0).

**"Fixed at its real config value"** means literally that coin's real, unmodified value — e.g. H3's Y = USDT's real `liquidity_score` (0.98). **"Varied"** means constructing a modified copy of the searched coin's `CurrencyConfig` (via `.model_copy(update={...})`) or, for gas-fee rows, a modified `ChainConfig` copy, at each binary-search trial — never mutating the real config objects sub-project A already loads.

New module `src/economy/equivalence_framework.py` holds this table as data:

```python
@dataclass(frozen=True)
class EquivalenceComparison:
    hypothesis: str
    fixed_currency: str
    varied_currency: str
    varied_field: str  # "liquidity_score" | "peg_error" | "gas_fee"
    bounds: tuple[float, float]
```

When `varied_field == "gas_fee"`, the modification target is `varied_currency`'s assigned chain (per its `HYPOTHESIS_CHAIN_PINS` entry from sub-project A), not the currency itself — `gas_fee` lives on `ChainConfig`, not `CurrencyConfig`. Every other `varied_field` value modifies `varied_currency`'s own `CurrencyConfig` directly.

## 4. The elicitation question: a new, parallel LLM call path

No existing infrastructure fits: `src/llm/llm_router.py`'s `call_model(prompt, model_id, client, retry_config) -> Decision` is hardcoded to return a negotiation `Decision` (offer/accept/reject fields), and `src/llm/agent_reasoning.py`'s `render_prompt`/`build_decision_context` are built around the much richer negotiation context (candidates, live prices, currency profiles, conversation history) a yes/no switch question doesn't need. This spec adds a parallel, simpler path rather than overloading either.

**New response schema**, `src/llm/switch_elicitation.py`:

```python
class SwitchDecision(BaseModel):
    will_switch: bool
    reasoning: str
```

**New prompt template**, `src/llm/prompts/switch_question_prompt.txt` — describes the agent's current holdings (from `agent.build_llm_context()`, already exists) and the two coins being compared (their real/modified characteristic values, per §3), and asks the agent whether it would switch. Reuses the existing 3-tier failure handling pattern (`src/llm/llm_router.py`'s technical-failure retry + malformed-output repair) via a new `call_model_for_switch(prompt, model_id, client, retry_config) -> SwitchDecision`, mirroring `call_model`'s structure with a different response type and no economic-validity tier (a yes/no question has no wallet/currency/chain constraint to violate the way a negotiation `Decision` does).

## 5. Per-agent binary search, then cohort mean

For one agent, one `EquivalenceComparison`, one utility-function environment: binary search over `comparison.bounds`, asking `call_model_for_switch` at the midpoint each round, narrowing toward "yes" or "no" based on the answer, for a **fixed 7 rounds** (bounded LLM-call cost per agent; 2⁻⁷ ≈ 1/128 of the initial range's width, finer than the doc's own worked-example precision of 0.01%). The agent's own indifference point is the final round's midpoint; the reported compensation is that midpoint minus `comparison.fixed_currency`'s real reference value (X−Y).

The cohort's reported compensation value is the **mean of its 14 agents' individual indifference points** — the same per-agent-then-cohort-average pattern sub-project B's `holdings_by_cohort` already established.

New function, alongside the `EquivalenceComparison` dataclass in `src/economy/equivalence_framework.py`:

```python
def cohort_indifference_points(env: Environment, comparison: EquivalenceComparison) -> dict[float, float]:
    """Returns {risk_aversion_cohort: mean_X_minus_Y} for one comparison
    (per §3's table), across env's cohorted agents. Callers testing H2
    call this twice — once per its two EquivalenceComparison rows."""
```

## 6. Testing

- Binary search converges to the correct value against a fake `call_model_for_switch` that deterministically answers "yes" above/below a known threshold (no real LLM call).
- Each of the 10 hypotheses' fixed/varied assignment (§3's table) is asserted directly, matching `HYPOTHESIS_CURRENCIES`.
- Cohort aggregation is the correct arithmetic mean across a cohort's agents (same style of test as sub-project B's).
- H2's governance-variant path is exercised distinctly from the other 9 hypotheses' liquidity/volatility/gas-fee variants.
- One real end-to-end test using a mock OpenRouter client (matching `tests/llm_test_helpers.py`'s existing `mock_openrouter_client` pattern) proves the full path — real environment, real agents, a canned `SwitchDecision` response — without a real network call.

## 7. Out of scope (this spec)

- H1's measurement (already sub-project B).
- The existing econometrics engine's fate (E).
- Runner-wiring into `run_matrix` (deferred from sub-project A).
- Persisting search results to the database — this spec returns plain in-memory dicts, matching sub-project B's `holdings_by_cohort` convention (no persistence layer exists yet for hypothesis-sims).
