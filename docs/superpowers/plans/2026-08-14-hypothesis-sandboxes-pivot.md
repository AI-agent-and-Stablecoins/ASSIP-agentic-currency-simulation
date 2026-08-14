# Hypothesis Sandboxes Pivot Implementation Plan (Sub-Project A: Core Mechanism)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three self-contained mechanism pieces the approved spec (`docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-pivot-design.md`) needs: (1) the ability to pin a specific currency to a specific chain so gas-fee hypotheses actually force a tradeoff, (2) a population generator that produces the new 4-fixed-risk-aversion-cohort structure under one forced utility function, and (3) the data defining all 24 hypothesis-cell configurations (11 baseline + 5 cross-border + 8 event-based). Each hypothesis-sim can then be run directly via `Environment.build_from_population` + a manual day loop (the same pattern existing tests already use), without needing the full `run_matrix` batch-orchestration machinery — that wiring is an explicitly separate, later plan (per user decision during brainstorming: this scope is safer and fully satisfies what the spec requires).

**Architecture:** No new currencies, no new chains, no schema changes. `generate_candidates` (`src/blockchain/routing_engine.py`) gains an optional `currency_chain_pins` parameter; `Environment` gains a `currency_chain_pins` attribute the two `timestep.py` call sites read. `src/agents/population.py` gains `generate_hypothesis_population` alongside (not replacing) `generate_agent_population`. A new `src/economy/hypothesis_scenarios.py` module holds the 11 hypotheses' currency/chain-pin data and builds the 24 cell specs.

**Tech Stack:** Python 3.12, pydantic 2.x, pytest. No new dependencies.

## Global Constraints

- Follow the spec exactly: `docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-pivot-design.md`.
- No new currency or chain configs — every hypothesis uses only the 9 existing real currencies (`configs/currencies/*.yaml`) and 2 existing chains (Ethereum, Solana).
- `generate_candidates`'s default behavior (no `currency_chain_pins` argument) must stay byte-for-byte identical to today — every existing caller passes nothing new and must see no change.
- `generate_agent_population` (the existing function) is untouched — `generate_hypothesis_population` is a new, separate function.
- Epstein-Zin runs use `eis=1.0` (the codebase's existing "neutral" reference value, per `tests/test_utility_epstein_zin.py`'s own convention) — not a new number invented for this plan.
- No comments beyond what the codebase already uses at each touched call site.
- All new tests follow the existing style in `tests/test_routing_engine.py` / `tests/test_population.py` (plain `assert`, no docstrings on trivial tests, one behavior per test).

---

### Task 1: Chain-pinning support in `generate_candidates`

**Files:**
- Modify: `src/blockchain/routing_engine.py` (`generate_candidates`)
- Modify: `src/simulation/environment.py` (`Environment.__init__`)
- Modify: `src/simulation/timestep.py` (both `generate_candidates` call sites)
- Test: `tests/test_routing_engine.py`, `tests/test_simulation.py`

**Interfaces:**
- Produces: `generate_candidates(..., currency_chain_pins: dict[str, str] | None = None)` — when a symbol is a key in `currency_chain_pins`, it only ever pairs with that one chain name; every other symbol pairs with every chain, unchanged. `Environment.currency_chain_pins: dict[str, str]` (default `{}`), settable after construction exactly like `_seed_sandbox_wallets` already mutates a constructed `Environment`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_routing_engine.py`:

```python
def test_generate_candidates_pins_a_currency_to_its_assigned_chain():
    currencies = load_currency_universe()
    chains = load_chain_universe()

    candidates = generate_candidates(
        {"USDC": 100.0, "USDT": 100.0},
        currencies,
        chains,
        currency_chain_pins={"USDC": "ethereum", "USDT": "solana"},
    )

    usdc_chains = {c.chain_name for c in candidates if c.currency_symbol == "USDC"}
    usdt_chains = {c.chain_name for c in candidates if c.currency_symbol == "USDT"}
    assert usdc_chains == {"ethereum"}
    assert usdt_chains == {"solana"}


def test_generate_candidates_leaves_unpinned_currencies_on_every_chain():
    currencies = load_currency_universe()
    chains = load_chain_universe()

    candidates = generate_candidates(
        {"USDC": 100.0, "USDT": 100.0},
        currencies,
        chains,
        currency_chain_pins={"USDC": "ethereum"},
    )

    usdt_chains = {c.chain_name for c in candidates if c.currency_symbol == "USDT"}
    assert usdt_chains == set(chains.keys())


def test_generate_candidates_default_behavior_is_unchanged_without_pins():
    currencies = load_currency_universe()
    chains = load_chain_universe()

    candidates = generate_candidates({"USDC": 100.0}, currencies, chains)

    assert {c.chain_name for c in candidates} == set(chains.keys())
```

Add to `tests/test_simulation.py` (near the other `Environment`/`run_timestep` tests):

```python
def test_run_timestep_respects_currency_chain_pins():
    env = Environment.build("baseline", {"consumer": 1, "merchant": 1})
    env.currency_chain_pins = {"USDC": "solana"}
    consumer = next(a for a in env.agents.values() if a.agent_class == "buyer")
    consumer.wallet.balances = {"USDC": 1000.0}
    rng = random.Random(0)

    result = run_timestep(env, day=0, rng=rng)

    settled = [tx for tx in result.transactions if tx.status == TransactionStatus.SETTLED]
    assert all(tx.chain_name == "solana" for tx in settled if tx.currency_symbol == "USDC")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_routing_engine.py -k pins -v`
Expected: FAIL — `TypeError: generate_candidates() got an unexpected keyword argument 'currency_chain_pins'`.

- [ ] **Step 3: Add the parameter to `generate_candidates`**

In `src/blockchain/routing_engine.py`, change:

```python
def generate_candidates(
    available_balances: dict[str, float],
    currencies: dict[str, CurrencyConfig],
    chains: dict[str, ChainConfig],
    liquidity_pools: LiquidityPoolRegistry | None = None,
    trust_ledger: TrustLedger | None = None,
) -> list[CurrencyChainOption]:
    """One candidate per (currency the agent holds a positive balance of) x (chain)."""
    liquidity_pools = liquidity_pools or LiquidityPoolRegistry()
    options: list[CurrencyChainOption] = []
    for symbol, balance in available_balances.items():
        if balance <= 0 or symbol not in currencies:
            continue
        currency = currencies[symbol]
        if trust_ledger is not None:
            peg_error = trust_ledger.effective_peg_error(symbol, currency.peg_error)
        else:
            peg_error = currency.peg_error
        for chain in chains.values():
```

to:

```python
def generate_candidates(
    available_balances: dict[str, float],
    currencies: dict[str, CurrencyConfig],
    chains: dict[str, ChainConfig],
    liquidity_pools: LiquidityPoolRegistry | None = None,
    trust_ledger: TrustLedger | None = None,
    currency_chain_pins: dict[str, str] | None = None,
) -> list[CurrencyChainOption]:
    """One candidate per (currency the agent holds a positive balance of) x (chain),
    unless currency_chain_pins restricts a currency to exactly one chain."""
    liquidity_pools = liquidity_pools or LiquidityPoolRegistry()
    currency_chain_pins = currency_chain_pins or {}
    options: list[CurrencyChainOption] = []
    for symbol, balance in available_balances.items():
        if balance <= 0 or symbol not in currencies:
            continue
        currency = currencies[symbol]
        if trust_ledger is not None:
            peg_error = trust_ledger.effective_peg_error(symbol, currency.peg_error)
        else:
            peg_error = currency.peg_error
        pinned_chain = currency_chain_pins.get(symbol)
        candidate_chains = [chains[pinned_chain]] if pinned_chain is not None else list(chains.values())
        for chain in candidate_chains:
```

- [ ] **Step 4: Add `currency_chain_pins` to `Environment`**

In `src/simulation/environment.py`, change:

```python
        self.previous_real_purchasing_power: dict[str, float] = {}
```

to:

```python
        self.previous_real_purchasing_power: dict[str, float] = {}
        self.currency_chain_pins: dict[str, str] = {}
```

- [ ] **Step 5: Pass `env.currency_chain_pins` through both call sites**

In `src/simulation/timestep.py`, change (the `_process_buyer_llm_day` call site):

```python
        candidates = generate_candidates(
            buyer.wallet.balances,
            env.currencies,
            env.chains,
            env.liquidity_pools,
            trust_ledger=env.trust_ledger,
        )
        if not candidates:
            continue

        spread_optimal_currency, spread_optimal_chain, gas_optimal_currency, gas_optimal_chain = (
            _spread_and_gas_optimal(candidates)
        )
```

to:

```python
        candidates = generate_candidates(
            buyer.wallet.balances,
            env.currencies,
            env.chains,
            env.liquidity_pools,
            trust_ledger=env.trust_ledger,
            currency_chain_pins=env.currency_chain_pins,
        )
        if not candidates:
            continue

        spread_optimal_currency, spread_optimal_chain, gas_optimal_currency, gas_optimal_chain = (
            _spread_and_gas_optimal(candidates)
        )
```

And change (the deterministic-path call site):

```python
                candidates = generate_candidates(
                    buyer.wallet.balances,
                    env.currencies,
                    env.chains,
                    env.liquidity_pools,
                    trust_ledger=env.trust_ledger,
                )
                if not candidates:
                    continue
```

to:

```python
                candidates = generate_candidates(
                    buyer.wallet.balances,
                    env.currencies,
                    env.chains,
                    env.liquidity_pools,
                    trust_ledger=env.trust_ledger,
                    currency_chain_pins=env.currency_chain_pins,
                )
                if not candidates:
                    continue
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_routing_engine.py tests/test_simulation.py -k "pins or chain_pins" -v`
Expected: PASS — all 4 new tests.

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS, full suite green (same pass count as before this task plus 4).

- [ ] **Step 8: Commit**

```bash
git add src/blockchain/routing_engine.py src/simulation/environment.py src/simulation/timestep.py tests/test_routing_engine.py tests/test_simulation.py
git commit -m "feat: support pinning a currency to one chain in generate_candidates"
```

---

### Task 2: `generate_hypothesis_population`

**Files:**
- Modify: `src/agents/population.py`
- Test: `tests/test_population.py`

**Interfaces:**
- Consumes: `build_agent(profile, *, currency_zone=None, assigned_model=None, cara_override: tuple[str, float | None] | None = None, agent_id=None)` (`src/agents/agent_factory.py`, already exists — `cara_override=(utility_type, risk_aversion)` works for any `utility_type` string, not just `"cara"`).
- Produces: `HYPOTHESIS_ROLE_COUNTS: dict[str, int]`, `RISK_AVERSION_COHORTS: list[float]`, `HYPOTHESIS_EIS: float`, `generate_hypothesis_population(seed: int, model_candidates: list[str], utility_type: str) -> list[BaseAgent]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_population.py`:

```python
from src.agents.population import generate_hypothesis_population


def test_hypothesis_population_generates_exactly_100_agents():
    population = generate_hypothesis_population(seed=0, model_candidates=CANDIDATE_MODELS, utility_type="crra")
    assert len(population) == 100


def test_hypothesis_population_role_composition_matches_spec():
    population = generate_hypothesis_population(seed=0, model_candidates=CANDIDATE_MODELS, utility_type="crra")

    counts = {}
    for agent in population:
        counts[agent.profile_name] = counts.get(agent.profile_name, 0) + 1

    assert counts == {"consumer": 40, "merchant": 35, "bank": 8, "investor": 8, "institution": 9}


def test_hypothesis_population_cohorts_are_exactly_a_0_2_4_6():
    population = generate_hypothesis_population(seed=0, model_candidates=CANDIDATE_MODELS, utility_type="crra")

    cohorted = [a for a in population if a.profile_name in ("consumer", "bank", "investor")]
    assert len(cohorted) == 56
    risk_aversions = [a.risk_aversion for a in cohorted]
    assert sorted(set(risk_aversions)) == [0.0, 2.0, 4.0, 6.0]
    for level in (0.0, 2.0, 4.0, 6.0):
        assert risk_aversions.count(level) == 14


def test_hypothesis_population_forces_the_requested_utility_type_on_cohorted_agents():
    population = generate_hypothesis_population(seed=0, model_candidates=CANDIDATE_MODELS, utility_type="cara")

    cohorted = [a for a in population if a.profile_name in ("consumer", "bank", "investor")]
    assert all(a.utility_type == "cara" for a in cohorted)

    non_cohorted = [a for a in population if a.profile_name in ("merchant", "institution")]
    assert all(a.utility_type == "multi_attribute" for a in non_cohorted)


def test_hypothesis_population_supports_epstein_zin_proxy():
    from src.utility.epstein_zin import EpsteinZinProxyUtility

    population = generate_hypothesis_population(
        seed=0, model_candidates=CANDIDATE_MODELS, utility_type="epstein_zin_proxy"
    )

    cohorted = [a for a in population if a.profile_name in ("consumer", "bank", "investor")]
    assert all(a.utility_type == "epstein_zin_proxy" for a in cohorted)
    assert all(isinstance(a.utility_fn, EpsteinZinProxyUtility) for a in cohorted)


def test_hypothesis_population_rejects_an_unknown_utility_type():
    with pytest.raises(ValueError):
        generate_hypothesis_population(seed=0, model_candidates=CANDIDATE_MODELS, utility_type="not_a_real_type")


def test_hypothesis_population_same_seed_is_reproducible():
    population_a = generate_hypothesis_population(seed=5, model_candidates=CANDIDATE_MODELS, utility_type="crra")
    population_b = generate_hypothesis_population(seed=5, model_candidates=CANDIDATE_MODELS, utility_type="crra")

    ids_a = [a.agent_id for a in population_a]
    ids_b = [a.agent_id for a in population_b]
    risk_a = [a.risk_aversion for a in population_a]
    risk_b = [a.risk_aversion for a in population_b]
    assert ids_a == ids_b
    assert risk_a == risk_b
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_population.py -k hypothesis_population -v`
Expected: FAIL — `ImportError: cannot import name 'generate_hypothesis_population'`.

- [ ] **Step 3: Implement `generate_hypothesis_population`**

In `src/agents/population.py`, add after the existing `generate_agent_population` function:

```python
HYPOTHESIS_ROLE_COUNTS = {
    "consumer": 40,
    "bank": 8,
    "investor": 8,
    "merchant": 35,
    "institution": 9,
}

RISK_AVERSION_COHORTS = [0.0, 2.0, 4.0, 6.0]

HYPOTHESIS_EIS = 1.0

HYPOTHESIS_UTILITY_TYPES = {"crra", "cara", "epstein_zin_proxy"}


def generate_hypothesis_population(seed: int, model_candidates: list[str], utility_type: str) -> list[BaseAgent]:
    if not model_candidates:
        raise ValueError("generate_hypothesis_population requires at least one verified model candidate")
    if utility_type not in HYPOTHESIS_UTILITY_TYPES:
        raise ValueError(f"utility_type must be one of {HYPOTHESIS_UTILITY_TYPES}, got {utility_type!r}")

    rng = random.Random(seed)
    profiles = load_agent_profiles()

    total_agents = sum(HYPOTHESIS_ROLE_COUNTS.values())
    zones = ["USD"] * (total_agents // 2) + ["EUR"] * (total_agents // 2)
    rng.shuffle(zones)

    shuffled_models = list(model_candidates)
    rng.shuffle(shuffled_models)

    population: list[BaseAgent] = []
    slot_index = 0
    for profile_name, count in HYPOTHESIS_ROLE_COUNTS.items():
        profile = profiles[profile_name]
        if profile_name in CARA_ELIGIBLE_ROLES:
            profile = profile.model_copy(update={"eis": HYPOTHESIS_EIS}) if utility_type == "epstein_zin_proxy" else profile
            cohort_assignment = [RISK_AVERSION_COHORTS[i % len(RISK_AVERSION_COHORTS)] for i in range(count)]
            rng.shuffle(cohort_assignment)

        for i in range(count):
            cara_override = (utility_type, cohort_assignment[i]) if profile_name in CARA_ELIGIBLE_ROLES else None

            assigned_model = shuffled_models[slot_index % len(shuffled_models)]
            deterministic_id = f"{profile_name}-seed{seed}-{slot_index:03d}"
            agent = build_agent(
                profile,
                currency_zone=zones[slot_index],
                assigned_model=assigned_model,
                cara_override=cara_override,
                agent_id=deterministic_id,
            )
            population.append(agent)
            slot_index += 1

    return population
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_population.py -k hypothesis_population -v`
Expected: PASS — all 7 new tests.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS, full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/agents/population.py tests/test_population.py
git commit -m "feat: add generate_hypothesis_population with fixed risk-aversion cohorts"
```

---

### Task 3: Hypothesis cell-spec data

**Files:**
- Create: `src/economy/hypothesis_scenarios.py`
- Test: `tests/test_hypothesis_scenarios.py`

**Interfaces:**
- Consumes: nothing new (pure data + one pure function).
- Produces: `HypothesisCellSpec` (a frozen dataclass: `hypothesis: str`, `currencies: tuple[str, ...]`, `chain_pins: dict[str, str] | None`, `cross_border: bool`, `event_shock: str | None`, `event_target_currency: str | None`), `HYPOTHESIS_CURRENCIES: dict[str, tuple[str, ...]]`, `build_hypothesis_cell_specs() -> list[HypothesisCellSpec]`. A later, separate plan (run_matrix wiring) consumes `build_hypothesis_cell_specs()`'s output.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hypothesis_scenarios.py`:

```python
from src.economy.hypothesis_scenarios import (
    CROSS_BORDER_HYPOTHESES,
    EVENT_BASED_HYPOTHESES,
    HYPOTHESIS_CURRENCIES,
    build_hypothesis_cell_specs,
)


def test_all_11_hypotheses_have_currency_definitions():
    assert set(HYPOTHESIS_CURRENCIES.keys()) == {f"H{i}" for i in range(1, 12)}


def test_h1_isolates_medium_of_exchange_alone_with_three_currencies():
    assert HYPOTHESIS_CURRENCIES["H1"] == ("USDC", "EURC", "PAXG")


def test_h2_isolates_governance_by_medium_of_exchange_with_six_currencies():
    assert set(HYPOTHESIS_CURRENCIES["H2"]) == {"USDC", "USDT", "EURC", "EURT", "PAXG", "XAUT"}


def test_two_currency_hypotheses_have_exactly_two_currencies():
    for hypothesis in ("H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"):
        assert len(HYPOTHESIS_CURRENCIES[hypothesis]) == 2


def test_total_cell_count_is_24():
    specs = build_hypothesis_cell_specs()
    assert len(specs) == 24


def test_11_baseline_cells_have_no_cross_border_or_event_shock():
    specs = build_hypothesis_cell_specs()
    baseline = [s for s in specs if not s.cross_border and s.event_shock is None]
    assert len(baseline) == 11
    assert {s.hypothesis for s in baseline} == {f"H{i}" for i in range(1, 12)}


def test_5_cross_border_cells_match_the_spec_priority_list():
    specs = build_hypothesis_cell_specs()
    cross_border = [s for s in specs if s.cross_border]
    assert len(cross_border) == 5
    assert {s.hypothesis for s in cross_border} == set(CROSS_BORDER_HYPOTHESES)
    assert set(CROSS_BORDER_HYPOTHESES) == {"H1", "H2", "H6", "H7", "H8"}


def test_8_event_based_cells_are_4_hypotheses_times_2_shocks():
    specs = build_hypothesis_cell_specs()
    event_based = [s for s in specs if s.event_shock is not None]
    assert len(event_based) == 8
    assert {s.hypothesis for s in event_based} == set(EVENT_BASED_HYPOTHESES)
    assert set(EVENT_BASED_HYPOTHESES) == {"H1", "H2", "H4", "H9"}
    shock_types = {s.event_shock for s in event_based}
    assert shock_types == {"depeg", "banking_crisis"}
    for hypothesis in EVENT_BASED_HYPOTHESES:
        this_hypothesis_shocks = {s.event_shock for s in event_based if s.hypothesis == hypothesis}
        assert this_hypothesis_shocks == {"depeg", "banking_crisis"}


def test_event_based_cells_target_a_currency_actually_in_that_hypothesis():
    specs = build_hypothesis_cell_specs()
    for spec in specs:
        if spec.event_shock is not None:
            assert spec.event_target_currency in spec.currencies


def test_gas_fee_hypotheses_have_chain_pins_covering_both_currencies():
    specs = build_hypothesis_cell_specs()
    for spec in specs:
        if spec.hypothesis in ("H5", "H8", "H10", "H11") and not spec.cross_border and spec.event_shock is None:
            assert spec.chain_pins is not None
            assert set(spec.chain_pins.keys()) == set(spec.currencies)
            assert set(spec.chain_pins.values()) == {"ethereum", "solana"}


def test_non_gas_fee_hypotheses_have_no_chain_pins():
    specs = build_hypothesis_cell_specs()
    for spec in specs:
        if spec.hypothesis not in ("H5", "H8", "H10", "H11"):
            assert spec.chain_pins is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_hypothesis_scenarios.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.economy.hypothesis_scenarios'`.

- [ ] **Step 3: Implement `src/economy/hypothesis_scenarios.py`**

```python
"""The 11 new hypotheses' sandbox definitions, per
docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-pivot-design.md.
Every hypothesis uses only real currencies (configs/currencies/*.yaml) --
no synthetic currencies, per that spec's explicit user decision. Gas-fee
hypotheses (H5, H8, H10, H11) additionally pin each currency to one real
chain via generate_candidates' currency_chain_pins, so the "better" trait
is always the one that costs more gas -- otherwise an agent could pick the
better currency on the cheaper chain and the tradeoff the hypothesis exists
to test would never actually be forced.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HypothesisCellSpec:
    hypothesis: str
    currencies: tuple[str, ...]
    chain_pins: dict[str, str] | None = None
    cross_border: bool = False
    event_shock: str | None = None
    event_target_currency: str | None = None


HYPOTHESIS_CURRENCIES: dict[str, tuple[str, ...]] = {
    "H1": ("USDC", "EURC", "PAXG"),
    "H2": ("USDC", "USDT", "EURC", "EURT", "PAXG", "XAUT"),
    "H3": ("TDUSD", "USDT"),
    "H4": ("DAI", "USDT"),
    "H5": ("USDC", "USDT"),
    "H6": ("USDC", "EURC"),
    "H7": ("USDC", "EURT"),
    "H8": ("USDC", "EURC"),
    "H9": ("TDUSD", "USDT"),
    "H10": ("USDT", "TDUSD"),
    "H11": ("TDUSD", "DAI"),
}

HYPOTHESIS_CHAIN_PINS: dict[str, dict[str, str]] = {
    "H5": {"USDC": "ethereum", "USDT": "solana"},
    "H8": {"USDC": "solana", "EURC": "ethereum"},
    "H10": {"USDT": "ethereum", "TDUSD": "solana"},
    "H11": {"TDUSD": "ethereum", "DAI": "solana"},
}

CROSS_BORDER_HYPOTHESES = ("H1", "H2", "H6", "H7", "H8")

EVENT_BASED_HYPOTHESES = ("H1", "H2", "H4", "H9")

EVENT_TARGET_CURRENCY: dict[str, str] = {
    "H1": "USDC",
    "H2": "USDT",
    "H4": "DAI",
    "H9": "USDT",
}


def build_hypothesis_cell_specs() -> list[HypothesisCellSpec]:
    specs: list[HypothesisCellSpec] = []

    for hypothesis, currencies in HYPOTHESIS_CURRENCIES.items():
        specs.append(
            HypothesisCellSpec(
                hypothesis=hypothesis,
                currencies=currencies,
                chain_pins=HYPOTHESIS_CHAIN_PINS.get(hypothesis),
            )
        )

    for hypothesis in CROSS_BORDER_HYPOTHESES:
        specs.append(
            HypothesisCellSpec(
                hypothesis=hypothesis,
                currencies=HYPOTHESIS_CURRENCIES[hypothesis],
                chain_pins=HYPOTHESIS_CHAIN_PINS.get(hypothesis),
                cross_border=True,
            )
        )

    for hypothesis in EVENT_BASED_HYPOTHESES:
        for shock in ("depeg", "banking_crisis"):
            specs.append(
                HypothesisCellSpec(
                    hypothesis=hypothesis,
                    currencies=HYPOTHESIS_CURRENCIES[hypothesis],
                    chain_pins=HYPOTHESIS_CHAIN_PINS.get(hypothesis),
                    event_shock=shock,
                    event_target_currency=EVENT_TARGET_CURRENCY[hypothesis],
                )
            )

    return specs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_hypothesis_scenarios.py -v`
Expected: PASS — all 11 tests.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS, full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/economy/hypothesis_scenarios.py tests/test_hypothesis_scenarios.py
git commit -m "feat: add hypothesis cell-spec data for the 11-hypothesis pivot"
```
