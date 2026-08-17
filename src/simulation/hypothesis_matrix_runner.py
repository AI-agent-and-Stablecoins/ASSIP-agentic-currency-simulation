"""The hypothesis-sandbox matrix runner: drives `build_hypothesis_cell_specs()`'s
24 `HypothesisCellSpec`s x 3 utility functions (`HYPOTHESIS_UTILITY_TYPES`) x
`seeds` through a real day-loop, persisting every day via the existing
`persist_full_timestep`, then running the appropriate post-run analysis
(`holdings_by_cohort` for H1, `cohort_indifference_points` for H2-H11) and
persisting those results to `CohortHoldingsRecord`/`IndifferencePointRecord`.

See docs/superpowers/specs/2026-08-14-runner-wiring-design.md for the full
design. This module is a parallel, standalone runner next to
`src/simulation/matrix_runner.py`'s 13-cell path -- it does not modify that
module's behavior at all, only reuses its generic `CrossZoneMarketplace` and
three of its module-private helpers (`_resolve_available_models`,
`_SHARED_CONFIG_PATHS`, `_CURRENCY_UNIVERSE_PATHS`) rather than duplicating
them.

`use_llm` is unconditionally `True` for every day-loop call: per sub-project
A's binding decision (see `src/economy/hypothesis_scenarios.py`'s module
docstring), CRRA/CARA/EpsteinZinProxy utility are monotone transforms of one
wealth scalar under the deterministic path, so only the LLM path actually
varies its answer by risk-aversion cohort. There is no `dry_run`/
`exercise_llm_path` duality here (unlike `run_matrix`): `openrouter_client` is
a required parameter with no default, so the caller (real run or test) always
decides explicitly what it points at.

Model-candidate verification: `available_models` is resolved once per call via
`matrix_runner._resolve_available_models(model_candidates, openrouter_client)`,
exactly like `run_matrix`'s `dry_run=False` path -- since `openrouter_client`
is never `None` here, this always performs the real OpenRouter `/models`
preflight check. A caller supplying a mock client (tests) must therefore mock
that endpoint too, alongside `/chat/completions` -- see
`tests/test_hypothesis_matrix_runner.py`'s own combined mock client.

Checkpointing/resume follows `run_matrix`'s exact contract (see that module's
docstring's "Memory, progress, and per-cell error recovery" /
`checkpoint_dir` sections), with two deliberate differences:

1. A checkpoint (`next_day=0`, zeroed counters) is written immediately after
   the `SimulationRunRecord` commit, before day 0 even starts -- not only per
   day thereafter. Without this, a crash during day 0's own `run_timestep`/
   `persist_full_timestep` (this cell's very first network I/O, so the single
   likeliest moment for a transient failure across the 72 cell/utility_type
   combinations this runner drives) would leave a committed
   `SimulationRunRecord` with NO checkpoint at all -- exactly what the
   skip-check above reads as "fully done" -- silently and permanently
   skipping that cell/seed/utility_type on every later call.
2. The checkpoint is NOT deleted the moment the day loop finishes -- it stays
   alive until the post-run analysis phase (holdings/indifference-point
   search) also commits successfully. The analysis phase has no
   checkpointing of its own (per the design spec Sec 0's third binding
   decision, no second checkpointing concept was added for it); instead, a
   crash during it leaves the day loop's own checkpoint in place (at
   `next_day == num_days`), so the next call resumes into an
   already-exhausted day range (a no-op) and simply retries the analysis
   phase.

Together these two mean the `SimulationRunRecord`-exists-with-no-checkpoint
skip-check only ever fires once registration, every simulated day, AND the
analysis phase have all durably committed -- a crash at any other point
resumes from a checkpoint instead of being silently skipped. (One narrower
gap remains, shared with `run_matrix`'s identical pattern and not introduced
here: a crash strictly between one day's `persist_full_timestep` commit and
that same day's `save_checkpoint` call leaves a checkpoint pointing at an
already-persisted day, so resuming re-attempts it and hits a composite-key
`IntegrityError` on `timestep_logs`/`agent_states` -- recovering that
specific cell/seed/utility_type currently requires deleting its rows by hand
before retrying.)
"""

import random
import traceback
import warnings
from pathlib import Path
from typing import Callable

import httpx
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.models import SimulationRunRecord
from database.repository import (
    CohortHoldingsLogEntry,
    CohortHoldingsRepository,
    IndifferencePointLogEntry,
    IndifferencePointRepository,
    SimulationRunLogEntry,
    SimulationRunRepository,
    persist_full_timestep,
)
from src.agents.base_agent import BaseAgent
from src.agents.population import HYPOTHESIS_UTILITY_TYPES, generate_hypothesis_population
from src.currencies.currency import CurrencyConfig, load_currency_universe
from src.currencies.synthetic_hypothesis_currencies import (
    BID_ASK_SPREAD_LEVELS,
    GAS_FEE_LEVELS,
    GOVERNANCE_LEVELS,
    NEUTRAL_FIXED_VALUES,
    SYNTHETIC_CHAINS,
    SYNTHETIC_DIMENSION_PAIRS,
    VOLATILITY_LEVELS,
)
from src.economy.equilibrium_holdings import holdings_by_cohort
from src.economy.equivalence_framework import EQUIVALENCE_COMPARISONS, cohort_indifference_points
from src.economy.hypothesis_scenarios import (
    HYPOTHESIS_CURRENCIES,
    HypothesisCellSpec,
    _EVENT_DAY,
    build_hypothesis_cell_specs,
    scenario_for,
)
from src.economy.shocks import ScenarioConfig, load_scenario
from src.economy.synthetic_hypothesis_scenarios import (
    SyntheticHypothesisCellSpec,
    build_synthetic_hypothesis_cell_specs,
)
from src.economy.synthetic_switch_search import SyntheticEquivalenceComparison, cohort_discrete_switch_points
from src.economy.wallet_seeding import seed_restricted_wallets
from src.simulation.checkpointing import (
    CellSeedCheckpoint,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.simulation.environment import Environment
from src.simulation.matrix_runner import (
    MASTER_SCENARIO_NAME,
    CrossZoneMarketplace,
    _CURRENCY_UNIVERSE_PATHS,
    _SHARED_CONFIG_PATHS,
    _prompt_version_hash,
    _resolve_available_models,
)
from src.simulation.provenance import compute_config_hash, compute_git_commit_hash, model_roster_summary_for
from src.simulation.timestep import run_timestep


class HypothesisCellResult(BaseModel):
    """One hypothesis cell/seed/utility_type combination's outcome -- mirrors
    `matrix_runner.MatrixCellResult`'s shape, plus the extra `hypothesis`/
    `utility_type` axes this runner has, plus whichever of
    `holdings_by_cohort`'s or `cohort_indifference_points`' post-run analysis
    result this cell produced.

    For `track="real"`, only one of `cohort_holdings`/`cohort_indifference`
    is ever populated, per `spec.hypothesis == "H1"` or not. For
    `track="synthetic"`, EVERY hypothesis (H1-H11) populates
    `cohort_holdings` (per design spec §6, "every hypothesis, not just H1"),
    and every hypothesis except H1 ALSO populates `cohort_indifference` (via
    the discrete switch search) -- so both fields can be populated
    simultaneously for a synthetic H2-H11 cell.

    `cohort_indifference` is keyed by `EquivalenceComparison.varied_currency`
    (not `hypothesis`, since H2 is the one hypothesis with two comparisons
    sharing a `hypothesis` value, distinguished only by `varied_currency` --
    EURC vs. PAXG) -- for every other real-track hypothesis, and for every
    synthetic-track hypothesis (which has exactly one comparison each), this
    dict has exactly one key.
    """

    run_id: str
    cell_key: str
    hypothesis: str
    seed: int
    utility_type: str
    is_cross_border: bool
    num_days_completed: int = 0
    total_transactions: int = 0
    total_llm_decisions: int = 0
    cohort_holdings: dict[float, dict[str, float]] | None = None
    cohort_indifference: dict[str, dict[float, float]] | None = None


def _build_fresh_cell_environment(
    spec: HypothesisCellSpec | SyntheticHypothesisCellSpec,
    utility_type: str,
    seed: int,
    available_models: list[str],
    real_currency_universe: dict[str, CurrencyConfig],
    base_scenario: ScenarioConfig,
    track: str = "real",
) -> tuple[Environment, list[BaseAgent]]:
    """Population -> restricted-universe Environment -> chain-pin wiring ->
    wallet seeding -> (optional) cross-border marketplace swap, for one fresh
    (non-resumed) cell/seed/utility_type. Extracted from `run_hypothesis_matrix`
    so it -- and, in particular, the `env.currency_chain_pins` wiring below --
    can be exercised directly by a small, targeted unit test rather than only
    through the full runner.

    `track="real"` (the default) is the original, unchanged real-coin path:
    `spec` is a `HypothesisCellSpec`, currencies come from restricting
    `real_currency_universe` down to `HYPOTHESIS_CURRENCIES[spec.hypothesis]`,
    the scenario may be an event variant (`scenario_for`), chain pins default
    to `{}` when absent, and a cross-border spec swaps in
    `CrossZoneMarketplace`.

    `track="synthetic"`: `spec` is a `SyntheticHypothesisCellSpec` whose
    `.currencies` IS the currency universe for that cell already (no
    restriction-into-a-larger-universe step) -- every synthetic coin always
    carries a chain pin (`spec.chain_pins`, no `or {}` fallback needed), there
    is no cross-border or event variant for this track (baseline-only per
    design spec §8), so the scenario is always `base_scenario` as-is.

    `env.currency_chain_pins = spec.chain_pins or {}` (real track) is the
    missing wiring the design spec's Sec 4 surfaced: without it, H5/H8/H10/H11's
    chain-pinning (the whole point of those four hypotheses) silently never
    takes effect. Set unconditionally (not just for chain-pinned specs) since
    `{}` is the correct, harmless value for every other real-track hypothesis.
    """
    population = generate_hypothesis_population(seed, available_models, utility_type)

    if track == "synthetic":
        restricted_currencies = spec.currencies
        cell_scenario = base_scenario
    else:
        restricted_currencies = {
            symbol: real_currency_universe[symbol] for symbol in HYPOTHESIS_CURRENCIES[spec.hypothesis]
        }
        cell_scenario = scenario_for(spec, base_scenario)

    env = Environment.build_from_population(
        MASTER_SCENARIO_NAME, population, currencies=restricted_currencies, scenario=cell_scenario
    )
    if track == "synthetic":
        # Environment.build_from_population always loads the REAL chain
        # universe (load_chain_universe()'s ethereum/arbitrum/base/solana) --
        # it has no hook for a caller-supplied chain set. Every synthetic
        # coin is pinned to one of Task 2's 3 synthetic chains
        # (synthetic_gas_low/mid/high), which don't exist in that real
        # universe at all, so without this merge every candidate-generation
        # lookup of a synthetic-pinned chain (`env.chains[pinned_chain]`)
        # would KeyError. Merging (not replacing) keeps the real chains
        # present too -- harmless, since no synthetic-track currency is ever
        # pinned to one of them.
        env.chains = {**env.chains, **SYNTHETIC_CHAINS}
    env.currency_chain_pins = spec.chain_pins if track == "synthetic" else (spec.chain_pins or {})
    seed_restricted_wallets(env.agents, restricted_currencies, real_currency_universe, env.macro_state.peg_reference_rates)
    if track == "real" and spec.cross_border:
        env.marketplace = CrossZoneMarketplace(env.agents)
    return env, population


# ---------------------------------------------------------------------------
# Synthetic-track discrete compensation-search comparison construction.
#
# `docs/superpowers/plans/2026-08-15-synthetic-coin-track.md`'s Task 5
# interface sketch named a `synthetic_equivalence_comparisons_for` function
# that Task 5's implementer did not end up building (see
# `src/economy/synthetic_switch_search.py`'s module docstring/exports) --
# this Task 6 wiring builds that comparison-construction logic itself, here,
# since it is the one place that needs to translate a hypothesis's tested
# dimension pair into one concrete (fixed_currency, varied_currency,
# varied_field, levels) tuple.
#
# FIXED (post-whole-plan-review): an earlier revision made fixed_currency and
# varied_currency AGREE on the non-swept ("differentiator") dimension and
# differ only on the swept one -- which measures no trade-off at all (the two
# coins were identical on every dimension except the one being searched, so
# there was nothing for the searched dimension to compensate FOR). The
# correct construction, matching the real-coin track's own natural example
# (H3: fixed_currency=USDT is low-governance/high-liquidity, varied_currency=
# TDUSD is high-governance/low-liquidity -- two DIFFERENT real coins sitting
# at opposite corners of the governance x liquidity grid) is:
#   - `differentiator_dim`: the dimension fixed_currency and varied_currency
#     DIFFER on -- this is the trait being "compensated for". If "medium"
#     (peg zone) is one of the two tested dimensions, it is ALWAYS the
#     differentiator (medium is categorical, not one of
#     `SyntheticEquivalenceComparison.varied_field`'s four supported numeric
#     fields, so it can never be the swept dimension) -- a deliberate
#     departure from "first tuple entry = differentiator" for H2 only
#     (`SYNTHETIC_DIMENSION_PAIRS["H2"] == ("governance", "medium")`).
#     Otherwise (both tested dimensions numeric), the FIRST tuple entry is
#     the differentiator.
#   - `swept_dim`: the OTHER tested dimension -- what `comparison.levels`
#     sweeps. Always numeric.
#   - `fixed_currency` = the coin at (differentiator = its LOW/worst value,
#     swept_dim = its BEST value) -- a coin that's bad on the differentiator
#     but otherwise excellent (its swept-dim value becomes `fixed_value`, the
#     Y reference the compensation is measured against).
#   - `varied_currency` = the coin at (differentiator = its HIGH/best value,
#     swept_dim = its WORST value) -- a coin that's good on the
#     differentiator but starts out with the flaw being compensated for.
#     Note `varied_currency`'s own swept-dim value is never actually read by
#     `_agent_discrete_switch_point` (each of `comparison.levels` overrides
#     it in turn) -- starting it at "worst" is for a clean, human-readable
#     comparison identity (mirroring TDUSD's real, genuinely-low liquidity),
#     not a computational necessity.
#   - `medium`'s two differentiator values, when it's the differentiator:
#     `NEUTRAL_FIXED_VALUES["medium"]` ("USD", the low/reference side) and
#     `_MEDIUM_ALTERNATE_LEVEL` ("EUR", the high/alternate side) -- a
#     categorical zone choice has no natural high/low ordering, so this is a
#     deliberate, fixed, documented convention, not a discovered result.
# ---------------------------------------------------------------------------

_MEDIUM_ALTERNATE_LEVEL = "EUR"

_SYNTHETIC_DIM_TO_FIELD: dict[str, str] = {
    "governance": "governance_score",
    "liquidity": "bid_ask_spread",
    "volatility": "peg_error",
    "gas_fee": "gas_fee",
}
_SYNTHETIC_DIM_LEVELS: dict[str, tuple[float, ...]] = {
    "governance": GOVERNANCE_LEVELS,
    "liquidity": BID_ASK_SPREAD_LEVELS,
    "volatility": VOLATILITY_LEVELS,
    "gas_fee": GAS_FEE_LEVELS,
}
_SYNTHETIC_DIM_HIGHER_IS_BETTER: dict[str, bool] = {
    "governance": True,
    "liquidity": False,
    "volatility": False,
    "gas_fee": False,
}


def _synthetic_dim_value(
    currency: CurrencyConfig, symbol: str, chain_pins: dict[str, str], dimension: str
) -> float | str:
    if dimension == "medium":
        return currency.peg
    if dimension == "gas_fee":
        return SYNTHETIC_CHAINS[chain_pins[symbol]].gas_fee
    return getattr(currency, _SYNTHETIC_DIM_TO_FIELD[dimension])


def _synthetic_equivalence_comparison_for(
    hypothesis: str, currencies: dict[str, CurrencyConfig], chain_pins: dict[str, str]
) -> SyntheticEquivalenceComparison:
    """Builds the ONE `SyntheticEquivalenceComparison` for `hypothesis` (H2-H11
    only -- H1 is medium-alone, holdings-only, and must not be passed here).
    See this module's judgment-call comment above for the fixed/varied coin
    selection rule (opposite corners of the two tested dimensions)."""
    dims = SYNTHETIC_DIMENSION_PAIRS[hypothesis]

    if "medium" in dims:
        differentiator_dim = "medium"
        swept_dim = dims[0] if dims[1] == "medium" else dims[1]
        low_differentiator_value: float | str = NEUTRAL_FIXED_VALUES["medium"]
        high_differentiator_value: float | str = _MEDIUM_ALTERNATE_LEVEL
    else:
        differentiator_dim, swept_dim = dims
        differentiator_levels = _SYNTHETIC_DIM_LEVELS[differentiator_dim]
        differentiator_higher_is_better = _SYNTHETIC_DIM_HIGHER_IS_BETTER[differentiator_dim]
        low_differentiator_value = (
            min(differentiator_levels) if differentiator_higher_is_better else max(differentiator_levels)
        )
        high_differentiator_value = (
            max(differentiator_levels) if differentiator_higher_is_better else min(differentiator_levels)
        )

    varied_field = _SYNTHETIC_DIM_TO_FIELD[swept_dim]
    swept_levels = _SYNTHETIC_DIM_LEVELS[swept_dim]
    swept_higher_is_better = _SYNTHETIC_DIM_HIGHER_IS_BETTER[swept_dim]
    best_swept = max(swept_levels) if swept_higher_is_better else min(swept_levels)
    worst_swept = min(swept_levels) if swept_higher_is_better else max(swept_levels)

    fixed_currency: str | None = None
    varied_currency: str | None = None
    for symbol, currency in currencies.items():
        differentiator_value = _synthetic_dim_value(currency, symbol, chain_pins, differentiator_dim)
        swept_value = _synthetic_dim_value(currency, symbol, chain_pins, swept_dim)
        if differentiator_value == low_differentiator_value and swept_value == best_swept:
            fixed_currency = symbol
        elif differentiator_value == high_differentiator_value and swept_value == worst_swept:
            varied_currency = symbol

    if fixed_currency is None or varied_currency is None:
        raise ValueError(
            f"could not locate a (fixed_currency, varied_currency) pair for {hypothesis!r}'s synthetic "
            "equivalence comparison -- this indicates a mismatch between this function's dimension logic "
            "and build_synthetic_hypothesis_currencies' actual grid"
        )

    return SyntheticEquivalenceComparison(
        hypothesis=hypothesis,
        fixed_currency=fixed_currency,
        varied_currency=varied_currency,
        varied_field=varied_field,
        levels=tuple(swept_levels),
    )


def run_hypothesis_matrix(
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    openrouter_client: httpx.Client,
    session: Session,
    matrix_run_id: str,
    polygon_client: httpx.Client | None = None,
    utility_types: list[str] | None = None,
    hypotheses: list[str] | None = None,
    cell_keys: list[str] | None = None,
    progress_callback: Callable[[str, int, str, int], None] | None = None,
    checkpoint_dir: Path | None = None,
    llm_max_workers: int = 1,
    track: str = "real",
) -> tuple[list[HypothesisCellResult], list[tuple[str, int, str, Exception]]]:
    """Run every requested hypothesis-sandbox cell x utility_type x seed for
    `num_days` days each, persisting every day via `persist_full_timestep`
    plus a post-run analysis phase persisted to
    `CohortHoldingsRecord`/`IndifferencePointRecord`.

    `track` selects which of the two parallel hypothesis-sandbox tracks to
    run -- `"real"` (the default, unchanged) uses real stablecoins via
    `build_hypothesis_cell_specs()`, with `holdings_by_cohort` for H1 and
    `cohort_indifference_points` (continuous binary search) for H2-H11.
    `"synthetic"` uses `build_synthetic_hypothesis_cell_specs()`'s fully
    controlled coin grids instead: EVERY hypothesis (H1-H11) gets
    `holdings_by_cohort`, and every hypothesis except H1 ALSO gets
    `cohort_discrete_switch_points` (discrete-level search) against one
    `SyntheticEquivalenceComparison` built from
    `SYNTHETIC_DIMENSION_PAIRS[hypothesis]` -- per
    docs/superpowers/specs/2026-08-15-synthetic-coin-track-design.md §6.
    Any other value raises `ValueError`.

    Returns `(results, failures)`. `failures` is a list of
    `(cell_key, seed, utility_type, exception)` for any cell/seed/utility_type
    whose day loop OR post-run analysis phase raised; every other combination
    still runs. `failures` is empty on a fully clean run.

    `openrouter_client` is required (no default, never `None` -- raises
    `ValueError` if it is): every day-loop call here uses `use_llm=True`
    unconditionally (see this module's docstring), and the post-run
    indifference-point search (H2-H11) also needs a real client. There is no
    `dry_run`/`exercise_llm_path` concept for this runner.

    `utility_types`, if `None` (the default), runs all of
    `HYPOTHESIS_UTILITY_TYPES`. `hypotheses`, if `None` (the default), runs
    every spec from `build_hypothesis_cell_specs()`; otherwise restricts to
    specs whose `.hypothesis` is in the list (raises `ValueError` if any
    requested hypothesis matches no spec). `cell_keys`, if given, further
    restricts to specs whose `.key` is in the list (raises `ValueError` if
    any requested key matches no spec) -- use this to select baseline-only,
    cross-border-only, or event-only cells (`hypotheses` alone can't: it
    selects a whole hypothesis's baseline + cross-border + event variants
    together). Both filters apply together (AND) when both are given.
    `src.economy.hypothesis_scenarios.baseline_cell_keys()` returns every
    baseline (non-cross-border, non-event) key for a "baseline model only"
    run.

    `checkpoint_dir`/resume semantics and `progress_callback` timing exactly
    mirror `run_matrix`'s (see that module's docstring), with the extra
    `utility_type` axis threaded through `run_id`:
    `f"{matrix_run_id}-{spec.key}-{utility_type}-seed{seed}"` for
    `track="real"` (byte-for-byte the original shape, already live in
    production databases), or
    `f"{matrix_run_id}-{track}-{spec.key}-{utility_type}-seed{seed}"` for
    `track="synthetic"` (a distinguishing segment, since this track has no
    pre-existing data to stay compatible with). `progress_callback(cell_key,
    seed, utility_type, day)` and `failures`' tuple shape are unaffected.
    """
    if openrouter_client is None:
        raise ValueError(
            "run_hypothesis_matrix requires a real (or test-mock) openrouter_client -- there is no dry_run "
            "concept for this runner, since every hypothesis-sim day-loop call uses use_llm=True unconditionally."
        )

    if track not in ("real", "synthetic"):
        raise ValueError(f"track must be 'real' or 'synthetic', got {track!r}")

    available_models = _resolve_available_models(model_candidates, openrouter_client)

    git_commit_hash = compute_git_commit_hash()
    prompt_version_hash = _prompt_version_hash()
    config_hash = compute_config_hash(_SHARED_CONFIG_PATHS + _CURRENCY_UNIVERSE_PATHS)
    real_currency_universe = load_currency_universe()
    base_scenario = load_scenario(MASTER_SCENARIO_NAME)

    all_specs: list[HypothesisCellSpec] | list[SyntheticHypothesisCellSpec]
    if track == "synthetic":
        all_specs = build_synthetic_hypothesis_cell_specs()
    else:
        all_specs = build_hypothesis_cell_specs()
    if hypotheses is not None:
        unknown = set(hypotheses) - {spec.hypothesis for spec in all_specs}
        if unknown:
            raise ValueError(f"hypotheses contains unknown hypothesis id(s): {sorted(unknown)}")
        specs_to_run = [spec for spec in all_specs if spec.hypothesis in hypotheses]
    else:
        specs_to_run = all_specs

    if cell_keys is not None:
        unknown_keys = set(cell_keys) - {spec.key for spec in all_specs}
        if unknown_keys:
            raise ValueError(f"cell_keys contains unknown cell key(s): {sorted(unknown_keys)}")
        specs_to_run = [spec for spec in specs_to_run if spec.key in cell_keys]

    # An event-based spec's shock is scheduled at day _EVENT_DAY -- a
    # num_days short of that fires it never, producing an "event" cell
    # indistinguishable from (in fact strictly milder than) its own baseline,
    # silently and misleadingly. A warning, not a ValueError: short num_days
    # is legitimate for fast tests exercising an event-based hypothesis's
    # OTHER wiring (chain pins, currency restriction, persistence) without
    # caring whether the event itself fires -- this only needs to catch a
    # real research run's config mistake, not block every short test run.
    event_specs_too_short = [
        spec for spec in specs_to_run if getattr(spec, "event_shock", None) is not None and num_days <= _EVENT_DAY
    ]
    if event_specs_too_short:
        warnings.warn(
            f"num_days={num_days} is too short for {len(event_specs_too_short)} selected event-based cell(s) "
            f"(e.g. {event_specs_too_short[0].key!r}) whose shock fires at day {_EVENT_DAY} -- their event will "
            "never trigger, so that cell's results will be indistinguishable from its own baseline.",
            stacklevel=2,
        )

    if utility_types is not None:
        unknown = set(utility_types) - set(HYPOTHESIS_UTILITY_TYPES)
        if unknown:
            raise ValueError(f"utility_types contains unknown utility type id(s): {sorted(unknown)}")
        resolved_utility_types = list(utility_types)
    else:
        resolved_utility_types = sorted(HYPOTHESIS_UTILITY_TYPES)

    results: list[HypothesisCellResult] = []
    failures: list[tuple[str, int, str, Exception]] = []

    for utility_type in resolved_utility_types:
        for spec in specs_to_run:
            for seed in seeds:
                # track="real" keeps the exact pre-existing run_id shape (no
                # track segment) -- this run_id scheme is already live in
                # production databases (including an active real-money study
                # in progress), and inserting a segment unconditionally here
                # broke every downstream reporting/resume lookup that still
                # builds the old shape (src/reporting/hypothesis_tables.py's
                # _run_id, scripts/generate_hypothesis_report.py) -- silently
                # returning empty tables and risking duplicate re-runs on
                # resume. Only the brand-new synthetic track, which has no
                # existing data to preserve compatibility with, gets the
                # distinguishing segment.
                run_id = (
                    f"{matrix_run_id}-{spec.key}-{utility_type}-seed{seed}"
                    if track == "real"
                    else f"{matrix_run_id}-{track}-{spec.key}-{utility_type}-seed{seed}"
                )

                checkpoint = load_checkpoint(checkpoint_dir, run_id) if checkpoint_dir is not None else None

                if (
                    checkpoint is None
                    and checkpoint_dir is not None
                    and session.get(SimulationRunRecord, run_id) is not None
                ):
                    # Already fully completed in an earlier call -- see
                    # run_matrix's identical guard for the full rationale.
                    continue

                if checkpoint is not None:
                    env = checkpoint.env
                    rng = checkpoint.rng
                    start_day = checkpoint.next_day
                    num_days_completed = checkpoint.num_days_completed
                    total_transactions = checkpoint.total_transactions
                    total_llm_decisions = checkpoint.total_llm_decisions
                else:
                    env, population = _build_fresh_cell_environment(
                        spec, utility_type, seed, available_models, real_currency_universe, base_scenario, track=track
                    )

                    # Deliberately OUTSIDE the try/except below -- see
                    # run_matrix's identical comment for why a colliding
                    # run_id must raise straight to the caller.
                    SimulationRunRepository(session).record(
                        SimulationRunLogEntry(
                            run_id=run_id,
                            scenario_name=MASTER_SCENARIO_NAME,
                            research_mode="factual",
                            random_seed=seed,
                            model_roster_summary=model_roster_summary_for(population),
                            prompt_version_hash=prompt_version_hash,
                            git_commit_hash=git_commit_hash,
                            config_hash=config_hash,
                        )
                    )
                    session.commit()

                    rng = random.Random(seed)
                    start_day = 0
                    num_days_completed = 0
                    total_transactions = 0
                    total_llm_decisions = 0

                    if checkpoint_dir is not None:
                        # Written immediately, before day 0 even starts --
                        # otherwise a crash during day 0's run_timestep/
                        # persist_full_timestep (this cell's very first
                        # network I/O, the single likeliest moment for a
                        # transient failure across 72 cell/utility_type
                        # combinations) leaves the SimulationRunRecord above
                        # committed with NO checkpoint at all, which the
                        # skip-check above reads as "fully done" -- silently
                        # and permanently skipping this cell/seed/utility_type
                        # on every later call.
                        save_checkpoint(
                            checkpoint_dir,
                            run_id,
                            CellSeedCheckpoint(
                                env=env,
                                rng=rng,
                                next_day=0,
                                daily_results=[],
                                num_days_completed=0,
                                total_transactions=0,
                                total_llm_decisions=0,
                            ),
                        )

                try:
                    for day in range(start_day, num_days):
                        result = run_timestep(
                            env,
                            day=day,
                            rng=rng,
                            use_llm=True,
                            openrouter_client=openrouter_client,
                            polygon_client=polygon_client,
                            max_workers=llm_max_workers,
                        )
                        persist_full_timestep(session, env, result, run_id=run_id)
                        if progress_callback is not None:
                            progress_callback(spec.key, seed, utility_type, day)
                        num_days_completed += 1
                        total_transactions += len(result.transactions)
                        total_llm_decisions += len(result.llm_decisions)

                        if checkpoint_dir is not None:
                            save_checkpoint(
                                checkpoint_dir,
                                run_id,
                                CellSeedCheckpoint(
                                    env=env,
                                    rng=rng,
                                    next_day=day + 1,
                                    daily_results=[],
                                    num_days_completed=num_days_completed,
                                    total_transactions=total_transactions,
                                    total_llm_decisions=total_llm_decisions,
                                ),
                            )

                    cohort_holdings: dict[float, dict[str, float]] | None = None
                    cohort_indifference: dict[str, dict[float, float]] | None = None

                    if track == "synthetic":
                        # Every synthetic hypothesis (H1-H11), not just H1,
                        # gets holdings_by_cohort persisted -- per design spec
                        # §6.
                        cohort_holdings = holdings_by_cohort(env)
                        holdings_repo = CohortHoldingsRepository(session)
                        for cohort, per_symbol in cohort_holdings.items():
                            for symbol, pct in per_symbol.items():
                                holdings_repo.record(
                                    CohortHoldingsLogEntry(
                                        run_id=run_id,
                                        risk_aversion_cohort=cohort,
                                        currency_symbol=symbol,
                                        pct_of_wealth=pct,
                                    )
                                )

                        if spec.hypothesis != "H1":
                            # Every synthetic hypothesis except H1 ALSO gets
                            # the discrete compensation search, persisted to
                            # the SAME IndifferencePointRecord table the real
                            # track uses.
                            cohort_indifference = {}
                            indifference_repo = IndifferencePointRepository(session)
                            comparison = _synthetic_equivalence_comparison_for(
                                spec.hypothesis, spec.currencies, spec.chain_pins
                            )
                            per_cohort = cohort_discrete_switch_points(env, comparison, openrouter_client)
                            cohort_indifference[comparison.varied_currency] = {
                                cohort: result.compensation for cohort, result in per_cohort.items()
                            }
                            for cohort, result in per_cohort.items():
                                indifference_repo.record(
                                    IndifferencePointLogEntry(
                                        run_id=run_id,
                                        hypothesis=comparison.hypothesis,
                                        fixed_currency=comparison.fixed_currency,
                                        varied_currency=comparison.varied_currency,
                                        varied_field=comparison.varied_field,
                                        risk_aversion_cohort=cohort,
                                        compensation=result.compensation,
                                        censored_fraction=result.censored_fraction,
                                    )
                                )
                    elif spec.hypothesis == "H1":
                        cohort_holdings = holdings_by_cohort(env)
                        holdings_repo = CohortHoldingsRepository(session)
                        for cohort, per_symbol in cohort_holdings.items():
                            for symbol, pct in per_symbol.items():
                                holdings_repo.record(
                                    CohortHoldingsLogEntry(
                                        run_id=run_id,
                                        risk_aversion_cohort=cohort,
                                        currency_symbol=symbol,
                                        pct_of_wealth=pct,
                                    )
                                )
                    else:
                        cohort_indifference = {}
                        indifference_repo = IndifferencePointRepository(session)
                        for comparison in EQUIVALENCE_COMPARISONS[spec.hypothesis]:
                            per_cohort = cohort_indifference_points(env, comparison, openrouter_client)
                            cohort_indifference[comparison.varied_currency] = per_cohort
                            for cohort, compensation in per_cohort.items():
                                indifference_repo.record(
                                    IndifferencePointLogEntry(
                                        run_id=run_id,
                                        hypothesis=comparison.hypothesis,
                                        fixed_currency=comparison.fixed_currency,
                                        varied_currency=comparison.varied_currency,
                                        varied_field=comparison.varied_field,
                                        risk_aversion_cohort=cohort,
                                        compensation=compensation,
                                    )
                                )

                    session.commit()

                    # Deleted only now, after the analysis phase's own commit
                    # succeeds -- not right after the day loop. The day loop's
                    # checkpoint is the only recovery mechanism this run_id has
                    # (per this module's "accept full restart on crash"
                    # decision, no second checkpoint concept exists for the
                    # analysis phase itself); deleting it before the analysis
                    # phase completes would let a crash there leave behind a
                    # committed SimulationRunRecord with no checkpoint, which
                    # the skip-check above reads as "fully done" -- silently
                    # and permanently losing that cell/seed/utility_type's
                    # CohortHoldingsRecord/IndifferencePointRecord rows on any
                    # later retry. Keeping the checkpoint alive until here
                    # means a crash during analysis instead resumes with an
                    # empty (already-exhausted) day range and simply retries
                    # the analysis phase, exactly the "restart" the design
                    # intends -- without needing to redo the day loop at all.
                    if checkpoint_dir is not None:
                        delete_checkpoint(checkpoint_dir, run_id)

                    results.append(
                        HypothesisCellResult(
                            run_id=run_id,
                            cell_key=spec.key,
                            hypothesis=spec.hypothesis,
                            seed=seed,
                            utility_type=utility_type,
                            is_cross_border=getattr(spec, "cross_border", False),
                            num_days_completed=num_days_completed,
                            total_transactions=total_transactions,
                            total_llm_decisions=total_llm_decisions,
                            cohort_holdings=cohort_holdings,
                            cohort_indifference=cohort_indifference,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- deliberately broad: one cell/seed/
                    # utility_type's failure must never abort the rest of the matrix (see
                    # run_matrix's identical philosophy).
                    print(
                        f"[CELL FAILED] cell={spec.key} seed={seed} utility={utility_type}: {exc!r}",
                        flush=True,
                    )
                    traceback.print_exc()
                    session.rollback()
                    failures.append((spec.key, seed, utility_type, exc))
                    continue

    return results, failures
