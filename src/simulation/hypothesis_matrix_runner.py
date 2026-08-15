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
`checkpoint_dir` sections), with one deliberate difference: the checkpoint is
NOT deleted the moment the day loop finishes -- it stays alive until the
post-run analysis phase (holdings/indifference-point search) also commits
successfully. The analysis phase has no checkpointing of its own (per the
design spec Sec 0's third binding decision, no second checkpointing concept
was added for it); instead, a crash during it leaves the day loop's own
checkpoint in place (at `next_day == num_days`), so the next call with the
same `matrix_run_id`/database/`checkpoint_dir` resumes into an already-
exhausted day range (a no-op) and simply retries the analysis phase, rather
than silently skipping this cell/seed/utility_type forever -- the
`SimulationRunRecord`-exists-with-no-checkpoint skip-check only ever fires
once BOTH the day loop and the analysis phase have durably committed. A
cell/seed/utility_type interrupted mid-day-loop resumes from its last
persisted day exactly as `run_matrix` does.
"""

import random
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
from src.economy.equilibrium_holdings import holdings_by_cohort
from src.economy.equivalence_framework import EQUIVALENCE_COMPARISONS, cohort_indifference_points
from src.economy.hypothesis_scenarios import (
    HYPOTHESIS_CURRENCIES,
    HypothesisCellSpec,
    build_hypothesis_cell_specs,
    scenario_for,
)
from src.economy.shocks import ScenarioConfig, load_scenario
from src.economy.wallet_seeding import seed_restricted_wallets
from src.llm.agent_reasoning import PROMPT_VERSIONS, hash_rendered_prompt
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
    _resolve_available_models,
)
from src.simulation.provenance import compute_config_hash, compute_git_commit_hash, model_roster_summary_for
from src.simulation.timestep import run_timestep


class HypothesisCellResult(BaseModel):
    """One hypothesis cell/seed/utility_type combination's outcome -- mirrors
    `matrix_runner.MatrixCellResult`'s shape, plus the extra `hypothesis`/
    `utility_type` axes this runner has and neither of `holdings_by_cohort`'s
    or `cohort_indifference_points`' post-run analysis result, whichever this
    cell produced (only one of `cohort_holdings`/`cohort_indifference` is ever
    populated, per `spec.hypothesis == "H1"` or not).

    `cohort_indifference` is keyed by `EquivalenceComparison.varied_currency`
    (not `hypothesis`, since H2 is the one hypothesis with two comparisons
    sharing a `hypothesis` value, distinguished only by `varied_currency` --
    EURC vs. PAXG) -- for every other hypothesis this dict has exactly one
    key.
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


def _prompt_version_hash() -> str:
    return hash_rendered_prompt(",".join(sorted(PROMPT_VERSIONS.values())))


def _build_fresh_cell_environment(
    spec: HypothesisCellSpec,
    utility_type: str,
    seed: int,
    available_models: list[str],
    real_currency_universe: dict[str, CurrencyConfig],
    base_scenario: ScenarioConfig,
) -> tuple[Environment, list[BaseAgent]]:
    """Population -> restricted-universe Environment -> chain-pin wiring ->
    wallet seeding -> (optional) cross-border marketplace swap, for one fresh
    (non-resumed) cell/seed/utility_type. Extracted from `run_hypothesis_matrix`
    so it -- and, in particular, the `env.currency_chain_pins` wiring below --
    can be exercised directly by a small, targeted unit test rather than only
    through the full runner.

    `env.currency_chain_pins = spec.chain_pins or {}` is the missing wiring
    the design spec's Sec 4 surfaced: without it, H5/H8/H10/H11's chain-pinning
    (the whole point of those four hypotheses) silently never takes effect.
    Set unconditionally (not just for chain-pinned specs) since `{}` is the
    correct, harmless value for every other hypothesis.
    """
    population = generate_hypothesis_population(seed, available_models, utility_type)
    restricted_currencies = {symbol: real_currency_universe[symbol] for symbol in HYPOTHESIS_CURRENCIES[spec.hypothesis]}
    cell_scenario = scenario_for(spec, base_scenario)
    env = Environment.build_from_population(
        MASTER_SCENARIO_NAME, population, currencies=restricted_currencies, scenario=cell_scenario
    )
    env.currency_chain_pins = spec.chain_pins or {}
    seed_restricted_wallets(env.agents, restricted_currencies, real_currency_universe, env.macro_state.peg_reference_rates)
    if spec.cross_border:
        env.marketplace = CrossZoneMarketplace(env.agents)
    return env, population


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
    progress_callback: Callable[[str, int, str, int], None] | None = None,
    checkpoint_dir: Path | None = None,
    llm_max_workers: int = 1,
) -> tuple[list[HypothesisCellResult], list[tuple[str, int, str, Exception]]]:
    """Run every requested hypothesis-sandbox cell x utility_type x seed for
    `num_days` days each, persisting every day via `persist_full_timestep`
    plus a post-run analysis phase (`holdings_by_cohort` for H1,
    `cohort_indifference_points` for H2-H11) persisted to
    `CohortHoldingsRecord`/`IndifferencePointRecord`.

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
    requested hypothesis matches no spec).

    `checkpoint_dir`/resume semantics and `progress_callback` timing exactly
    mirror `run_matrix`'s (see that module's docstring), with the extra
    `utility_type` axis threaded through `run_id`
    (`f"{matrix_run_id}-{spec.key}-{utility_type}-seed{seed}"`),
    `progress_callback(cell_key, seed, utility_type, day)`, and `failures`'
    tuple shape.
    """
    if openrouter_client is None:
        raise ValueError(
            "run_hypothesis_matrix requires a real (or test-mock) openrouter_client -- there is no dry_run "
            "concept for this runner, since every hypothesis-sim day-loop call uses use_llm=True unconditionally."
        )

    available_models = _resolve_available_models(model_candidates, openrouter_client)

    git_commit_hash = compute_git_commit_hash()
    prompt_version_hash = _prompt_version_hash()
    config_hash = compute_config_hash(_SHARED_CONFIG_PATHS + _CURRENCY_UNIVERSE_PATHS)
    real_currency_universe = load_currency_universe()
    base_scenario = load_scenario(MASTER_SCENARIO_NAME)

    all_specs = build_hypothesis_cell_specs()
    if hypotheses is not None:
        unknown = set(hypotheses) - {spec.hypothesis for spec in all_specs}
        if unknown:
            raise ValueError(f"hypotheses contains unknown hypothesis id(s): {sorted(unknown)}")
        specs_to_run = [spec for spec in all_specs if spec.hypothesis in hypotheses]
    else:
        specs_to_run = all_specs

    resolved_utility_types = list(utility_types) if utility_types is not None else sorted(HYPOTHESIS_UTILITY_TYPES)

    results: list[HypothesisCellResult] = []
    failures: list[tuple[str, int, str, Exception]] = []

    for utility_type in resolved_utility_types:
        for spec in specs_to_run:
            for seed in seeds:
                run_id = f"{matrix_run_id}-{spec.key}-{utility_type}-seed{seed}"

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
                        spec, utility_type, seed, available_models, real_currency_universe, base_scenario
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

                    if spec.hypothesis == "H1":
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
                            is_cross_border=spec.cross_border,
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
                    session.rollback()
                    failures.append((spec.key, seed, utility_type, exc))
                    continue

    return results, failures
