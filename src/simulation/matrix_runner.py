"""The matrix runner: drives the 13-cell x N-seed experiment matrix.

The 13 cells (per Phase 3 Plan 4's design spec Sec 6.2 and the plan's
Global Constraints -- confirmed 6 sandboxes, not 7): the master simulation
(full 9-currency universe, domestic pairing) + 6 factor-isolation sandboxes,
each run once domestically and once cross-border (12 sandbox cells), each
sandbox using its own synthetic 2-currency pair
(`src.currencies.sandbox_currencies.SANDBOX_CURRENCY_PAIRS`) and the shared
365-day `master_simulation` shock schedule (Sec 9's "Resolution: reusing one
shock schedule across all 13 cells keeps every cell's macro/shock
conditions identical, isolating the currency-universe restriction as the
only difference between cells").

Cross-border pairing mechanism (this task's own open design question,
resolved after reading `src/market/marketplace.py` in full): today,
`Marketplace.find_counterparties` filters only by `good.name` and excludes
the requesting buyer's own listings -- it has no concept of currency zone
at all, and neither does `Listing` (only `seller_id`, never a zone).
`run_timestep`'s buyer loop (`src/simulation/timestep.py` ~line 503) always
takes `listings[0]`, whichever seller listed first for that good, with no
zone-awareness at either layer. Rather than threading a `require_cross_zone`
bool through `Marketplace.find_counterparties` and `run_timestep`'s call
site -- which would touch two files every one of the other 12 (non-cross-
border) cells also exercises, for a concern only 6 of the 13 cells actually
have -- this module defines `CrossZoneMarketplace`, a `Marketplace`
subclass that overrides `find_counterparties` to post-filter candidate
listings down to only those whose seller's `currency_zone` differs from the
requesting buyer's. It is swapped in for `env.marketplace` right after
`Environment.build_from_population` returns (before any `run_timestep` call
reads it), for the 6 cross-border cells only -- a purely additive,
cell-local change that touches neither `marketplace.py` nor `timestep.py`.

Sandbox wallet-seeding gap (discovered while implementing this task, not
specified by any prior task): every agent profile's `initial_wallet`
(configs/agent_profiles/*.yaml) holds real-universe symbols only
(USDC/EURC/PAXG/...). Two separate problems follow from that, for every one
of the 12 sandbox cells (whose `env.currencies` is restricted to a
synthetic 2-symbol pair no profile ever mentions):

1. `src.blockchain.routing_engine.generate_candidates` only offers a
   currency the buyer already holds a positive balance of *and* that's
   present in `env.currencies` -- left unaddressed, every buyer's candidate
   list would be empty forever, silently defeating the entire point of the
   factor-isolation sandboxes (zero transactions, zero signal).
2. `database.repository.persist_full_timestep` calls
   `real_purchasing_power` -> `wallet.total_value_usd(rates)` for every
   agent (not just buyers), which tries to convert every symbol the wallet
   holds -- including leftover real-universe symbols -- through an
   `ExchangeRateTable` built from only the sandbox's 2 currencies. That
   raises `KeyError` on the very first persisted day (confirmed by running
   this task's own tests before this fix), since the table has no peg
   reference for a symbol it was never built to know about.

This module's `_seed_sandbox_wallets` fixes both: every agent (buyer,
seller, and the non-participating bank/investor/institution classes alike
-- all of them get logged by `persist_full_timestep`) has its wallet
*replaced* (not merely topped up) with a balance in both of the sandbox
pair's symbols, sized to that agent's pre-existing total wallet value
(summed raw balances -- no cross-currency conversion is attempted, since
the agent's original symbols aren't even part of this sandbox's universe).
Replacing rather than appending is deliberate: leaving old real-universe
balances in place would keep triggering problem 2 above.

`dry_run` safety gate: `dry_run=False` requires the caller to supply BOTH a
real `openrouter_client` and a real `polygon_client` explicitly (raises
`ValueError` otherwise) -- the code-level half of "explicit confirmation
before billed spend"; this function never decides on its own to make that
call. Whether the LLM-driven day-loop path (`run_timestep(use_llm=True)`)
actually runs is governed by whether an `openrouter_client` was supplied at
all, not by `dry_run` directly: `dry_run=True` (the default) is only a
promise that *no real client is required*, not that a client can never be
passed. A caller may still pass their own mock `openrouter_client` (e.g.
`tests/llm_test_helpers.py`'s `mock_openrouter_client`) alongside
`dry_run=True` to exercise the LLM-vs-LLM negotiation path in a fast test
without touching a real API -- `run_matrix` doesn't care which kind of
client it was handed, only that `dry_run=False` guarantees a real one.

This module deliberately does NOT construct mock `httpx.Client`s internally
(unlike the design spec Sec 10's literal wording, "both clients are
mock-transport fakes constructed internally if not supplied") -- the task
brief that superseded that spec explicitly left this as the implementer's
call ("or just skip use_llm/live-price wiring entirely in dry-run if
that's simpler"). Building mock clients internally would mean importing
`tests/llm_test_helpers.py` (a tests-only module) from `src/`, which this
codebase never does elsewhere. Instead, when no `openrouter_client` is
supplied (the common `dry_run=True` case), the day loop simply runs the
deterministic rule-based path (`use_llm=False`) -- zero network access,
zero mock-client bookkeeping, and still exercises every other mechanism
this task must integrate (population generation, environment construction,
cross-border pairing, sandbox wallet seeding, persistence, provenance).

Model-candidate verification (`src.llm.llm_router.verify_model_candidates`)
is called at most ONCE per `run_matrix` call -- not once per cell/seed --
and only when `dry_run=False` (the only case that guarantees a real
`openrouter_client`, per the safety gate above). `dry_run=True` never
verifies, even if the caller supplied their own mock `openrouter_client`
for the LLM-vs-LLM path: `tests/llm_test_helpers.py`'s
`mock_openrouter_client` only fakes the `/chat/completions` endpoint
`call_model` uses, not the `/models` endpoint `verify_model_candidates`
needs (confirmed by running this task's own tests before this decision --
routing a test-supplied mock client through verification raised an
unhandled 404 assertion inside the shared test helper). Treating
"`dry_run=True` was passed" as "skip verification" avoids that mismatch
entirely and keeps `dry_run=True`'s promise ("no real API calls,
ever") unconditional rather than dependent on what the caller happened to
pass in. The resulting `available_models` list is cached in a local
variable and reused by every `generate_agent_population` call in the
matrix.
"""

import random
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from database.repository import SimulationRunLogEntry, SimulationRunRepository, persist_full_timestep
from src.agents.base_agent import BaseAgent
from src.agents.population import generate_agent_population
from src.currencies.currency import CurrencyConfig
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.llm.agent_reasoning import PROMPT_VERSIONS, hash_rendered_prompt
from src.llm.llm_router import verify_model_candidates
from src.market.marketplace import Listing, Marketplace
from src.simulation.environment import Environment
from src.simulation.provenance import compute_config_hash, compute_git_commit_hash, model_roster_summary_for
from src.simulation.timestep import TimestepResult, run_timestep
from src.utils.constants import CONFIG_ROOT

MASTER_SCENARIO_NAME = "master_simulation"

# Provenance file set shared by every cell (the scenario itself, chain
# universe, and every economics/agent-profile config that shapes a run's
# behavior regardless of which currency universe it uses). The real
# 9-currency universe (configs/currencies/*.yaml) is appended only for the
# master cell -- the 6 sandbox pairs are Python-defined CurrencyConfig
# instances (src/currencies/sandbox_currencies.py), never loaded from YAML,
# so there is no sandbox-specific file to hash.
_SHARED_CONFIG_PATHS: list[Path] = [
    CONFIG_ROOT / "scenarios" / f"{MASTER_SCENARIO_NAME}.yaml",
    *sorted((CONFIG_ROOT / "blockchains").glob("*.yaml")),
    CONFIG_ROOT / "economy" / "trust_params.yaml",
    CONFIG_ROOT / "economy" / "fx_params.yaml",
    CONFIG_ROOT / "economy" / "risk_adaptation_params.yaml",
    *sorted((CONFIG_ROOT / "agent_profiles").glob("*.yaml")),
]
_CURRENCY_UNIVERSE_PATHS: list[Path] = sorted((CONFIG_ROOT / "currencies").glob("*.yaml"))


class MatrixCellResult(BaseModel):
    """One experiment cell's outcome for one seed: which cell, which run_id
    it was persisted under, and every simulated day's `TimestepResult` (so
    callers/tests can inspect transactions/negotiations/decisions without
    re-querying the database)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    cell_key: str
    seed: int
    is_cross_border: bool
    num_currencies: int
    daily_results: list[TimestepResult] = []


class CrossZoneMarketplace(Marketplace):
    """`Marketplace` subclass that only surfaces zone-mismatched listings to
    a buyer, forcing cross-border pairing for the 6 cross-border cells --
    see this module's docstring for the full rationale.

    Needs an `agents` lookup (by agent_id) to resolve both the requesting
    buyer's and each listing's seller's `currency_zone`; `Environment`
    itself doesn't accept a marketplace override at construction time (it
    always builds a plain `Marketplace()` in `__init__`), so callers build
    the `Environment` normally and then replace `env.marketplace` with one
    of these, passing `env.agents` (already fully populated by that point).
    """

    def __init__(self, agents: dict[str, BaseAgent]):
        super().__init__()
        self._agents = agents

    def find_counterparties(self, good_name: str, exclude_agent_id: str | None = None) -> list[Listing]:
        candidates = super().find_counterparties(good_name, exclude_agent_id)
        buyer = self._agents.get(exclude_agent_id) if exclude_agent_id is not None else None
        if buyer is None or buyer.currency_zone is None:
            return candidates
        return [
            listing
            for listing in candidates
            if (seller := self._agents.get(listing.seller_id)) is not None
            and seller.currency_zone is not None
            and seller.currency_zone != buyer.currency_zone
        ]


class _CellSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    currencies: dict[str, CurrencyConfig] | None  # None -> full real universe
    cross_border: bool


def _build_cell_specs() -> list[_CellSpec]:
    specs = [_CellSpec(key="master", currencies=None, cross_border=False)]
    for sandbox_name, (option_a, option_b) in SANDBOX_CURRENCY_PAIRS.items():
        pair = {option_a.symbol: option_a, option_b.symbol: option_b}
        specs.append(_CellSpec(key=f"{sandbox_name}_domestic", currencies=pair, cross_border=False))
        specs.append(_CellSpec(key=f"{sandbox_name}_cross_border", currencies=pair, cross_border=True))
    return specs


def _config_paths_for(spec: _CellSpec) -> list[Path]:
    paths = list(_SHARED_CONFIG_PATHS)
    if spec.currencies is None:
        paths.extend(_CURRENCY_UNIVERSE_PATHS)
    return paths


def _prompt_version_hash() -> str:
    return hash_rendered_prompt(",".join(sorted(PROMPT_VERSIONS.values())))


def _resolve_available_models(model_candidates: list[str], openrouter_client: httpx.Client | None) -> list[str]:
    """Preflight-verify the candidate pool against OpenRouter exactly once
    for the whole matrix run, but only when a client (real or test-mock) was
    actually supplied -- there is nothing to verify against otherwise, and
    the common `dry_run=True`, no-client case must stay at zero network
    access."""
    if openrouter_client is None:
        return list(model_candidates)

    available, unavailable = verify_model_candidates(model_candidates, openrouter_client)
    if not available:
        raise ValueError(
            f"None of the supplied model_candidates are available on OpenRouter: unavailable={unavailable}"
        )
    return available


def _seed_sandbox_wallets(agents: dict[str, BaseAgent], currencies: dict[str, CurrencyConfig]) -> None:
    """See this module's docstring ("Sandbox wallet-seeding gap"). REPLACES
    every agent's wallet (buyer, seller, and non-participating classes
    alike -- `persist_full_timestep` logs all of them) with a balance split
    evenly across both of `currencies`' symbols, sized to that agent's
    pre-existing total wallet value. Buyers need this to get any candidates
    at all from `generate_candidates`; every agent needs their leftover
    real-universe balances removed so `real_purchasing_power`'s
    `ExchangeRateTable` lookup (built from only these 2 sandbox currencies)
    never KeyErrors on a symbol it doesn't know."""
    symbols = list(currencies.keys())
    for agent in agents.values():
        total_value = sum(agent.wallet.balances.values())
        if total_value <= 0:
            total_value = 1000.0  # safe floor, matching consumer.yaml's USDC scale
        share = total_value / len(symbols)
        agent.wallet.balances = {symbol: share for symbol in symbols}


def run_matrix(
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    dry_run: bool = True,
    openrouter_client: httpx.Client | None = None,
    polygon_client: httpx.Client | None = None,
    session: Session | None = None,
) -> list[MatrixCellResult]:
    """Run the 13-cell x `seeds` experiment matrix for `num_days` days each.

    `dry_run=False` requires both `openrouter_client` and `polygon_client`
    to be supplied explicitly (raises `ValueError` otherwise); see this
    module's docstring for the full safety-gate and cross-border-pairing
    rationale.

    `session`, if `None` (the default), opens the project's normal database
    session (`database.session.new_session`, creating tables if needed) --
    the right default for a real run, whose whole point is durable
    persistence. Tests must pass their own throwaway
    `sqlite:///:memory:`-backed `Session` explicitly (matching every other
    `tests/test_*_persistence.py` file's convention) so a fast dry-run test
    suite never touches the real on-disk database.
    """
    if not dry_run and (openrouter_client is None or polygon_client is None):
        raise ValueError(
            "run_matrix(dry_run=False) requires both a real openrouter_client and a real polygon_client to be "
            "supplied explicitly -- this is the code-level half of the explicit-confirmation-before-billed-spend "
            "gate. dry_run=True (the default) never requires real clients."
        )

    if session is None:
        from database.session import create_all_tables, new_session

        create_all_tables()
        session = new_session()

    available_models = _resolve_available_models(model_candidates, None if dry_run else openrouter_client)
    use_llm = openrouter_client is not None

    git_commit_hash = compute_git_commit_hash()
    prompt_version_hash = _prompt_version_hash()

    results: list[MatrixCellResult] = []

    for spec in _build_cell_specs():
        config_hash = compute_config_hash(_config_paths_for(spec))

        for seed in seeds:
            population = generate_agent_population(seed, available_models)
            env = Environment.build_from_population(MASTER_SCENARIO_NAME, population, currencies=spec.currencies)
            if spec.currencies is not None:
                _seed_sandbox_wallets(env.agents, spec.currencies)
            if spec.cross_border:
                env.marketplace = CrossZoneMarketplace(env.agents)

            run_id = f"{spec.key}-seed{seed}"

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
            daily_results: list[TimestepResult] = []
            for day in range(num_days):
                result = run_timestep(
                    env,
                    day=day,
                    rng=rng,
                    use_llm=use_llm,
                    openrouter_client=openrouter_client,
                    polygon_client=polygon_client,
                )
                persist_full_timestep(session, env, result, run_id=run_id)
                daily_results.append(result)

            results.append(
                MatrixCellResult(
                    run_id=run_id,
                    cell_key=spec.key,
                    seed=seed,
                    is_cross_border=spec.cross_border,
                    num_currencies=len(env.currencies),
                    daily_results=daily_results,
                )
            )

    return results
