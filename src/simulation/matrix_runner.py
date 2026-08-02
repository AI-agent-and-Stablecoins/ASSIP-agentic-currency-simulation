"""The matrix runner: drives the 13-cell x N-seed experiment matrix.

The 13 cells (per Phase 3 Plan 4's design spec Sec 6.2 and the plan's
Global Constraints -- confirmed 6 sandboxes, not 7): the master simulation
(full 9-currency universe, domestic pairing) + 6 factor-isolation sandboxes,
each run once domestically and once cross-border (12 sandbox cells), each
sandbox using its own synthetic 2-currency pair
(`src.currencies.sandbox_currencies.SANDBOX_CURRENCY_PAIRS`).

Per-cell scenario: the master cell uses the unmodified `master_simulation`
`ScenarioConfig` (loaded once, as `base_scenario`, and reused verbatim). Each
of the 12 sandbox cells instead uses `src.economy.sandbox_scenarios
.build_sandbox_scenario`'s per-sandbox `ScenarioConfig` -- built once per
sandbox (not per cell/seed) from that same `base_scenario` -- rather than
`master_simulation.yaml`'s shock schedule verbatim: `master_simulation
.yaml`'s 13 currency-targeted shocks name real-universe symbols exclusively
(USDC, PAXG, EURT, ...), none of which exist in any sandbox's synthetic
2-currency universe, so reusing it as-is would leave every currency-targeted
shock silently inert for all 12 sandbox cells (only its 5 macro-level shocks
would still apply). `build_sandbox_scenario` keeps those same 5 macro-level
shocks and adds two shocks of its own that target one of that sandbox's
actual currency symbols -- see that module's docstring for the full
per-sandbox rationale. Every cell's macro backdrop (`initial_state`,
`duration_days`) and 5 macro-level shocks still trace back to the same
`master_simulation.yaml` file, so the currency-universe restriction remains
the primary difference between the master and sandbox cells.

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
pair's symbols, sized to that agent's pre-existing total wallet VALUE in
USD (via `Wallet.total_value_usd` against the real 9-currency universe's
`ExchangeRateTable`, not a raw sum of balances across different real
currencies -- a unit of USDC and a unit of PAXG are not worth the same
amount), split evenly by VALUE (not by unit count) across the sandbox's two
symbols via that sandbox's own `ExchangeRateTable`. Splitting by raw unit
count would badly skew the two asset-backing sandboxes: a gold-pegged
sandbox symbol (`peg="XAU"`, ~2400 USD/unit) holding the same NUMBER of
units as a stablecoin-pegged sandbox symbol (`peg="USD"`, 1 USD/unit) would
hold ~2400x its counterpart's actual USD value, corrupting
`real_purchasing_power`, `adapt_cara_coefficient`, and any shock's measured
wealth impact for exactly the two sandboxes designed to isolate
`asset_class` as a factor. Replacing rather than appending old balances is
deliberate: leaving old real-universe balances in place would keep
triggering problem 2 above.

`dry_run` safety gate: `dry_run=False` requires the caller to supply BOTH a
real `openrouter_client` and a real `polygon_client` explicitly (raises
`ValueError` otherwise) -- the code-level half of "explicit confirmation
before billed spend"; this function never decides on its own to make that
call. `dry_run=True` (the default) is the mirror-image guarantee: it now
REFUSES any externally-supplied `openrouter_client`/`polygon_client` at
all (raises `ValueError`), real or mock -- `run_matrix` has no way to tell
a real `httpx.Client` from a test-only mock one apart by inspecting the
object, so the only way to make `dry_run=True` an unconditional "no real
network call is possible" guarantee is to never accept an external client
under it, full stop. (An earlier version of this gate let `dry_run=True`
accept any caller-supplied client, reasoning that a caller wouldn't pass a
real one under dry_run; that trust was the loophole -- nothing stopped a
caller from doing exactly that, real client + `dry_run=True`, and getting
silent real spend. Closed here.)

To exercise the LLM-driven day-loop path (`run_timestep(use_llm=True)`)
under `dry_run=True` without an external client, use `exercise_llm_path=
True`: `run_matrix` then builds its OWN mock OpenRouter/Polygon clients
internally, via `tests/llm_test_helpers.py`'s `mock_openrouter_client`/
`mock_polygon_client` -- clients that are mocks by construction, never by
caller promise. `mock_llm_decision` (a plain response dict, not a client)
optionally overrides the default canned decision used to build that
internal mock, for a test that needs a specific proposed currency/price/
action; passing it without `exercise_llm_path=True` (under `dry_run=True`)
raises `ValueError`, since it would otherwise silently do nothing.

This module does NOT construct mock `httpx.Client`s internally BY DEFAULT
(unlike the design spec Sec 10's literal wording, "both clients are
mock-transport fakes constructed internally if not supplied") -- the task
brief that superseded that spec explicitly left this as the implementer's
call ("or just skip use_llm/live-price wiring entirely in dry-run if
that's simpler"). Instead, when `exercise_llm_path=False` (the default),
the day loop simply runs the deterministic rule-based path
(`use_llm=False`) -- zero network access, zero mock-client bookkeeping,
and still exercises every other mechanism this task must integrate
(population generation, environment construction, cross-border pairing,
sandbox wallet seeding, persistence, provenance).

A later fix round revisited this for one specific, narrow reason: `run_
matrix`'s OWN test suite (`tests/test_matrix_runner.py`) never exercised
the LLM-decision + LLM-decision-persistence path end-to-end, since every
test used `dry_run=True` with no client supplied. `exercise_llm_path:
bool = False` closes that gap without changing the default path at all:
when `True` (always alongside `dry_run=True`, since `exercise_llm_path`
with `dry_run=False` is simply the normal real-client path), `run_matrix`
builds mock clients internally and runs with `use_llm=True`. This DOES
mean `src/simulation/matrix_runner.py` imports `tests/llm_test_helpers.py`
-- a tests-only module -- but only inside the `exercise_llm_path` branch
(a local import, not a module-level one), and only when a caller
explicitly opts in; the default (`exercise_llm_path=False`) path never
touches `tests/` and behaves exactly as before. Every existing caller/test
that doesn't pass `exercise_llm_path=True` sees zero behavior change.

A review fix to the above: the canned mock decision's `proposed_currency`
must be a symbol `adapt_decision` will actually accept for THIS cell --
`supported_currencies` there is narrowed to exactly the candidates
`generate_candidates` offered that round (see `src/simulation/timestep.py`
~line 535), which for the 12 sandbox cells is always one of that
sandbox's own two synthetic `SBX*` symbols (`src/currencies
.sandbox_currencies.py`), never a real-universe symbol. A single mock
client shared across all 13 cells therefore cannot use one hardcoded
symbol for every cell -- "USDC" (the real universe's symbol, valid for
the master cell only, since every `configs/agent_profiles/*.yaml`
profile's `initial_wallet` holds it) would be rejected as
"Unsupported currency" in all 12 sandbox cells, producing a synthetic
`WALK_AWAY` there instead of a genuine `ACCEPT`. Fixed by building a
*per-cell* mock OpenRouter client whose canned `proposed_currency` is
`"USDC"` for the master cell but `next(iter(spec.currencies))` -- one of
that sandbox's own two symbols, guaranteed held by every agent post-
`_seed_sandbox_wallets` -- for each of the 12 sandbox cells (unless
`mock_llm_decision` was supplied, which is then used verbatim for every
cell instead). `mock_polygon_client`, whose canned responses are
currency-ticker-keyed but degrade to an empty `{"results": []}` for any
ticker with no entry regardless of which cell is running, has no such
per-cell dependency and is still built once for the whole matrix run.

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

Memory, progress, and per-cell error recovery (a later fix round, aimed at
real-scale runs -- 13 cells x 5 seeds x 365 days -- rather than this
module's own fast tests): three related, additive parameters.

- `keep_daily_results: bool = False` (new default). Every `TimestepResult`
  is already durably persisted via `persist_full_timestep` the moment it's
  produced; retaining the full day-by-day list in `MatrixCellResult
  .daily_results` on top of that, for every cell/seed of a real run, is
  tens of thousands of objects (including full negotiation logs and LLM
  reasoning strings) held in memory for no reason. When `False`,
  `daily_results` stays empty and only cheap per-cell aggregates
  (`num_days_completed`, `total_transactions`, `total_llm_decisions`) are
  kept. Passing `True` restores the original full-retention behavior
  (every test written before this fix round that reads `.daily_results`
  now passes `keep_daily_results=True` explicitly).
- `progress_callback: Callable[[str, int, int], None] | None = None`.
  Called once per simulated day, right after that day's
  `persist_full_timestep`, as `progress_callback(cell_key, seed, day)`.
  `run_matrix` hardcodes no logging/display mechanism of its own -- a
  caller driving a real multi-week run wires in whatever progress
  reporting it wants. `None` (the default) is a no-op: zero behavior
  change for every existing caller.
- Per-cell/seed error recovery. Each cell/seed's day loop (from
  `random.Random(seed)` through appending that cell/seed's
  `MatrixCellResult`) is wrapped in a `try/except Exception`; a failure
  aborts only that cell/seed's remaining days (day-to-day state within a
  cell is sequential/stateful, so partial completion of one cell/seed is
  not meaningfully resumable mid-cell) and is recorded as `(cell_key,
  seed, exception)` in a `failures` list, rather than propagating and
  losing every other cell/seed's results. `run_matrix` therefore returns
  `(results, failures)` -- a 2-tuple -- rather than a bare
  `list[MatrixCellResult]`; `failures` is empty on a fully clean run. The
  session is rolled back before continuing to the next cell/seed, so an
  uncommitted partial day never corrupts a later cell/seed's commits.
  Population generation, environment construction, and the
  `SimulationRunRepository`/`session.commit()` call that registers this
  cell/seed's `run_id` all stay OUTSIDE this try/except, deliberately: a
  colliding `run_id` (see `matrix_run_id`'s docstring) must still raise
  `IntegrityError` straight to the caller, not be silently downgraded to a
  `failures` entry.
"""

import random
from pathlib import Path
from typing import Callable

import httpx
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from database.repository import SimulationRunLogEntry, SimulationRunRepository, persist_full_timestep
from src.agents.base_agent import BaseAgent
from src.agents.population import generate_agent_population
from src.currencies.currency import CurrencyConfig, load_currency_universe
from src.currencies.exchange_rates import ExchangeRateTable
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.economy.sandbox_scenarios import build_sandbox_scenario
from src.economy.shocks import ScenarioConfig, load_scenario
from src.llm.agent_reasoning import PROMPT_VERSIONS, hash_rendered_prompt
from src.llm.llm_router import verify_model_candidates
from src.market.marketplace import Listing, Marketplace
from src.simulation.environment import Environment
from src.simulation.provenance import compute_config_hash, compute_git_commit_hash, model_roster_summary_for
from src.simulation.timestep import TimestepResult, run_timestep
from src.utils.constants import CONFIG_ROOT
from src.utils.helpers import generate_id

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
    it was persisted under, cheap per-cell aggregates
    (`num_days_completed`/`total_transactions`/`total_llm_decisions`,
    always populated), and -- only when `run_matrix(keep_daily_results=
    True)` -- every simulated day's full `TimestepResult` in
    `daily_results`, so callers/tests can inspect transactions/
    negotiations/decisions without re-querying the database. When
    `keep_daily_results=False` (the default), `daily_results` stays empty:
    every day is already durably persisted via `persist_full_timestep` as
    it happens, so retaining the full list too is pure redundancy at real
    scale (see this module's docstring)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    cell_key: str
    seed: int
    is_cross_border: bool
    num_currencies: int
    num_days_completed: int = 0
    total_transactions: int = 0
    total_llm_decisions: int = 0
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
    # None for the master cell (uses the unmodified master_simulation.yaml
    # scenario); one of SANDBOX_CURRENCY_PAIRS's keys for the 12 sandbox
    # cells, identifying which of run_matrix's precomputed
    # build_sandbox_scenario results this cell should use -- see
    # src/economy/sandbox_scenarios.py for why the sandboxes need their own
    # scenario rather than reusing master_simulation.yaml's shocks verbatim
    # (those target real-universe currency symbols exclusively, which no
    # sandbox's synthetic 2-currency universe ever holds).
    sandbox_key: str | None = None


def _build_cell_specs() -> list[_CellSpec]:
    specs = [_CellSpec(key="master", currencies=None, cross_border=False, sandbox_key=None)]
    for sandbox_name, (option_a, option_b) in SANDBOX_CURRENCY_PAIRS.items():
        pair = {option_a.symbol: option_a, option_b.symbol: option_b}
        specs.append(
            _CellSpec(key=f"{sandbox_name}_domestic", currencies=pair, cross_border=False, sandbox_key=sandbox_name)
        )
        specs.append(
            _CellSpec(
                key=f"{sandbox_name}_cross_border", currencies=pair, cross_border=True, sandbox_key=sandbox_name
            )
        )
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


def _seed_sandbox_wallets(
    agents: dict[str, BaseAgent],
    sandbox_currencies: dict[str, CurrencyConfig],
    real_currencies: dict[str, CurrencyConfig],
    peg_reference_rates: dict[str, float],
) -> None:
    """See this module's docstring ("Sandbox wallet-seeding gap"). REPLACES
    every agent's wallet (buyer, seller, and non-participating classes
    alike -- `persist_full_timestep` logs all of them) with a balance split
    evenly BY USD VALUE (not raw unit count -- see the module docstring for
    why unit-count splitting badly skews the asset-backing sandboxes) across
    both of `sandbox_currencies`' symbols, sized to that agent's pre-existing
    total wallet value. Buyers need this to get any candidates at all from
    `generate_candidates`; every agent needs their leftover real-universe
    balances removed so `real_purchasing_power`'s `ExchangeRateTable` lookup
    (built from only these 2 sandbox currencies) never KeyErrors on a symbol
    it doesn't know.

    `real_currencies` and `peg_reference_rates` let this function build an
    `ExchangeRateTable` for the agent's ORIGINAL (real-universe) balances --
    by the time this is called, `env.currencies` has already been replaced
    with `sandbox_currencies`, so the real universe must be supplied
    separately. `peg_reference_rates` (e.g. `{"USD": 1.0, "EUR": 1.08, "XAU":
    2400.0}`) is shared between both tables: every real and sandbox currency
    alike pegs to one of USD/EUR/XAU, so the same reference dict prices both
    sides consistently.
    """
    real_rates = ExchangeRateTable(real_currencies, peg_reference_rates)
    sandbox_rates = ExchangeRateTable(sandbox_currencies, peg_reference_rates)
    symbols = list(sandbox_currencies.keys())
    for agent in agents.values():
        total_usd_value = agent.wallet.total_value_usd(real_rates)
        if total_usd_value <= 0:
            total_usd_value = 1000.0  # safe floor, matching consumer.yaml's USDC scale
        share_usd = total_usd_value / len(symbols)
        agent.wallet.balances = {
            symbol: sandbox_rates.convert(share_usd, "USD", symbol) for symbol in symbols
        }


def run_matrix(
    model_candidates: list[str],
    seeds: list[int],
    num_days: int,
    dry_run: bool = True,
    openrouter_client: httpx.Client | None = None,
    polygon_client: httpx.Client | None = None,
    session: Session | None = None,
    matrix_run_id: str | None = None,
    keep_daily_results: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
    exercise_llm_path: bool = False,
    mock_llm_decision: dict | None = None,
) -> tuple[list[MatrixCellResult], list[tuple[str, int, Exception]]]:
    """Run the 13-cell x `seeds` experiment matrix for `num_days` days each.

    Returns `(results, failures)`. `failures` is a list of
    `(cell_key, seed, exception)` for any cell/seed whose day loop raised --
    that cell/seed's remaining days are aborted (day-to-day state within a
    cell is sequential, so there is no meaningful way to resume mid-cell),
    but every OTHER cell/seed in the matrix still runs. `failures` is empty
    on a fully clean run; every pre-existing test (before this parameter
    existed) asserts `failures == []`.

    `keep_daily_results` (default `False`) controls whether each
    `MatrixCellResult.daily_results` retains every simulated day's full
    `TimestepResult` (the original, pre-fix behavior, restored by passing
    `True`) or stays empty, relying on the already-durable
    `persist_full_timestep` writes plus `MatrixCellResult`'s cheap
    aggregate counts instead (the new default -- see this module's
    docstring for why unconditional full retention is unsafe at real
    scale).

    `progress_callback`, if given, is called once per simulated day as
    `progress_callback(cell_key, seed, day)`, right after that day's
    `persist_full_timestep`. `None` (the default) is a no-op.

    `exercise_llm_path` (default `False`) only matters when `dry_run=True`:
    when both are `True`, `run_matrix` builds mock OpenRouter/Polygon
    clients internally (via `tests/llm_test_helpers.py`) for whichever of
    `openrouter_client`/`polygon_client` the caller didn't already supply,
    and runs every cell with `use_llm=True` -- see this module's docstring
    for the full rationale (letting `run_matrix`'s own test suite exercise
    the LLM-decision + LLM-decision-persistence path end-to-end). The mock
    OpenRouter client is built freshly PER CELL (not once for the whole
    matrix), since its canned response's `proposed_currency` must be a
    symbol that cell's own currency universe actually supports -- see this
    module's docstring's review-fix paragraph for why a single shared mock
    response cannot satisfy both the master cell and the 12 sandbox cells
    at once.

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

    `matrix_run_id`, if `None` (the default), is generated fresh
    (`generate_id("matrix")`) once at the start of this call and prefixed
    onto every cell/seed's `run_id`
    (`f"{matrix_run_id}-{spec.key}-seed{seed}"`). `SimulationRunRecord.run_id`
    is a primary key with no upsert, so two separate `run_matrix` calls
    against the SAME database (a pilot run followed by the real run, or a
    restart after a partial failure) would otherwise always collide on the
    first invocation's deterministic `run_id`s and raise `IntegrityError`.
    Auto-generating a fresh `matrix_run_id` per call makes that the default,
    safe behavior. A caller may still pass an explicit `matrix_run_id` to
    deliberately resume/retry under a stable, predictable prefix -- doing so
    means a second call with the SAME `matrix_run_id` (and the same
    cells/seeds) WILL collide again; that is intentional, not a bug, since
    "resume under this exact id" implies the caller wants the collision (or
    is retrying against an empty/rolled-back database).
    """
    if not dry_run and (openrouter_client is None or polygon_client is None):
        raise ValueError(
            "run_matrix(dry_run=False) requires both a real openrouter_client and a real polygon_client to be "
            "supplied explicitly -- this is the code-level half of the explicit-confirmation-before-billed-spend "
            "gate. dry_run=True (the default) never requires real clients."
        )

    if dry_run and (openrouter_client is not None or polygon_client is not None):
        raise ValueError(
            "run_matrix(dry_run=True) never accepts an externally-supplied openrouter_client/polygon_client -- "
            "real or mock, run_matrix cannot tell them apart, and that ambiguity is exactly what let a real "
            "client slip through under dry_run=True. Use exercise_llm_path=True (optionally with "
            "mock_llm_decision to customize the canned response) to exercise the LLM path under dry_run instead."
        )

    if mock_llm_decision is not None and not (dry_run and exercise_llm_path):
        raise ValueError(
            "mock_llm_decision only applies alongside dry_run=True and exercise_llm_path=True -- passing it "
            "otherwise would silently do nothing."
        )

    if session is None:
        from database.session import create_all_tables, new_session

        create_all_tables()
        session = new_session()

    if matrix_run_id is None:
        matrix_run_id = generate_id("matrix")

    # Under dry_run=True, openrouter_client/polygon_client are guaranteed
    # None by the gate above -- any client this run uses is either the real
    # one required by dry_run=False, or one this function builds itself
    # below. `mock_openrouter_client` is deliberately NOT built here for the
    # whole matrix -- unlike `mock_polygon_client`, its canned response's
    # `proposed_currency` must be valid for whichever cell is currently
    # running (the master cell's real universe vs. each sandbox's own
    # synthetic pair), so it's built fresh per cell inside the loop below
    # instead. See this module's docstring's review-fix paragraph.
    if dry_run and exercise_llm_path:
        from tests.llm_test_helpers import mock_polygon_client

        polygon_client = mock_polygon_client({})

    available_models = _resolve_available_models(model_candidates, None if dry_run else openrouter_client)
    use_llm = (not dry_run) or exercise_llm_path

    git_commit_hash = compute_git_commit_hash()
    prompt_version_hash = _prompt_version_hash()
    real_currency_universe = load_currency_universe()

    # The master scenario, loaded once and reused unmodified by the master
    # cell; also the macro/shock backdrop every sandbox scenario below is
    # derived from. Precomputing all 6 sandbox ScenarioConfigs once here
    # (rather than per cell/seed) mirrors how `available_models` is resolved
    # once for the whole matrix -- none of this depends on seed.
    base_scenario = load_scenario(MASTER_SCENARIO_NAME)
    sandbox_scenarios: dict[str, ScenarioConfig] = {
        sandbox_name: build_sandbox_scenario(sandbox_name, option_a, option_b, base_scenario)
        for sandbox_name, (option_a, option_b) in SANDBOX_CURRENCY_PAIRS.items()
    }

    results: list[MatrixCellResult] = []
    failures: list[tuple[str, int, Exception]] = []

    for spec in _build_cell_specs():
        config_hash = compute_config_hash(_config_paths_for(spec))
        cell_scenario = base_scenario if spec.sandbox_key is None else sandbox_scenarios[spec.sandbox_key]

        # exercise_llm_path's mock OpenRouter client, built fresh per cell so
        # its canned proposed_currency is one this cell's currency universe
        # actually supports -- "USDC" for the master cell (every agent
        # profile's initial_wallet holds it, confirmed against every
        # configs/agent_profiles/*.yaml file), or one of the sandbox's own
        # two symbols (guaranteed held by every agent post-
        # `_seed_sandbox_wallets`) for each of the 12 sandbox cells. Under
        # dry_run=False, openrouter_client is the real client required above.
        cell_openrouter_client = openrouter_client
        if dry_run and exercise_llm_path:
            from tests.llm_test_helpers import mock_openrouter_client

            cell_mock_currency = "USDC" if spec.currencies is None else next(iter(spec.currencies))
            default_decision = {
                "action": "ACCEPT",
                "proposed_currency": cell_mock_currency,
                "proposed_chain": "ethereum",
                "amount": 1.0,
                "price": 1.0,
                "reasoning": "exercise_llm_path canned response",
            }
            decision = mock_llm_decision if mock_llm_decision is not None else default_decision
            cell_openrouter_client = mock_openrouter_client({model_id: decision for model_id in model_candidates})

        for seed in seeds:
            population = generate_agent_population(seed, available_models)
            env = Environment.build_from_population(
                MASTER_SCENARIO_NAME, population, currencies=spec.currencies, scenario=cell_scenario
            )
            if spec.currencies is not None:
                _seed_sandbox_wallets(
                    env.agents, spec.currencies, real_currency_universe, env.macro_state.peg_reference_rates
                )
            if spec.cross_border:
                env.marketplace = CrossZoneMarketplace(env.agents)

            run_id = f"{matrix_run_id}-{spec.key}-seed{seed}"

            # Deliberately OUTSIDE the try/except below: a colliding run_id
            # (see `matrix_run_id`'s docstring -- reusing the same explicit
            # matrix_run_id against the same database is an intentional
            # "resume under this exact id" request) must still raise
            # IntegrityError to the caller immediately, not be swallowed
            # into `failures` as if it were an ordinary per-cell/seed
            # simulation failure.
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

            # Everything from here on is this cell/seed's actual simulated
            # day loop -- the part that scales with `num_days` and can fail
            # partway through a long real run. A failure here aborts only
            # this cell/seed's remaining days (day-to-day state within a
            # cell is sequential/stateful, so there's no meaningful way to
            # resume mid-cell) and is recorded in `failures` rather than
            # aborting every other cell/seed in the matrix.
            try:
                rng = random.Random(seed)
                daily_results: list[TimestepResult] = []
                num_days_completed = 0
                total_transactions = 0
                total_llm_decisions = 0
                for day in range(num_days):
                    result = run_timestep(
                        env,
                        day=day,
                        rng=rng,
                        use_llm=use_llm,
                        openrouter_client=cell_openrouter_client,
                        polygon_client=polygon_client,
                    )
                    persist_full_timestep(session, env, result, run_id=run_id)
                    if progress_callback is not None:
                        progress_callback(spec.key, seed, day)
                    num_days_completed += 1
                    total_transactions += len(result.transactions)
                    total_llm_decisions += len(result.llm_decisions)
                    if keep_daily_results:
                        daily_results.append(result)

                results.append(
                    MatrixCellResult(
                        run_id=run_id,
                        cell_key=spec.key,
                        seed=seed,
                        is_cross_border=spec.cross_border,
                        num_currencies=len(env.currencies),
                        num_days_completed=num_days_completed,
                        total_transactions=total_transactions,
                        total_llm_decisions=total_llm_decisions,
                        daily_results=daily_results,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: one cell/seed's
                # failure must never abort the rest of the matrix (see this module's
                # docstring's "Memory, progress, and per-cell error recovery" section).
                session.rollback()
                failures.append((spec.key, seed, exc))
                continue

    return results, failures
