# Phase 2 LLM Negotiation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real, OpenRouter-backed LLM decision/negotiation path alongside Phase 1's deterministic rule-based path, plus two new utility functions (risk-neutral, EZ-inspired proxy), so the simulation can run live LLM-driven negotiations across 5 real models and produce an actual governance-prompting experiment result.

**Architecture:** Additive layering on top of the existing Phase 1 codebase — nothing in `src/utility/crra.py`, `cara.py`, `multi_attribute.py`, `src/negotiation/negotiation_engine.py`, or any existing test is modified. New code lives in `src/llm/`, a new `src/negotiation/llm_negotiation_engine.py`, two new `src/utility/*.py` files, new DB tables, and one implemented experiment script. The flow is: `AgentDecisionContext` → prompt render → `llm_router` (OpenRouter, retry/fallback) → `Decision` (pydantic) → `decision_adapter` (economic validation) → `llm_negotiation_engine` → existing deterministic validation/settlement → hallucination detection → DB + W&B.

**Tech Stack:** Python 3.12, pydantic v2, SQLAlchemy 2.0, pyyaml, httpx (new), pytest. No new templating library (str.format-based rendering). No new retry library (hand-rolled `RetryConfig`).

## Global Constraints

- Python >=3.12, pydantic>=2.6, sqlalchemy>=2.0 (from `pyproject.toml` — unchanged).
- New dependency: `httpx>=0.27`, added under a new `[project.optional-dependencies] llm` group (mirrors the existing `observability`/`market-data` groups) — not a core dependency.
- No hardcoded economic constants (fees, weights, scores, thresholds) — everything configurable, per `src/utils/constants.py`'s documented rule.
- `overpayment_pct(expected, paid)` must retain its exact existing signature and behavior: signed `(paid - expected) / expected * 100`, raises `ValueError` when `expected <= 0` (tested in `tests/test_hallucinations.py`, already committed — never change this contract).
- An LLM decision must never mutate a `Wallet`, `Ledger`, or `Transaction` directly — it only ever produces a `Decision` that flows through `src/llm/decision_adapter.py` and then the same deterministic validate/settle path Phase 1 already uses.
- All OpenRouter/Polygon HTTP calls are mocked in the default test suite — no real network calls except in tests explicitly marked `@pytest.mark.live`, which are skipped unless `RUN_LIVE_LLM_TESTS=1` is set.
- Verified OpenRouter model slugs (checked 2026-07-22): `anthropic/claude-sonnet-5`, `openai/gpt-5.6-luna`, `deepseek/deepseek-v4-pro`, `google/gemini-3.5-flash-lite`, `perplexity/sonar`.
- Full design rationale lives in `docs/superpowers/specs/2026-07-22-phase2-llm-negotiation-layer-design.md` — consult it for the "why" behind any task below.

---

## Task 1: RiskNeutralUtility

**Files:**
- Create: `src/utility/risk_neutral.py`
- Test: `tests/test_utility_risk_neutral.py`

**Interfaces:**
- Consumes: `src.utility.base.UtilityFunction` (ABC with `evaluate(self, option: CurrencyChainOption, **kwargs: float) -> float`), `src.blockchain.routing_engine.CurrencyChainOption` (fields: `currency_symbol: str`, `chain_name: str`, `governance_score: float`, `liquidity_score: float`, `peg_error: float`, `gas_fee: float`, `finality_seconds: float`, `genius_compliant: bool`).
- Produces: `class RiskNeutralUtility(UtilityFunction)` with `evaluate(self, option: CurrencyChainOption, wealth: float = 1.0, **kwargs: float) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utility_risk_neutral.py
from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.risk_neutral import RiskNeutralUtility


def _option(**overrides) -> CurrencyChainOption:
    defaults = dict(
        currency_symbol="USDC",
        chain_name="ethereum",
        governance_score=0.95,
        liquidity_score=0.97,
        peg_error=0.0003,
        gas_fee=2.5,
        finality_seconds=12.0,
        genius_compliant=True,
    )
    defaults.update(overrides)
    return CurrencyChainOption(**defaults)


def test_risk_neutral_is_linear_in_wealth():
    utility = RiskNeutralUtility()
    option = _option()

    u_100 = utility.evaluate(option, wealth=100.0)
    u_200 = utility.evaluate(option, wealth=200.0)

    # Linear (no curvature): doubling wealth must double the wealth-driven
    # component exactly, since the safety multiplier and gas fee are constant.
    safety = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
    assert u_200 - u_100 == (200.0 - 100.0) * safety


def test_risk_neutral_subtracts_gas_fee_directly():
    utility = RiskNeutralUtility()
    cheap = _option(gas_fee=0.5)
    expensive = _option(gas_fee=5.0)

    assert utility.evaluate(cheap, wealth=100.0) > utility.evaluate(expensive, wealth=100.0)
    # The gap must equal exactly the gas fee difference (no hidden curvature
    # or extra penalty beyond the raw fee).
    assert utility.evaluate(cheap, wealth=100.0) - utility.evaluate(expensive, wealth=100.0) == 4.5


def test_risk_neutral_ranks_by_safety_adjusted_payoff_not_safety_alone():
    utility = RiskNeutralUtility()
    safer_but_costly = _option(governance_score=0.99, gas_fee=50.0)
    riskier_but_cheap = _option(governance_score=0.50, gas_fee=0.1)

    # Risk-neutral cares only about net payoff, so a large enough fee can beat
    # a governance edge -- unlike CRRA/CARA, which apply curvature that can
    # reverse this ranking.
    assert utility.evaluate(riskier_but_cheap, wealth=100.0) > utility.evaluate(safer_but_costly, wealth=100.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utility_risk_neutral.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utility.risk_neutral'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/utility/risk_neutral.py
"""Risk-neutral utility: the un-shaped baseline the other utility functions
(CRRA, CARA, the EZ-inspired proxy) are measured against.

Deliberately linear -- no curvature, no risk preference. Evaluates raw net
economic payoff (safety-adjusted wealth minus gas fee) so that any deviation
CRRA/CARA/EZ show from this baseline is attributable to their risk-shaping,
not to a difference in what "payoff" means.
"""

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction


class RiskNeutralUtility(UtilityFunction):
    def evaluate(self, option: CurrencyChainOption, wealth: float = 1.0, **kwargs: float) -> float:
        safety_multiplier = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
        return wealth * safety_multiplier - option.gas_fee
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utility_risk_neutral.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/utility/risk_neutral.py tests/test_utility_risk_neutral.py
git commit -m "feat: add RiskNeutralUtility as an unshaped payoff baseline"
```

---

## Task 2: EpsteinZinProxyUtility

**Files:**
- Create: `src/utility/epstein_zin.py`
- Test: `tests/test_utility_epstein_zin.py`

**Interfaces:**
- Consumes: same `UtilityFunction`/`CurrencyChainOption` as Task 1.
- Produces: `class EpsteinZinProxyUtility(UtilityFunction)` with `__init__(self, risk_aversion: float, eis: float)` and `evaluate(self, option: CurrencyChainOption, wealth: float = 1.0, **kwargs: float) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utility_epstein_zin.py
import math

import pytest

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.epstein_zin import EpsteinZinProxyUtility


def _option(**overrides) -> CurrencyChainOption:
    defaults = dict(
        currency_symbol="USDC",
        chain_name="ethereum",
        governance_score=0.95,
        liquidity_score=0.97,
        peg_error=0.0003,
        gas_fee=2.5,
        finality_seconds=12.0,
        genius_compliant=True,
    )
    defaults.update(overrides)
    return CurrencyChainOption(**defaults)


def test_log_utility_at_unit_risk_aversion():
    utility = EpsteinZinProxyUtility(risk_aversion=1.0, eis=1.0)
    option = _option()

    safety = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
    effective_wealth = 100.0 * safety - option.gas_fee * (1.0 / 1.0)

    assert utility.evaluate(option, wealth=100.0) == pytest.approx(math.log(effective_wealth))


def test_zero_or_negative_eis_is_rejected():
    with pytest.raises(ValueError):
        EpsteinZinProxyUtility(risk_aversion=2.0, eis=0.0)
    with pytest.raises(ValueError):
        EpsteinZinProxyUtility(risk_aversion=2.0, eis=-1.0)


def test_risk_aversion_shapes_curvature_independently_of_eis():
    """Varying gamma (risk_aversion) with psi (eis) fixed must change the
    curvature over a *safety* difference, while leaving the ordering driven
    by fee differences untouched -- this is the design's independent-
    testability requirement for gamma."""
    low_gamma = EpsteinZinProxyUtility(risk_aversion=0.5, eis=1.0)
    high_gamma = EpsteinZinProxyUtility(risk_aversion=5.0, eis=1.0)

    safer = _option(governance_score=0.99, liquidity_score=0.99, peg_error=0.0001)
    riskier = _option(governance_score=0.60, liquidity_score=0.60, peg_error=0.02)

    low_gap = low_gamma.evaluate(safer, wealth=100.0) - low_gamma.evaluate(riskier, wealth=100.0)
    high_gap = high_gamma.evaluate(safer, wealth=100.0) - high_gamma.evaluate(riskier, wealth=100.0)

    # Higher risk aversion must widen the preference for the safer option.
    assert high_gap > low_gap


def test_eis_shapes_fee_sensitivity_independently_of_risk_aversion():
    """Varying psi (eis) with gamma (risk_aversion) fixed must change the gap
    between a cheap and an expensive option, while a fixed gamma keeps the
    safety-driven ordering direction unchanged -- the design's independent-
    testability requirement for psi."""
    reluctant_to_substitute = EpsteinZinProxyUtility(risk_aversion=2.0, eis=0.5)
    happy_to_substitute = EpsteinZinProxyUtility(risk_aversion=2.0, eis=5.0)

    cheap = _option(gas_fee=0.5)
    expensive = _option(gas_fee=20.0)

    low_eis_gap = reluctant_to_substitute.evaluate(cheap, wealth=100.0) - reluctant_to_substitute.evaluate(
        expensive, wealth=100.0
    )
    high_eis_gap = happy_to_substitute.evaluate(cheap, wealth=100.0) - happy_to_substitute.evaluate(
        expensive, wealth=100.0
    )

    # Low EIS (reluctant to substitute) must penalize the fee gap more
    # harshly than high EIS (happy to substitute).
    assert low_eis_gap > high_eis_gap > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utility_epstein_zin.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utility.epstein_zin'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/utility/epstein_zin.py
"""EZ-inspired static utility proxy.

This is not an Epstein-Zin utility function and should not be interpreted as
one in empirical results. It is an EZ-inspired static proxy designed to test
whether separately parameterizing risk aversion and an EIS-inspired
fee-sensitivity parameter changes choice behavior relative to CRRA. True
Epstein-Zin utility is recursive over a consumption stream and requires a
continuation value from future timesteps; this simulation's decision is a
single-period per-transaction choice, so no continuation value exists here.

`risk_aversion` (gamma) shapes curvature over the safety-adjusted payoff --
the same role it plays in CRRA. The constructor parameter `eis` (psi) is kept
under that name for compatibility with project_instructions.md's CRRA/CARA/EZ
framing, but the derived quantity `1 / eis` is named
`eis_inspired_fee_sensitivity` below, not `fee_sensitivity` alone: using
psi to scale a gas-fee penalty is a behavioral modeling choice, not a
derivation from formal EZ preferences over consumption.
"""

import math

from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction


class EpsteinZinProxyUtility(UtilityFunction):
    def __init__(self, risk_aversion: float, eis: float):
        if eis <= 0:
            raise ValueError("eis must be positive for the EZ-inspired proxy")
        self.risk_aversion = risk_aversion
        self.eis = eis

    def evaluate(self, option: CurrencyChainOption, wealth: float = 1.0, **kwargs: float) -> float:
        safety = option.governance_score * option.liquidity_score * (1.0 - option.peg_error)
        eis_inspired_fee_sensitivity = 1.0 / self.eis
        effective_wealth = max(wealth * safety - option.gas_fee * eis_inspired_fee_sensitivity, 1e-9)

        gamma = self.risk_aversion
        if abs(gamma - 1.0) < 1e-9:
            return math.log(effective_wealth)
        return (effective_wealth ** (1 - gamma)) / (1 - gamma)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utility_epstein_zin.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/utility/epstein_zin.py tests/test_utility_epstein_zin.py
git commit -m "feat: add EpsteinZinProxyUtility with independent risk/EIS axes"
```

---

## Task 3: Wire both utilities into utility_factory and agent profiles

**Files:**
- Modify: `src/utility/utility_factory.py`
- Modify: `src/agents/agent_factory.py:38-63` (`AgentProfileConfig`, `build_agent`)
- Test: `tests/test_utility_factory_phase2.py`

**Interfaces:**
- Consumes: `RiskNeutralUtility` (Task 1), `EpsteinZinProxyUtility` (Task 2, `__init__(risk_aversion, eis)`).
- Produces: `build_utility_function(utility_type: str, risk_aversion: float | None = None, weights: MultiAttributeWeights | None = None, eis: float | None = None) -> UtilityFunction` (new `eis` parameter, default `None`, positioned last so existing call sites with 2-3 positional/keyword args keep working); `AgentProfileConfig.utility_type` now `Literal["crra", "cara", "multi_attribute", "risk_neutral", "epstein_zin_proxy"]`; `AgentProfileConfig.eis: float | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utility_factory_phase2.py
import pytest

from src.utility.cara import CARAUtility
from src.utility.crra import CRRAUtility
from src.utility.epstein_zin import EpsteinZinProxyUtility
from src.utility.risk_neutral import RiskNeutralUtility
from src.utility.utility_factory import build_utility_function


def test_factory_builds_risk_neutral():
    utility = build_utility_function("risk_neutral")
    assert isinstance(utility, RiskNeutralUtility)


def test_factory_builds_epstein_zin_proxy():
    utility = build_utility_function("epstein_zin_proxy", risk_aversion=2.0, eis=0.8)
    assert isinstance(utility, EpsteinZinProxyUtility)
    assert utility.risk_aversion == 2.0
    assert utility.eis == 0.8


def test_factory_requires_eis_for_epstein_zin_proxy():
    with pytest.raises(ValueError):
        build_utility_function("epstein_zin_proxy", risk_aversion=2.0)


def test_factory_requires_risk_aversion_for_epstein_zin_proxy():
    with pytest.raises(ValueError):
        build_utility_function("epstein_zin_proxy", eis=0.8)


def test_existing_utility_types_still_work_unchanged():
    # Regression guard: Task 3 must not break the three Phase 1 utility types.
    assert isinstance(build_utility_function("crra", risk_aversion=3.0), CRRAUtility)
    assert isinstance(build_utility_function("cara", risk_aversion=0.8), CARAUtility)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utility_factory_phase2.py -v`
Expected: FAIL — `build_utility_function` raises `ValueError: Unknown utility_type: risk_neutral`

- [ ] **Step 3: Write minimal implementation**

Modify `src/utility/utility_factory.py` (full new file content):

```python
# src/utility/utility_factory.py
"""Assigns a concrete utility function from an agent profile's declared type.

Takes plain scalar/weights arguments rather than an AgentProfileConfig object
so src/utility has no dependency on src/agents -- agents call into utility,
not the reverse.
"""

from src.utility.base import UtilityFunction
from src.utility.cara import CARAUtility
from src.utility.crra import CRRAUtility
from src.utility.epstein_zin import EpsteinZinProxyUtility
from src.utility.multi_attribute import MultiAttributeUtility, MultiAttributeWeights
from src.utility.risk_neutral import RiskNeutralUtility


def build_utility_function(
    utility_type: str,
    risk_aversion: float | None = None,
    weights: MultiAttributeWeights | None = None,
    eis: float | None = None,
) -> UtilityFunction:
    if utility_type == "crra":
        if risk_aversion is None:
            raise ValueError("CRRA utility requires risk_aversion")
        return CRRAUtility(risk_aversion)
    if utility_type == "cara":
        if risk_aversion is None:
            raise ValueError("CARA utility requires risk_aversion")
        return CARAUtility(risk_aversion)
    if utility_type == "multi_attribute":
        return MultiAttributeUtility(weights or MultiAttributeWeights())
    if utility_type == "risk_neutral":
        return RiskNeutralUtility()
    if utility_type == "epstein_zin_proxy":
        if risk_aversion is None:
            raise ValueError("epstein_zin_proxy utility requires risk_aversion")
        if eis is None:
            raise ValueError("epstein_zin_proxy utility requires eis")
        return EpsteinZinProxyUtility(risk_aversion, eis)
    raise ValueError(f"Unknown utility_type: {utility_type}")
```

Modify `src/agents/agent_factory.py`: change line 42 and the `build_agent` call, and add the `eis` field. Full new file content:

```python
# src/agents/agent_factory.py
"""Builds concrete agent instances from configs/agent_profiles/*.yaml.

Not part of the original module list, but needed because profile files are
personality parameterizations (consumer, merchant, bank, institution,
investor) that don't map 1:1 onto the five agent classes -- an explicit
agent_class field in each profile picks the class to instantiate.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.agents.bank_agent import BankAgent
from src.agents.base_agent import BaseAgent
from src.agents.buyer_agent import BuyerAgent
from src.agents.investor_agent import InvestorAgent
from src.agents.regulator_agent import RegulatorAgent
from src.agents.seller_agent import SellerAgent
from src.agents.wallet import Wallet
from src.utility.multi_attribute import MultiAttributeWeights
from src.utility.utility_factory import build_utility_function
from src.utils.config_loader import load_yaml_dir_as
from src.utils.constants import CONFIG_ROOT
from src.utils.helpers import generate_id

AgentClass = Literal["buyer", "seller", "bank", "investor", "regulator"]

_AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "buyer": BuyerAgent,
    "seller": SellerAgent,
    "bank": BankAgent,
    "investor": InvestorAgent,
    "regulator": RegulatorAgent,
}


class AgentProfileConfig(BaseModel):
    name: str
    agent_class: AgentClass
    risk_tolerance: Literal["low", "medium", "high"]
    utility_type: Literal["crra", "cara", "multi_attribute", "risk_neutral", "epstein_zin_proxy"]
    risk_aversion: float | None = None
    eis: float | None = None
    weights: MultiAttributeWeights | None = None
    initial_wallet: dict[str, float] = {}


def load_agent_profiles(config_dir: Path = CONFIG_ROOT / "agent_profiles") -> dict[str, AgentProfileConfig]:
    return load_yaml_dir_as(config_dir, AgentProfileConfig)


def build_agent(profile: AgentProfileConfig) -> BaseAgent:
    agent_cls = _AGENT_CLASSES[profile.agent_class]
    utility_fn = build_utility_function(profile.utility_type, profile.risk_aversion, profile.weights, profile.eis)
    wallet = Wallet(balances=dict(profile.initial_wallet))
    return agent_cls(
        agent_id=generate_id(profile.agent_class),
        agent_class=profile.agent_class,
        profile_name=profile.name,
        risk_profile=profile.risk_tolerance,
        wallet=wallet,
        utility_fn=utility_fn,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utility_factory_phase2.py tests/test_agents.py tests/test_simulation.py -v`
Expected: all passed (the last two files are Task-1/2-unrelated regression checks that `agent_factory.py`'s edit didn't break Phase 1)

- [ ] **Step 5: Commit**

```bash
git add src/utility/utility_factory.py src/agents/agent_factory.py tests/test_utility_factory_phase2.py
git commit -m "feat: register risk_neutral and epstein_zin_proxy utility types"
```

---

## Task 4: Model roster config and static currency-profile configs

**Files:**
- Create: `configs/llm/models.yaml`
- Create: `configs/currencies/profiles/DAI.yaml`
- Create: `configs/currencies/profiles/EURC.yaml`
- Create: `configs/currencies/profiles/EURT.yaml`
- Create: `configs/currencies/profiles/FDUSD.yaml`
- Create: `configs/currencies/profiles/PAXG.yaml`
- Create: `configs/currencies/profiles/Tokenized_Deposits.yaml`
- Create: `configs/currencies/profiles/USDC.yaml`
- Create: `configs/currencies/profiles/USDT.yaml`
- Create: `configs/currencies/profiles/XAUT.yaml`

**Interfaces:**
- Produces: 10 YAML files on disk. No code in this task — Task 5 writes the loader and its tests, which are what actually exercise these files. This task has no test step of its own because there is no behavior to test yet (pure data); Task 5's tests are the verification for this task's content.

- [ ] **Step 1: Create the model roster config**

```yaml
# configs/llm/models.yaml
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
    fallbacks:
      - gpt-5.6-luna
      - deepseek-v4-pro
      - gemini-3.5-flash-lite
      - perplexity-sonar
  model_comparison:
    pinned_models:
      - claude-sonnet-5
      - gpt-5.6-luna
      - deepseek-v4-pro
      - gemini-3.5-flash-lite
      - perplexity-sonar
```

- [ ] **Step 2: Create `configs/currencies/profiles/DAI.yaml`**

```yaml
symbol: DAI
executive_summary: >-
  DAI is a decentralized USD-pegged stablecoin managed by MakerDAO (rebranded
  Sky). Launched Dec 2017 as Single-Collateral DAI, upgraded to
  Multi-Collateral DAI in Nov 2019. Backed by on-chain crypto collateral
  (ETH, WBTC, LINK, other stablecoins, tokenized US Treasuries) locked in
  Maker vaults, governed by MKR/SKY token holders. Circulating supply was
  ~4.7B by mid-2026 (combined with Sky's USDS, ~$13B).
timeline:
  - date: "2017-12-18"
    event: Launch of Single-Collateral DAI (SAI) on Ethereum
  - date: "2019-11-18"
    event: Multi-Collateral DAI (MCD) upgrade launched
  - date: "2020-03"
    event: "\"Black Thursday\" ETH crash; DAI briefly dipped to ~$0.90"
  - date: "2022-05"
    event: TerraUSD crash pressures the stablecoin market; DAI market cap fell from $8B to $6.3B
  - date: "2024-08-27"
    event: MakerDAO rebrands to Sky; announces USDS stablecoin
reserves_and_transparency: >-
  Reserve is on-chain crypto collateral, fully auditable via smart contracts
  (no off-chain custodian). Maker publishes audited reports; treasury
  includes yield-bearing assets like tokenized US Treasuries via Centrifuge.
  Risk parameters (collateralization ratios) are public via governance.
governance: >-
  Fully decentralized: MKR/SKY token holders vote on risk parameters,
  collateral types, and protocol changes via on-chain governance. No central
  issuer; issuance/burn is automated by smart contracts on vault lock/unlock.
price_and_market_cap: >-
  Soft peg to $1 via overcollateralization and stability fees; all-time high
  $3.67 (Nov 2019), low $0.897 (Mar 2023). Market cap peaked above $8B in
  2022, ~$4-5B by mid-2026.
crra_cara_note: "Not specified. MakerDAO manages risk via overcollateralization and adjustable stability fees, not CRRA/CARA terminology."
use_cases: >-
  DeFi collateral, lending/borrowing, trading, on-chain dollar exposure,
  liquidity for tokenized real-world-asset markets.
regulatory_and_controversies: >-
  Little direct regulatory enforcement given its decentralized structure.
  Notable: 2020 Black Thursday liquidation/oracle controversy; internal
  governance disputes; real-world-asset integration attracted scrutiny but is
  generally seen as regulator-friendly. No major sanctions or fines reported.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 3: Create `configs/currencies/profiles/EURC.yaml`**

```yaml
symbol: EURC
executive_summary: >-
  EURC is a euro-backed stablecoin issued by Circle, launched June 30 2022 on
  Ethereum, fully backed 1:1 by held euros/Eurozone assets, redeemable 1:1
  for EUR. Reserves held at regulated European banks, attested monthly.
  Expanded to Avalanche, Solana, Base, Cronos, Stellar. Operates under EU
  MiCA as an Electronic Money Token.
timeline:
  - date: "2022-06-30"
    event: EURC launches on Ethereum
  - date: "2022-10"
    event: EURC expands to Avalanche and other chains
  - date: "2023-09-26"
    event: EURC launches on Stellar
  - date: "2024-07"
    event: Circle obtains French EMI license (MiCA preparation)
  - date: "2025-11-26"
    event: Circle publishes MiCA whitepaper for EURC
reserves_and_transparency: >-
  Full-reserve model: every EURC backed by one euro or equivalent
  high-quality euro-denominated assets, held at regulated EU/EEA banks,
  segregated from operating funds, attested monthly by a Big Four firm
  (Deloitte). Weekly reserve breakdowns published.
governance: >-
  Centrally issued by Circle via its EU subsidiary; Centre Consortium
  (Circle + Coinbase) wound down Aug 2023, Circle became sole issuer. No
  on-chain governance; corporate board/management control issuance and
  redemption policy.
price_and_market_cap: >-
  Trades at ~EUR1 (~$1.10-1.15 following EUR/USD). Market cap ~EUR380M
  (~$436M) as of July 2026.
crra_cara_note: "Unused/unspecified."
use_cases: >-
  Euro liquidity in crypto: FX trading, euro-denominated DeFi lending,
  cross-border payments, institutional treasury management.
regulatory_and_controversies: >-
  Structured explicitly as an EU e-money token (EMT) under MiCA; proactively
  licensed (French EMI 2024); fully MiCA-compliant by 2026. No known
  enforcement actions; promoted as regulator-friendly.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 4: Create `configs/currencies/profiles/EURT.yaml`**

```yaml
symbol: EURT
executive_summary: >-
  EURT was Tether's euro-backed stablecoin, pegged 1:1 to EUR, launched Oct
  8 2020 on Ethereum. Claimed full euro-reserve backing with off-chain
  redemption via Tether. Usage was minor versus USD stablecoins; Tether wound
  down issuance in 2025.
timeline:
  - date: "2020-10-08"
    event: EURT launched on Ethereum
  - date: "2022"
    event: New EURT issuance mostly halts
  - date: "2025-10-27"
    event: Tether announces wind-down of EURT issuance/redemption (deadline Nov 27 2025)
  - date: "2025-11-27"
    event: Official redemption deadline for EURT holders
reserves_and_transparency: >-
  Claimed 100% EUR reserves, but composition was opaque and comingled with
  Tether's broader reserves; no dedicated external audit of EURT
  specifically, only broad Tether reserve reports.
governance: >-
  Centrally controlled by Tether Operations Ltd; no on-chain governance.
  Wind-down decision made by Tether executive management (Nov 2025
  statements from CEO Paolo Ardoino).
price_and_market_cap: >-
  Traded at ~EUR1 (~$1.10). Market cap peaked modestly (~EUR250M supply);
  by 2026, market cap was very small (~$0.2M) ahead of wind-down.
crra_cara_note: "Unused/unspecified."
use_cases: >-
  Euro-denominated crypto trading pairs and limited on-chain euro liquidity
  within Tether's ecosystem; adoption was low relative to USDT.
regulatory_and_controversies: >-
  Tether cited "lack of a risk-averse regulatory framework in Europe" as the
  reason for winding down issuance. General Tether controversies (2017
  reserve claims, 2021 NYAG settlement) apply by extension, not specific to
  EURT. Not redeemable in euros after Nov 27 2025.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 5: Create `configs/currencies/profiles/FDUSD.yaml`**

```yaml
symbol: FDUSD
executive_summary: >-
  First Digital USD (FDUSD) is a USD-backed stablecoin issued by First
  Digital Trust (Hong Kong), launched 2023 with full-reserve backing.
  Redeemable 1:1 for USD held in trust custody (cash, short-term US
  Treasuries, deposits), monthly third-party attestations. Marketed for
  institutional and cross-border use, especially in Asia.
timeline:
  - date: "2023-04-28"
    event: FDUSD launched on Ethereum
  - date: "2023-05-02"
    event: FDUSD launched on BNB Smart Chain
  - date: "2024-01-31"
    event: First attestation report (1Q24) reports $2.59B reserves
  - date: "2023-2025"
    event: Gradual expansion to Arbitrum, Solana, Sui, TON
reserves_and_transparency: >-
  Fully backed by cash and cash equivalents held by a licensed custodian
  (Legacy Trust Company). Monthly attestations by an independent auditor
  (Prescient Assurance); a Jan 2024 report showed $2.59B reserves (~59%
  T-bills, ~22% fixed deposits, ~15% USD cash, ~4% repos).
governance: >-
  Centralized under First Digital Trust (HK); trustee and parent group
  govern issuance/redemption. No DAO or token governance; compliance
  overseen by the trust board.
price_and_market_cap: >-
  Stayed very close to $1 ($0.997-$1.00). Circulating supply reached ~$3B by
  2026.
crra_cara_note: "Unused/unspecified."
use_cases: >-
  Institutional cross-border payments, remittances, treasury management,
  DeFi lending/trading integrations, particularly Asia-focused.
regulatory_and_controversies: >-
  Compliance-first positioning; regulated as a Hong Kong trust company
  subject to AML/investor-protection law. No controversies reported;
  general new-entrant adoption/liquidity risk applies.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 6: Create `configs/currencies/profiles/PAXG.yaml`**

```yaml
symbol: PAXG
executive_summary: >-
  PAXG is a gold-backed token by Paxos Trust Company (NY), launched Sept 5
  2019. Each PAXG token represents one fine troy ounce of London Good
  Delivery gold held in London vaults, redeemable 1:1 for gold or fiat.
  Market cap ~$1.8-2.0B as of 2026 (~450k tokens circulating).
timeline:
  - date: "2019-09-05"
    event: PAXG launched (NYDFS-approved)
  - date: "2020-08-26"
    event: PAXG listed on Binance
  - date: "2021"
    event: PAXG added on other exchanges and networks
  - date: "2024"
    event: Continual monthly audit attestations by Withum
reserves_and_transparency: >-
  100% backed by physical LBMA-grade gold in professional London vaults
  (e.g. Brink's). Monthly attestations by Withum matching PAXG supply to
  ounces in custody, published publicly; holders can verify entitlement via
  an Ethereum lookup tool. Custody regulated by NYDFS oversight.
governance: >-
  Centrally governed by Paxos Trust Company (NY state-chartered trust,
  NYDFS-regulated board-approved policy). No DAO; token holders have no
  governance rights beyond redemption.
price_and_market_cap: >-
  Price tracks gold spot (~$4,100-4,200/oz in 2026). All-time high $5,619,
  low $1,399. Market cap ~450k tokens x ~$4,100 ~= $1.85B.
crra_cara_note: "Unused/unspecified."
use_cases: >-
  Digital gold ownership, portfolio diversification, DeFi collateral, quick
  settlement of gold trades, inflation hedging.
regulatory_and_controversies: >-
  Regulated under NYDFS; audited monthly; avoided major controversy. Paxos's
  separate USD stablecoin (Paxos Standard/PAX) was shut down by NYDFS in
  2023, but PAXG itself remained active and unaffected.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 7: Create `configs/currencies/profiles/Tokenized_Deposits.yaml`**

```yaml
symbol: Tokenized_Deposits
executive_summary: >-
  "Tokenized deposits" is a general category: digital tokens representing
  bank deposits at regulated institutions, pegged 1:1 to fiat and fully
  reserved by customer deposits/equivalents. Under EU MiCA, most such tokens
  are classified as electronic money tokens (EMTs) requiring e-money issuer
  licensing. Unlike algorithmic or crypto-collateralized stablecoins, this
  category relies on trusted banking relationships.
timeline: []
reserves_and_transparency: >-
  Reserves are exactly equal to the stated deposits, held in bank accounts
  or regulated funds (cash or T-bills). Transparency is often high (subject
  to bank audits); under MiCA, monthly attestations are mandated for EMTs.
governance: >-
  Centralized under the issuing bank/fintech, governed by banking
  regulation. Decision-making follows corporate/regulatory frameworks, not
  on-chain voting.
price_and_market_cap: >-
  Category-level: pegged 1:1 to the referenced fiat currency by design; no
  single price/market-cap series applies across the category.
crra_cara_note: "None of the surveyed tokens explicitly use CRRA/CARA terms; this category is fully reserved / overcollateralized, with no algorithmic risk-return structure."
use_cases: >-
  Digital-form replacement for traditional deposits: payments, tokenized
  lending, collateral, bridging banking and blockchain rails (e.g. tokenized
  euro/yen).
regulatory_and_controversies: >-
  Typically requires banking or e-money licenses (e.g. Circle's EURC/USDC
  are EMTs under EU law). Risks include banking-regulation changes,
  interest-rate risk on bond-held reserves, and general compliance burden.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 8: Create `configs/currencies/profiles/USDC.yaml`**

```yaml
symbol: USDC
executive_summary: >-
  USDC is a USD-backed stablecoin issued by Circle since Sept 2018, 100%
  backed by cash and short-term US Treasuries. The largest fully-regulated
  stablecoin, used globally for payments, trading, and DeFi. Mint-and-burn
  on demand; grew from ~$33B supply in early 2024 to ~$60B by Q1 2026.
timeline:
  - date: "2018-09"
    event: USDC launched by Circle and Coinbase (Centre consortium)
  - date: "2021-03"
    event: SVB banking crisis causes a brief USDC depeg to $0.87
  - date: "2023-08"
    event: Centre consortium wound down; Circle becomes sole issuer
  - date: "2024-07"
    event: Circle obtains French EMI license for USDC/EURC
  - date: "2025-06-05"
    event: Circle IPO on NYSE (CRCL); reserve disclosures in SEC filings
reserves_and_transparency: >-
  100% collateralized by cash and short-duration US Treasuries; ~80% sits in
  the BlackRock-managed Circle Reserve Fund, rest in cash at large banks.
  Daily public breakdown via BlackRock; monthly Deloitte attestations.
  Considered industry-leading transparency.
governance: >-
  Centrally managed by Circle (US corporation, NYSE-listed post-IPO).
  Issuance/reserve-policy decisions made by Circle's executive team; no
  token-holder governance.
price_and_market_cap: >-
  Trades at ~$1 by design; only major dip was the SVB weekend (Mar 2023) to
  $0.87. Supply ~$60B by Q1 2026, second-largest stablecoin behind USDT.
crra_cara_note: "Unused/unspecified."
use_cases: >-
  General-purpose digital dollar: trading (esp. US-regulated exchanges),
  DeFi lending/borrowing, cross-border payments, institutional treasury
  management. Reserve yield accrues to Circle, not holders.
regulatory_and_controversies: >-
  Licensed in nearly all US states, EU Electronic Money Institution; fits
  the US GENIUS Act (2025) and EU MiCA. No major enforcement actions; SVB
  incident was a stress test resolved without regulator action.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 9: Create `configs/currencies/profiles/USDT.yaml`**

```yaml
symbol: USDT
executive_summary: >-
  USDT (Tether USD) is the largest USD-pegged stablecoin by market cap
  (~$100-130B), launched 2014, issued by Tether Operations Ltd. Pegged 1:1
  to USD by claim; reserves held off-chain and historically less
  transparent than USDC's, though composition has shifted toward cash and
  T-bills in recent years.
timeline:
  - date: "2014-10"
    event: Tether (USDT) launched (initially Bitcoin Omni Layer, later Ethereum, Tron, etc.)
  - date: "2021-02"
    event: NYAG settlement -- $18.5M fine, new transparency commitments
  - date: "2024"
    event: DOJ investigation into AML compliance reported
  - date: "2026-03-24"
    event: Tether announces a Big Four audit of USDT reserves
reserves_and_transparency: >-
  Claims 100% backing; historically held commercial paper and funded
  receivables considered lower-grade, shifted toward cash/T-bills after
  2022. Publishes daily circulation and periodic (quarterly) attestations;
  full independent audit was still in progress as of the Mar 2026
  announcement, so some reserve-composition uncertainty persists.
governance: >-
  Centrally governed by Tether Operations Ltd (Hong Kong entity, under
  Bitfinex ownership); no decentralized governance or token-holder
  transparency into policy decisions.
price_and_market_cap: >-
  Traded within cents of $1 (briefly ~$1.0005 in May 2023 on high demand).
  Market cap grew from ~$60B (2021) to $100B+ (2024+), ~60% crypto-market
  stablecoin share.
crra_cara_note: "Unused/unspecified."
use_cases: >-
  De facto on/off-ramp for crypto trading pairs and DeFi worldwide,
  especially in emerging markets; remittances and off-chain settlement.
regulatory_and_controversies: >-
  History of controversy: 2017-2021 allegations of unbacked issuance, 2021
  NYAG settlement ($18.5M fine, admitted "receivables" in reserves), 2024
  reports of DOJ AML/sanctions investigation. Widely used but carries
  ongoing counterparty/transparency risk versus USDC.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 10: Create `configs/currencies/profiles/XAUT.yaml`**

```yaml
symbol: XAUT
executive_summary: >-
  XAUT (Tether Gold) is Tether's gold-backed token, launched Aug 2020. Each
  token represents one fine troy ounce of allocated gold held by Tether
  (claimed ~140 metric tons / ~$23B by 2026). Trades tracking gold's USD
  spot price; market cap on the order of $10-12B.
timeline:
  - date: "2020-08"
    event: Tether Gold (XAUT) launched
  - date: "2020-08-26"
    event: XAUT listed on Binance
  - date: "2022-2024"
    event: Tether continues acquiring gold (140+ tons by mid-2026)
  - date: "2026-06-27"
    event: XAUT integrated with Ledn for gold-backed loans
  - date: "2026-07-21"
    event: XAUT recognized as an accepted commodity in Abu Dhabi Global Market (ADGM)
reserves_and_transparency: >-
  Physical gold bullion in vaults (e.g. Switzerland). Tether claims each
  XAUT is backed by a specific ounce held, but unlike PAXG, Tether does not
  publish regular independent external audits for the gold reserve --
  transparency is lower, relying on Tether's own periodic disclosures.
governance: >-
  Centrally managed by Tether Operations Ltd; no DAO or independent trustee
  for the gold. Mechanics are entirely internal to Tether's operations.
price_and_market_cap: >-
  Price tracks gold spot (~$4,100/oz mid-2026). Market cap roughly $10-12B,
  lower volatility than most crypto but mirrors gold's swings.
crra_cara_note: "Unused/unspecified."
use_cases: >-
  Digital gold custody and transfer, gold-backed lending (e.g. via Ledn),
  fiat/gold arbitrage without moving physical bullion, DeFi collateral for
  protocols accepting tokenized gold.
regulatory_and_controversies: >-
  Complex/uneven regulatory status; ADGM (Abu Dhabi) acceptance in 2026
  suggests some jurisdictional approval. No specific enforcement reported,
  but shares Tether's parent-company scrutiny and counterparty-trust risk in
  its large gold-holding claims.
source: deep-research-report.md
report_date: "2026-07"
```

- [ ] **Step 11: Commit**

```bash
git add configs/llm/models.yaml configs/currencies/profiles/
git commit -m "feat: add LLM model roster config and static stablecoin profile corpus"
```

---

## Task 5: Static currency-profile loader

**Files:**
- Create: `src/llm/market_intelligence.py`
- Test: `tests/test_market_intelligence.py`

**Interfaces:**
- Consumes: `src.utils.config_loader.load_yaml_as(path: Path, model: type[T]) -> T`, `src.utils.constants.CONFIG_ROOT`, the 9 files from Task 4.
- Produces: `class TimelineEvent(BaseModel)` (`date: str`, `event: str`); `class CurrencyProfile(BaseModel)` (`symbol`, `executive_summary`, `timeline: list[TimelineEvent]`, `reserves_and_transparency`, `governance`, `price_and_market_cap`, `crra_cara_note`, `use_cases`, `regulatory_and_controversies`, `source`, `report_date`, all `str` except `timeline`); `load_currency_profile(symbol: str, profiles_dir: Path = PROFILES_DIR) -> CurrencyProfile | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market_intelligence.py
from src.llm.market_intelligence import CurrencyProfile, load_currency_profile

_ALL_SYMBOLS = ["DAI", "EURC", "EURT", "FDUSD", "PAXG", "Tokenized_Deposits", "USDC", "USDT", "XAUT"]


def test_loads_every_currency_profile_file():
    for symbol in _ALL_SYMBOLS:
        profile = load_currency_profile(symbol)
        assert profile is not None
        assert isinstance(profile, CurrencyProfile)
        assert profile.symbol == symbol
        assert profile.executive_summary
        assert profile.source == "deep-research-report.md"


def test_missing_profile_returns_none_not_an_exception():
    assert load_currency_profile("NOTACOIN") is None


def test_usdc_profile_has_expected_timeline_entries():
    profile = load_currency_profile("USDC")
    assert profile is not None
    assert len(profile.timeline) >= 3
    assert any("Circle" in event.event or "Coinbase" in event.event for event in profile.timeline)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_market_intelligence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm.market_intelligence'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm/market_intelligence.py
"""Feeds real-world stablecoin context into LLM prompts.

Two clearly separate sources: a static, git-versioned profile corpus (this
module's load_currency_profile, compiled from deep-research-report.md) and
an optional live price snapshot (added in a later task, via Polygon). The
static corpus must be presented to the LLM as background/historical
information, not current market state -- see the design doc's §6.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

PROFILES_DIR = CONFIG_ROOT / "currencies" / "profiles"


class TimelineEvent(BaseModel):
    date: str
    event: str


class CurrencyProfile(BaseModel):
    symbol: str
    executive_summary: str
    timeline: list[TimelineEvent] = Field(default_factory=list)
    reserves_and_transparency: str
    governance: str
    price_and_market_cap: str
    crra_cara_note: str
    use_cases: str
    regulatory_and_controversies: str
    source: str
    report_date: str


def load_currency_profile(symbol: str, profiles_dir: Path = PROFILES_DIR) -> CurrencyProfile | None:
    """Return the static profile for symbol, or None if no profile file exists.

    None (not an exception) on a missing file: a currency without a curated
    profile must degrade gracefully in the LLM context rather than crash the
    decision pipeline -- the same principle the live-price fetch (added
    later in this module) also follows.
    """
    path = profiles_dir / f"{symbol}.yaml"
    if not path.exists():
        return None
    return load_yaml_as(path, CurrencyProfile)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_market_intelligence.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm/market_intelligence.py tests/test_market_intelligence.py
git commit -m "feat: load static per-currency profile corpus for LLM context"
```

---

## Task 6: LLM decision and market-snapshot persistence

**Files:**
- Modify: `database/models.py`
- Modify: `database/repository.py`
- Test: `tests/test_llm_persistence.py`

**Interfaces:**
- Consumes: existing `Base` (DeclarativeBase) from `database/models.py`.
- Produces: ORM classes `LLMDecisionRecord` (table `llm_decisions`) and `MarketSnapshotRecord` (table `market_snapshots`) in `database/models.py`; plain pydantic input contracts `LLMDecisionLogEntry` and `MarketSnapshotLogEntry` plus `class LLMDecisionRepository` (`record(self, entry: LLMDecisionLogEntry) -> None`) and `class MarketSnapshotRepository` (`record(self, entry: MarketSnapshotLogEntry) -> None`) in `database/repository.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_persistence.py
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, LLMDecisionRecord, MarketSnapshotRecord
from database.repository import (
    LLMDecisionLogEntry,
    LLMDecisionRepository,
    MarketSnapshotLogEntry,
    MarketSnapshotRepository,
)


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_llm_decision_repository_persists_full_record():
    session = _session()
    repo = LLMDecisionRepository(session)
    entry = LLMDecisionLogEntry(
        decision_id="dec-1",
        simulation_id="sim-1",
        timestep=3,
        agent_id="buyer-1",
        agent_type="buyer",
        requested_model="claude-sonnet-5",
        actual_model="claude-sonnet-5",
        fallback_used=False,
        fallback_reason=None,
        model_attempts=["claude-sonnet-5"],
        prompt_version="buyer_prompt@v1",
        rendered_prompt_hash="abc123",
        action="OFFER",
        currency="USDC",
        chain="ethereum",
        amount=100.0,
        price=99.5,
        reported_reasoning="USDC offers the best governance/liquidity trade-off.",
        negotiation_id="neg-1",
        round=1,
        risk_profile="low",
        utility_type="crra",
        utility_parameters={"risk_aversion": 3.0},
        scenario="baseline",
        domestic_or_cross_border="domestic",
        governance_prompt_enabled=True,
    )

    repo.record(entry)
    session.commit()

    rows = session.query(LLMDecisionRecord).all()
    assert len(rows) == 1
    assert rows[0].decision_id == "dec-1"
    assert rows[0].actual_model == "claude-sonnet-5"
    assert rows[0].model_attempts == ["claude-sonnet-5"]
    assert rows[0].fallback_used is False


def test_market_snapshot_repository_persists_and_allows_missing_price():
    session = _session()
    repo = MarketSnapshotRepository(session)
    entry = MarketSnapshotLogEntry(source="polygon", ticker="X:USDCUSD", price=None, data_window="live", negotiation_id="neg-1")

    repo.record(entry)
    session.commit()

    rows = session.query(MarketSnapshotRecord).all()
    assert len(rows) == 1
    assert rows[0].price is None
    assert rows[0].ticker == "X:USDCUSD"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_persistence.py -v`
Expected: FAIL with `ImportError: cannot import name 'LLMDecisionRecord' from 'database.models'`

- [ ] **Step 3: Write minimal implementation**

Modify `database/models.py`: change the import line and append two new classes at the end of the file.

```python
# database/models.py (line 12 changes from:)
from sqlalchemy import Float, ForeignKey, Integer, String
# to:
from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
```

Append to the end of `database/models.py`:

```python
class LLMDecisionRecord(Base):
    """Every LLM decision, whether it came from the primary model, a fallback,
    or a same-model economic-correction reprompt (see fallback_used /
    fallback_reason / model_attempts)."""

    __tablename__ = "llm_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(String)
    simulation_id: Mapped[str] = mapped_column(String)
    timestep: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"))
    agent_type: Mapped[str] = mapped_column(String)
    requested_model: Mapped[str] = mapped_column(String)
    actual_model: Mapped[str] = mapped_column(String)
    fallback_used: Mapped[bool] = mapped_column(Boolean)
    fallback_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    model_attempts: Mapped[list] = mapped_column(JSON)
    prompt_version: Mapped[str] = mapped_column(String)
    rendered_prompt_hash: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    currency: Mapped[str] = mapped_column(String)
    chain: Mapped[str] = mapped_column(String)
    amount: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    reported_reasoning: Mapped[str] = mapped_column(String)
    negotiation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    round: Mapped[int] = mapped_column(Integer)
    risk_profile: Mapped[str] = mapped_column(String)
    utility_type: Mapped[str] = mapped_column(String)
    utility_parameters: Mapped[dict] = mapped_column(JSON)
    scenario: Mapped[str] = mapped_column(String)
    domestic_or_cross_border: Mapped[str] = mapped_column(String)
    governance_prompt_enabled: Mapped[bool] = mapped_column(Boolean)
    timestamp: Mapped[datetime] = mapped_column(DateTime)


class MarketSnapshotRecord(Base):
    """A timestamped external market-data fetch (Polygon live price, or the
    static profile corpus's report_date) shown to an LLM -- persisted so a
    later re-run can see exactly what data the model was shown."""

    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    retrieval_timestamp: Mapped[datetime] = mapped_column(DateTime)
    source: Mapped[str] = mapped_column(String)
    ticker: Mapped[str] = mapped_column(String)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_window: Mapped[str | None] = mapped_column(String, nullable=True)
    negotiation_id: Mapped[str | None] = mapped_column(String, nullable=True)
```

Modify `database/repository.py`: add imports and two new repository classes. Add to the top import block:

```python
from database.models import (
    AgentRecord,
    LLMDecisionRecord,
    MarketSnapshotRecord,
    MetricRecord,
    NegotiationRecord,
    TransactionRecord,
    WalletRecord,
)
```

(replacing the existing narrower import line), and add near the top of the file, after the existing imports:

```python
from pydantic import BaseModel


class LLMDecisionLogEntry(BaseModel):
    decision_id: str
    simulation_id: str
    timestep: int
    agent_id: str
    agent_type: str
    requested_model: str
    actual_model: str
    fallback_used: bool
    fallback_reason: str | None
    model_attempts: list[str]
    prompt_version: str
    rendered_prompt_hash: str
    action: str
    currency: str
    chain: str
    amount: float
    price: float
    reported_reasoning: str
    negotiation_id: str | None
    round: int
    risk_profile: str
    utility_type: str
    utility_parameters: dict
    scenario: str
    domestic_or_cross_border: str
    governance_prompt_enabled: bool


class MarketSnapshotLogEntry(BaseModel):
    source: str
    ticker: str
    price: float | None
    data_window: str | None
    negotiation_id: str | None = None
```

Then append the two repository classes at the end of the file (after `MetricsRepository`, before `persist_timestep`):

```python
class LLMDecisionRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: LLMDecisionLogEntry) -> None:
        self.session.add(
            LLMDecisionRecord(
                **entry.model_dump(),
                timestamp=datetime.now(timezone.utc),
            )
        )


class MarketSnapshotRepository:
    def __init__(self, session: Session):
        self.session = session

    def record(self, entry: MarketSnapshotLogEntry) -> None:
        self.session.add(
            MarketSnapshotRecord(
                **entry.model_dump(),
                retrieval_timestamp=datetime.now(timezone.utc),
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_persistence.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -v`
Expected: all previously-passing tests still pass (this task only adds to `database/models.py`/`repository.py`, it doesn't remove or rename anything existing)

- [ ] **Step 6: Commit**

```bash
git add database/models.py database/repository.py tests/test_llm_persistence.py
git commit -m "feat: persist LLM decisions and market snapshots"
```

---

## Task 7: Structured Decision schema and economic validation

**Files:**
- Create: `src/llm/decision_schema.py`
- Test: `tests/test_decision_schema.py`

**Interfaces:**
- Produces: `class DecisionAction(str, Enum)` (`OFFER`, `COUNTER_OFFER`, `ACCEPT`, `REJECT`, `WALK_AWAY`); `class Decision(BaseModel)` (`action: DecisionAction`, `proposed_currency: str`, `proposed_chain: str`, `amount: float`, `price: float`, `reasoning: str`, `confidence: float | None = None`, `utility_estimate: float | None = None`, `risk_assessment: str | None = None`, `preferred_alternative_currency: str | None = None`, `preferred_alternative_chain: str | None = None`); `class DecisionValidationResult(BaseModel)` (`is_valid: bool`, `reason: str | None = None`); `validate_decision(decision: Decision, supported_currencies: set[str], supported_chains: set[str], wallet_balances: dict[str, float]) -> DecisionValidationResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decision_schema.py
from src.llm.decision_schema import Decision, DecisionAction, validate_decision


def _decision(**overrides) -> Decision:
    defaults = dict(
        action=DecisionAction.OFFER,
        proposed_currency="USDC",
        proposed_chain="ethereum",
        amount=1.0,
        price=100.0,
        reasoning="test reasoning",
    )
    defaults.update(overrides)
    return Decision(**defaults)


def test_valid_offer_passes():
    result = validate_decision(_decision(), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is True


def test_unsupported_currency_rejected():
    result = validate_decision(_decision(proposed_currency="NOTACOIN"), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is False
    assert "currency" in result.reason.lower()


def test_unsupported_chain_rejected():
    result = validate_decision(_decision(proposed_chain="notachain"), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is False
    assert "chain" in result.reason.lower()


def test_nonpositive_amount_rejected():
    result = validate_decision(_decision(amount=0.0), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is False


def test_nonpositive_price_rejected():
    result = validate_decision(_decision(price=-5.0), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})
    assert result.is_valid is False


def test_accept_with_insufficient_funds_rejected():
    result = validate_decision(
        _decision(action=DecisionAction.ACCEPT, price=500.0), {"USDC"}, {"ethereum"}, {"USDC": 100.0}
    )
    assert result.is_valid is False
    assert "funds" in result.reason.lower()


def test_reject_and_walk_away_are_always_valid_regardless_of_funds():
    assert validate_decision(_decision(action=DecisionAction.REJECT), {"USDC"}, {"ethereum"}, {}).is_valid is True
    assert validate_decision(_decision(action=DecisionAction.WALK_AWAY), {"USDC"}, {"ethereum"}, {}).is_valid is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_decision_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm.decision_schema'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm/decision_schema.py
"""Structured LLM decision output and its economic-validity check.

The LLM proposes; it never mutates state. Decision is the schema every model
must fill in (phase_2_instructions_v2.md §4C). validate_decision is the
"economically invalid" tier of the three-tier failure handling in
llm_router.py -- distinct from JSON/schema malformation, which llm_router
itself repairs before a Decision object exists at all.
"""

from enum import Enum

from pydantic import BaseModel


class DecisionAction(str, Enum):
    OFFER = "OFFER"
    COUNTER_OFFER = "COUNTER_OFFER"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    WALK_AWAY = "WALK_AWAY"


class Decision(BaseModel):
    action: DecisionAction
    proposed_currency: str
    proposed_chain: str
    amount: float
    price: float
    reasoning: str
    confidence: float | None = None
    utility_estimate: float | None = None
    risk_assessment: str | None = None
    preferred_alternative_currency: str | None = None
    preferred_alternative_chain: str | None = None


class DecisionValidationResult(BaseModel):
    is_valid: bool
    reason: str | None = None


def validate_decision(
    decision: Decision,
    supported_currencies: set[str],
    supported_chains: set[str],
    wallet_balances: dict[str, float],
) -> DecisionValidationResult:
    if decision.action in (DecisionAction.REJECT, DecisionAction.WALK_AWAY):
        return DecisionValidationResult(is_valid=True)
    if decision.proposed_currency not in supported_currencies:
        return DecisionValidationResult(is_valid=False, reason=f"Unsupported currency: {decision.proposed_currency}")
    if decision.proposed_chain not in supported_chains:
        return DecisionValidationResult(is_valid=False, reason=f"Unsupported chain: {decision.proposed_chain}")
    if decision.amount <= 0:
        return DecisionValidationResult(is_valid=False, reason="Amount must be positive")
    if decision.price <= 0:
        return DecisionValidationResult(is_valid=False, reason="Price must be positive")
    if decision.action == DecisionAction.ACCEPT:
        available = wallet_balances.get(decision.proposed_currency, 0.0)
        if available < decision.price:
            return DecisionValidationResult(is_valid=False, reason="Insufficient funds")
    return DecisionValidationResult(is_valid=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_decision_schema.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm/decision_schema.py tests/test_decision_schema.py
git commit -m "feat: add structured Decision schema and economic validation"
```

---

## Task 8: httpx dependency, model roster loader, and OpenRouter preflight check

**Files:**
- Modify: `pyproject.toml:15-19`
- Create: `src/llm/llm_router.py` (replaces the `NotImplementedError` stub)
- Test: `tests/test_llm_router.py` (this file grows further in Tasks 9-11)

**Interfaces:**
- Consumes: `configs/llm/models.yaml` (Task 4), `src.utils.config_loader.load_yaml_as`.
- Produces: `class ModelEntry(BaseModel)` (`id: str`, `label: str`); `class ReliabilityChain(BaseModel)` (`primary: str`, `fallbacks: list[str]`); `class ModelComparisonPolicy(BaseModel)` (`pinned_models: list[str]`); `class RoutingPolicies(BaseModel)` (`default_reliability_chain: ReliabilityChain`, `model_comparison: ModelComparisonPolicy`); `class ModelRosterConfig(BaseModel)` (`models: list[ModelEntry]`, `routing_policies: RoutingPolicies`, method `resolve(self, label: str) -> str`); `load_model_roster(path: Path = MODELS_CONFIG_PATH) -> ModelRosterConfig`; `class ModelNotAvailableError(Exception)` (attributes `label: str`, `model_id: str`); `verify_model_roster(roster: ModelRosterConfig, client: httpx.Client) -> None`.

- [ ] **Step 1: Add httpx as an optional dependency**

Modify `pyproject.toml`, changing the `[project.optional-dependencies]` block from:

```toml
[project.optional-dependencies]
# Not required for Phase 1 core simulation -- only needed for the optional
# integrations in metrics/wandb_logger.py and scripts/calibrate_currency_configs.py.
observability = ["wandb>=0.17"]
market-data = ["requests>=2.31"]
```

to:

```toml
[project.optional-dependencies]
# Not required for Phase 1 core simulation -- only needed for the optional
# integrations in metrics/wandb_logger.py and scripts/calibrate_currency_configs.py.
observability = ["wandb>=0.17"]
market-data = ["requests>=2.31"]
# Phase 2: OpenRouter (src/llm/llm_router.py) and Polygon
# (src/llm/market_intelligence.py) both go over HTTP via httpx.
llm = ["httpx>=0.27"]
```

- [ ] **Step 2: Install the new dependency into the dev environment**

Run: `pip install -e ".[llm,dev]"`
Expected: `httpx` installs successfully alongside the existing dev dependencies

- [ ] **Step 3: Write the failing test**

```python
# tests/test_llm_router.py
import httpx
import pytest

from src.llm.llm_router import ModelNotAvailableError, load_model_roster, verify_model_roster


def _client_with_models(available_ids: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/models"
        return httpx.Response(200, json={"data": [{"id": model_id} for model_id in available_ids]})

    return httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))


def test_loads_roster_from_config():
    roster = load_model_roster()
    labels = {entry.label for entry in roster.models}
    assert labels == {"claude-sonnet-5", "gpt-5.6-luna", "deepseek-v4-pro", "gemini-3.5-flash-lite", "perplexity-sonar"}
    assert roster.routing_policies.default_reliability_chain.primary == "claude-sonnet-5"
    assert roster.routing_policies.model_comparison.pinned_models == [
        "claude-sonnet-5",
        "gpt-5.6-luna",
        "deepseek-v4-pro",
        "gemini-3.5-flash-lite",
        "perplexity-sonar",
    ]


def test_resolve_looks_up_id_by_label():
    roster = load_model_roster()
    assert roster.resolve("claude-sonnet-5") == "anthropic/claude-sonnet-5"


def test_resolve_raises_for_unknown_label():
    roster = load_model_roster()
    with pytest.raises(ValueError):
        roster.resolve("not-a-real-model")


def test_verify_model_roster_passes_when_all_ids_available():
    roster = load_model_roster()
    all_ids = [entry.id for entry in roster.models]
    client = _client_with_models(all_ids)

    verify_model_roster(roster, client)  # must not raise


def test_verify_model_roster_fails_loudly_on_missing_model():
    roster = load_model_roster()
    ids_missing_one = [entry.id for entry in roster.models if entry.label != "gpt-5.6-luna"]
    client = _client_with_models(ids_missing_one)

    with pytest.raises(ModelNotAvailableError) as exc_info:
        verify_model_roster(roster, client)

    assert exc_info.value.label == "gpt-5.6-luna"
    assert exc_info.value.model_id == "openai/gpt-5.6-luna"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_llm_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm.llm_router'` (this replaces the existing 10-line `NotImplementedError` stub at `src/llm/llm_router.py`, so there is no previous passing test to regress)

- [ ] **Step 5: Write minimal implementation**

```python
# src/llm/llm_router.py
"""OpenRouter-backed LLM routing.

Model roster (what's available) and routing policy (how it's used) are
deliberately separate concepts (configs/llm/models.yaml) so a model's
identity is never implicitly read as "better" than another's -- see the
design doc §4. This file grows across Tasks 8-11: roster loading and the
OpenRouter preflight check here; the actual chat-completion call, retry,
and fallback-chain logic are added in Tasks 9-10.
"""

from pathlib import Path

import httpx
from pydantic import BaseModel

from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

MODELS_CONFIG_PATH = CONFIG_ROOT / "llm" / "models.yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelEntry(BaseModel):
    id: str
    label: str


class ReliabilityChain(BaseModel):
    primary: str
    fallbacks: list[str]


class ModelComparisonPolicy(BaseModel):
    pinned_models: list[str]


class RoutingPolicies(BaseModel):
    default_reliability_chain: ReliabilityChain
    model_comparison: ModelComparisonPolicy


class ModelRosterConfig(BaseModel):
    models: list[ModelEntry]
    routing_policies: RoutingPolicies

    def resolve(self, label: str) -> str:
        for entry in self.models:
            if entry.label == label:
                return entry.id
        raise ValueError(f"No model with label {label!r} in the roster")


def load_model_roster(path: Path = MODELS_CONFIG_PATH) -> ModelRosterConfig:
    return load_yaml_as(path, ModelRosterConfig)


class ModelNotAvailableError(Exception):
    def __init__(self, label: str, model_id: str, detail: str):
        self.label = label
        self.model_id = model_id
        super().__init__(f"Model {label!r} ({model_id}) is not available on OpenRouter: {detail}")


def verify_model_roster(roster: ModelRosterConfig, client: httpx.Client) -> None:
    """Preflight check: fail loudly and specifically if a configured model ID
    doesn't resolve against OpenRouter, rather than discovering it mid-run as
    an unexplained call failure or a silent fallback substitution."""
    response = client.get("/models")
    response.raise_for_status()
    available_ids = {entry["id"] for entry in response.json()["data"]}

    for entry in roster.models:
        if entry.id not in available_ids:
            raise ModelNotAvailableError(entry.label, entry.id, "not present in OpenRouter's /models response")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_llm_router.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/llm/llm_router.py tests/test_llm_router.py
git commit -m "feat: add httpx dependency, model roster loader, and OpenRouter preflight check"
```

---

## Task 9: Single-model call with technical retry and malformed-output repair

**Files:**
- Modify: `src/llm/llm_router.py` (append)
- Modify: `tests/test_llm_router.py` (append)

**Interfaces:**
- Consumes: `Decision` (Task 7), `httpx.Client` (already in use from Task 8).
- Produces: `@dataclass class RetryConfig` (`max_retries: int = 3`, `backoff_base_seconds: float = 0.5`, `sleep_fn: Callable[[float], None] = time.sleep`); `class AuthenticationError(Exception)`; `class ModelCallFailedError(Exception)` (attributes `model_id: str`, `reason: str`); `build_openrouter_client(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client`; `call_model(prompt: str, model_id: str, client: httpx.Client, retry_config: RetryConfig | None = None) -> Decision`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_router.py` (add these imports to the top of the file, alongside the existing `httpx`/`pytest` imports):

```python
import json

from src.llm.decision_schema import Decision, DecisionAction
from src.llm.llm_router import (
    AuthenticationError,
    ModelCallFailedError,
    ModelNotAvailableError,
    OPENROUTER_BASE_URL,
    RetryConfig,
    call_model,
    load_model_roster,
    verify_model_roster,
)
```

Append these tests to the end of `tests/test_llm_router.py`:

```python
def _decision_json(action: str = "OFFER") -> str:
    return json.dumps(
        {
            "action": action,
            "proposed_currency": "USDC",
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 100.0,
            "reasoning": "test",
        }
    )


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_call_model_succeeds_on_first_try():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert isinstance(decision, Decision)
    assert decision.proposed_currency == "USDC"


def test_call_model_retries_on_429_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert decision.action == DecisionAction.OFFER
    assert calls["count"] == 2


def test_call_model_retries_on_timeout_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.TimeoutException("timed out")
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert decision.action == DecisionAction.OFFER


def test_call_model_gives_up_after_persistent_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ModelCallFailedError):
        call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(max_retries=2, sleep_fn=lambda s: None))


def test_call_model_aborts_immediately_on_auth_failure_without_retrying():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "invalid api key"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationError):
        call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(max_retries=3, sleep_fn=lambda s: None))

    assert calls["count"] == 1


def test_call_model_repairs_malformed_json_on_first_attempt():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, json=_chat_response("not valid json"))
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    decision = call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(sleep_fn=lambda s: None))

    assert decision.action == DecisionAction.OFFER
    assert calls["count"] == 2


def test_call_model_gives_up_when_repair_also_fails_repeatedly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response("still not valid json"))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ModelCallFailedError):
        call_model("prompt", "anthropic/claude-sonnet-5", client, RetryConfig(max_retries=2, sleep_fn=lambda s: None))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_router.py -v`
Expected: FAIL with `ImportError: cannot import name 'call_model' from 'src.llm.llm_router'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/llm/llm_router.py` (add these imports at the top of the file, alongside the existing ones):

```python
import time as _time
from dataclasses import dataclass
from typing import Callable

from src.llm.decision_schema import Decision
```

Append to the bottom of `src/llm/llm_router.py`:

```python
_TECHNICAL_RETRY_STATUS_CODES = {429, 500, 502, 503}


@dataclass
class RetryConfig:
    """Plain dataclass, not a pydantic model: sleep_fn is a callable injected
    by tests (to skip real backoff delays), not serializable config data."""

    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    sleep_fn: Callable[[float], None] = _time.sleep


class AuthenticationError(Exception):
    pass


class ModelCallFailedError(Exception):
    def __init__(self, model_id: str, reason: str):
        self.model_id = model_id
        self.reason = reason
        super().__init__(f"Model {model_id} failed: {reason}")


def build_openrouter_client(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=OPENROUTER_BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        transport=transport,
        timeout=30.0,
    )


def _post_chat_completion(client: httpx.Client, model_id: str, messages: list[dict]) -> httpx.Response:
    return client.post(
        "/chat/completions",
        json={"model": model_id, "messages": messages, "response_format": {"type": "json_object"}},
    )


def _parse_decision(response: httpx.Response) -> Decision:
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return Decision.model_validate_json(content)


def call_model(
    prompt: str,
    model_id: str,
    client: httpx.Client,
    retry_config: RetryConfig | None = None,
) -> Decision:
    """Call one specific model, handling the first two failure tiers:
    technical failures (retried with exponential backoff) and malformed
    output (one repair reprompt per attempt). Raises AuthenticationError
    immediately on 401/403 -- a bad key won't fix itself by trying again --
    or ModelCallFailedError once retries and repair are exhausted.

    Economic validity (tier 3) is deliberately not handled here: only the
    caller (src/llm/agent_reasoning.py) knows the wallet/currency/chain
    constraints a Decision must satisfy.
    """
    retry_config = retry_config or RetryConfig()
    messages = [{"role": "user", "content": prompt}]
    last_error = "unknown error"

    for attempt in range(retry_config.max_retries):
        try:
            response = _post_chat_completion(client, model_id, messages)
        except httpx.TimeoutException as exc:
            last_error = f"timeout: {exc}"
            retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
            continue

        if response.status_code in (401, 403):
            raise AuthenticationError(
                f"OpenRouter rejected the API key for model {model_id}: HTTP {response.status_code}"
            )

        if response.status_code in _TECHNICAL_RETRY_STATUS_CODES:
            last_error = f"HTTP {response.status_code}"
            retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
            continue

        if response.status_code != 200:
            raise ModelCallFailedError(model_id, f"unexpected HTTP {response.status_code}")

        try:
            return _parse_decision(response)
        except (KeyError, IndexError, ValueError) as exc:
            repair_messages = messages + [
                {"role": "assistant", "content": response.text},
                {
                    "role": "user",
                    "content": (
                        f"Your last response was not valid JSON matching the required schema: {exc}. "
                        "Respond again with valid JSON only."
                    ),
                },
            ]
            try:
                repair_response = _post_chat_completion(client, model_id, repair_messages)
                return _parse_decision(repair_response)
            except (KeyError, IndexError, ValueError) as repair_exc:
                last_error = f"malformed output, repair failed: {repair_exc}"
                retry_config.sleep_fn(retry_config.backoff_base_seconds * (2**attempt))
                continue

    raise ModelCallFailedError(model_id, last_error)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_router.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm/llm_router.py tests/test_llm_router.py
git commit -m "feat: add single-model OpenRouter call with retry and repair-reprompt"
```

---

## Task 10: Fallback-chain traversal with requested/actual model tracking

**Files:**
- Modify: `src/llm/llm_router.py` (append)
- Modify: `tests/test_llm_router.py` (append)

**Interfaces:**
- Consumes: `call_model`, `RetryConfig`, `AuthenticationError`, `ModelCallFailedError`, `Decision` (Task 9).
- Produces: `class LLMCallResult(BaseModel)` (`requested_model: str`, `actual_model: str`, `fallback_used: bool`, `fallback_reason: str | None`, `model_attempts: list[str]`, `decision: Decision`); `class AllModelsFailedError(Exception)` (attributes `model_ids: list[str]`, `last_reason: str`); `call_with_fallback_chain(prompt: str, model_ids: list[str], client: httpx.Client, retry_config: RetryConfig | None = None) -> LLMCallResult`.

- [ ] **Step 1: Write the failing test**

Append to the top-of-file import block in `tests/test_llm_router.py`:

```python
from src.llm.llm_router import AllModelsFailedError, call_with_fallback_chain
```

Append these tests to the end of `tests/test_llm_router.py`:

```python
def test_fallback_chain_uses_primary_when_it_succeeds():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    result = call_with_fallback_chain("prompt", ["model-a", "model-b"], client, RetryConfig(sleep_fn=lambda s: None))

    assert result.requested_model == "model-a"
    assert result.actual_model == "model-a"
    assert result.fallback_used is False
    assert result.fallback_reason is None
    assert result.model_attempts == ["model-a"]


def test_fallback_chain_falls_through_when_primary_exhausts_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "model-a":
            return httpx.Response(500, json={"error": "down"})
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    result = call_with_fallback_chain(
        "prompt", ["model-a", "model-b"], client, RetryConfig(max_retries=1, sleep_fn=lambda s: None)
    )

    assert result.requested_model == "model-a"
    assert result.actual_model == "model-b"
    assert result.fallback_used is True
    assert result.fallback_reason == "HTTP 500"
    assert result.model_attempts == ["model-a", "model-b"]


def test_fallback_chain_raises_when_every_model_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(AllModelsFailedError):
        call_with_fallback_chain(
            "prompt", ["model-a", "model-b"], client, RetryConfig(max_retries=1, sleep_fn=lambda s: None)
        )


def test_fallback_chain_propagates_auth_error_without_trying_other_models():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationError):
        call_with_fallback_chain(
            "prompt", ["model-a", "model-b"], client, RetryConfig(max_retries=3, sleep_fn=lambda s: None)
        )

    assert calls["count"] == 1


def test_fallback_chain_rejects_empty_model_list():
    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with pytest.raises(ValueError):
        call_with_fallback_chain("prompt", [], client)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_router.py -v`
Expected: FAIL with `ImportError: cannot import name 'call_with_fallback_chain' from 'src.llm.llm_router'`

- [ ] **Step 3: Write minimal implementation**

Append to the bottom of `src/llm/llm_router.py`:

```python
class LLMCallResult(BaseModel):
    requested_model: str
    actual_model: str
    fallback_used: bool
    fallback_reason: str | None
    model_attempts: list[str]
    decision: Decision


class AllModelsFailedError(Exception):
    def __init__(self, model_ids: list[str], last_reason: str):
        self.model_ids = model_ids
        self.last_reason = last_reason
        super().__init__(f"All models in the chain failed: {model_ids}; last reason: {last_reason}")


def call_with_fallback_chain(
    prompt: str,
    model_ids: list[str],
    client: httpx.Client,
    retry_config: RetryConfig | None = None,
) -> LLMCallResult:
    """Try model_ids in order, stopping at the first success. model_ids[0] is
    the requested model; later entries are only tried once an earlier one
    exhausts its own retries/repair attempts inside call_model.

    Used by the default_reliability_chain routing policy. NOT used by the
    model_comparison policy, which calls call_model directly per pinned
    model with no substitution -- see Task 23's experiment_007, which must
    keep "model" a clean experimental factor rather than confounding it with
    reliability.
    """
    if not model_ids:
        raise ValueError("model_ids must contain at least one model")

    requested_model = model_ids[0]
    attempts: list[str] = []
    last_reason = "no models attempted"

    for index, model_id in enumerate(model_ids):
        attempts.append(model_id)
        try:
            decision = call_model(prompt, model_id, client, retry_config)
        except ModelCallFailedError as exc:
            last_reason = exc.reason
            continue

        return LLMCallResult(
            requested_model=requested_model,
            actual_model=model_id,
            fallback_used=index > 0,
            fallback_reason=last_reason if index > 0 else None,
            model_attempts=attempts,
            decision=decision,
        )

    raise AllModelsFailedError(model_ids, last_reason)
```

Note: `AuthenticationError` raised by `call_model` is not caught here, so it propagates immediately out of `call_with_fallback_chain` without trying later models in the chain — this is intentional per the three-tier design (a bad key won't be fixed by trying a different model).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_router.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm/llm_router.py tests/test_llm_router.py
git commit -m "feat: add fallback-chain traversal with requested/actual model tracking"
```

---

## Task 11: Decision adapter (economic validation → negotiation action)

**Files:**
- Create: `src/llm/decision_adapter.py`
- Test: `tests/test_decision_adapter.py`

**Interfaces:**
- Consumes: `Decision`, `DecisionAction`, `validate_decision` (Task 7).
- Produces: `class NegotiationAction(BaseModel)` (`action: DecisionAction`, `price: float`, `amount: float`, `currency_symbol: str`, `chain_name: str`, `reasoning: str`); `class DecisionValidationError(Exception)` (attribute `reason: str`); `adapt_decision(decision: Decision, supported_currencies: set[str], supported_chains: set[str], wallet_balances: dict[str, float]) -> NegotiationAction`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decision_adapter.py
import pytest

from src.llm.decision_adapter import DecisionValidationError, NegotiationAction, adapt_decision
from src.llm.decision_schema import Decision, DecisionAction


def _decision(**overrides) -> Decision:
    defaults = dict(
        action=DecisionAction.OFFER,
        proposed_currency="USDC",
        proposed_chain="ethereum",
        amount=1.0,
        price=100.0,
        reasoning="test reasoning",
    )
    defaults.update(overrides)
    return Decision(**defaults)


def test_adapt_valid_decision_produces_negotiation_action():
    action = adapt_decision(_decision(), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})

    assert isinstance(action, NegotiationAction)
    assert action.currency_symbol == "USDC"
    assert action.chain_name == "ethereum"
    assert action.price == 100.0


def test_adapt_invalid_currency_raises_with_reason():
    with pytest.raises(DecisionValidationError) as exc_info:
        adapt_decision(_decision(proposed_currency="NOTACOIN"), {"USDC"}, {"ethereum"}, {"USDC": 1000.0})

    assert "currency" in str(exc_info.value).lower()


def test_adapt_accept_with_insufficient_funds_raises():
    with pytest.raises(DecisionValidationError):
        adapt_decision(
            _decision(action=DecisionAction.ACCEPT, price=5000.0), {"USDC"}, {"ethereum"}, {"USDC": 100.0}
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_decision_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm.decision_adapter'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm/decision_adapter.py
"""Sits between raw LLM output and the negotiation engine.

Converts a schema-valid Decision into the negotiation engine's internal
action type after checking economic validity (currency/chain support,
positive amount/price, sufficient funds for ACCEPT) -- the "economically
invalid" tier of the three-tier failure handling described in
llm_router.py. Keeping this check here, not in llm_negotiation_engine.py,
means the negotiation state machine never has to know about the LLM-specific
Decision schema.
"""

from pydantic import BaseModel

from src.llm.decision_schema import Decision, DecisionAction, validate_decision


class NegotiationAction(BaseModel):
    action: DecisionAction
    price: float
    amount: float
    currency_symbol: str
    chain_name: str
    reasoning: str


class DecisionValidationError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def adapt_decision(
    decision: Decision,
    supported_currencies: set[str],
    supported_chains: set[str],
    wallet_balances: dict[str, float],
) -> NegotiationAction:
    result = validate_decision(decision, supported_currencies, supported_chains, wallet_balances)
    if not result.is_valid:
        raise DecisionValidationError(result.reason or "invalid decision")

    return NegotiationAction(
        action=decision.action,
        price=decision.price,
        amount=decision.amount,
        currency_symbol=decision.proposed_currency,
        chain_name=decision.proposed_chain,
        reasoning=decision.reasoning,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_decision_adapter.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm/decision_adapter.py tests/test_decision_adapter.py
git commit -m "feat: add decision adapter between LLM output and negotiation engine"
```

---

## Task 12: Agent-side utility context hook on BaseAgent

**Files:**
- Create: `src/llm/agent_reasoning.py` (first slice — grows in Tasks 13/15)
- Modify: `src/agents/base_agent.py`
- Modify: `src/agents/agent_factory.py:38-63` (`AgentProfileConfig`, `build_agent`)
- Test: `tests/test_agent_reasoning.py` (grows in Tasks 13/15)

**Interfaces:**
- Consumes: `MultiAttributeWeights` (existing, `src.utility.multi_attribute`).
- Produces: `class AgentUtilityContext(BaseModel)` (`agent_id: str`, `agent_class: str`, `risk_profile: str`, `utility_type: str`, `risk_aversion: float | None`, `eis: float | None`, `multi_attribute_weights: MultiAttributeWeights | None`, `wallet_balances: dict[str, float]`) in `src/llm/agent_reasoning.py`; `BaseAgent.utility_type: str`, `BaseAgent.risk_aversion: float | None = None`, `BaseAgent.eis: float | None = None`, `BaseAgent.multi_attribute_weights: MultiAttributeWeights | None = None`, and `BaseAgent.build_llm_context(self) -> AgentUtilityContext`.

Why these fields live on `BaseAgent` in addition to the already-built `utility_fn`: the LLM context must always present risk/utility parameters in a normalized numeric form regardless of which utility function the agent actually runs (per the design doc §5) — that requires the plain `risk_aversion`/`eis`/`weights` values, not just the opaque compiled `UtilityFunction` instance, which doesn't expose its constructor arguments.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_reasoning.py
from src.agents.agent_factory import build_agent, load_agent_profiles
from src.llm.agent_reasoning import AgentUtilityContext


def test_build_llm_context_surfaces_crra_agent_parameters():
    profiles = load_agent_profiles()
    agent = build_agent(profiles["consumer"])  # utility_type: crra, risk_aversion: 3.0

    context = agent.build_llm_context()

    assert isinstance(context, AgentUtilityContext)
    assert context.agent_id == agent.agent_id
    assert context.utility_type == "crra"
    assert context.risk_aversion == 3.0
    assert context.eis is None
    assert context.wallet_balances == agent.wallet.balances


def test_build_llm_context_surfaces_multi_attribute_agent_weights():
    profiles = load_agent_profiles()
    agent = build_agent(profiles["merchant"])  # utility_type: multi_attribute

    context = agent.build_llm_context()

    assert context.utility_type == "multi_attribute"
    assert context.multi_attribute_weights is not None
    assert context.multi_attribute_weights.liquidity_weight == 0.35
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm.agent_reasoning'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm/agent_reasoning.py
"""Assembles the context an LLM needs to make an economically meaningful
decision, and (in Task 15) drives the actual LLM call.

AgentUtilityContext is the agent-side slice only (identity, risk profile,
utility parameters, wallet) -- everything an agent knows about itself.
Environment-level context (currency governance, market intelligence, macro
state, opponent offers) is assembled separately in Task 13's
build_decision_context, which takes plain values rather than Environment/
BaseAgent objects, matching this codebase's existing layering convention
(e.g. src.blockchain.routing_engine.generate_candidates takes plain
balances, not a Wallet).
"""

from pydantic import BaseModel

from src.utility.multi_attribute import MultiAttributeWeights


class AgentUtilityContext(BaseModel):
    agent_id: str
    agent_class: str
    risk_profile: str
    utility_type: str
    risk_aversion: float | None = None
    eis: float | None = None
    multi_attribute_weights: MultiAttributeWeights | None = None
    wallet_balances: dict[str, float] = {}
```

Modify `src/agents/base_agent.py` (full new file content):

```python
# src/agents/base_agent.py
"""Parent class every agent inherits from."""

from pydantic import BaseModel, ConfigDict, Field

from src.agents.memory import AgentMemory
from src.agents.preferences import AgentPreferences
from src.agents.wallet import Wallet
from src.blockchain.routing_engine import CurrencyChainOption
from src.utility.base import UtilityFunction, choose_best
from src.utility.multi_attribute import MultiAttributeWeights


class BaseAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    agent_class: str
    profile_name: str
    risk_profile: str
    wallet: Wallet
    utility_fn: UtilityFunction
    utility_type: str
    risk_aversion: float | None = None
    eis: float | None = None
    multi_attribute_weights: MultiAttributeWeights | None = None
    memory: AgentMemory = Field(default_factory=AgentMemory)
    preferences: AgentPreferences = Field(default_factory=AgentPreferences)

    def choose_currency_and_chain(self, candidates: list[CurrencyChainOption]) -> CurrencyChainOption:
        wealth = sum(self.wallet.balances.values())
        return choose_best(candidates, self.utility_fn, wealth=wealth)

    def update_memory(self, symbol: str, success: bool) -> None:
        self.memory.record(symbol, success)
        self.preferences.update(symbol, 1.0 if success else 0.0)

    def build_llm_context(self) -> "AgentUtilityContext":
        from src.llm.agent_reasoning import AgentUtilityContext

        return AgentUtilityContext(
            agent_id=self.agent_id,
            agent_class=self.agent_class,
            risk_profile=self.risk_profile,
            utility_type=self.utility_type,
            risk_aversion=self.risk_aversion,
            eis=self.eis,
            multi_attribute_weights=self.multi_attribute_weights,
            wallet_balances=dict(self.wallet.balances),
        )
```

`build_llm_context` imports `AgentUtilityContext` lazily (inside the method, not at module top) because `src/llm/agent_reasoning.py` will, from Task 13 onward, import from `src/blockchain` and `src/currencies` modules that themselves sit below `src/agents` in this codebase's dependency order — a top-level `from src.llm.agent_reasoning import ...` in `base_agent.py` risks a future circular import once Task 13 adds those imports to `agent_reasoning.py`. The lazy import keeps `src/agents` -> `src/llm` a one-directional, late-bound dependency.

Modify `src/agents/agent_factory.py`'s `build_agent` function (the rest of the file is unchanged from Task 3):

```python
def build_agent(profile: AgentProfileConfig) -> BaseAgent:
    agent_cls = _AGENT_CLASSES[profile.agent_class]
    utility_fn = build_utility_function(profile.utility_type, profile.risk_aversion, profile.weights, profile.eis)
    wallet = Wallet(balances=dict(profile.initial_wallet))
    return agent_cls(
        agent_id=generate_id(profile.agent_class),
        agent_class=profile.agent_class,
        profile_name=profile.name,
        risk_profile=profile.risk_tolerance,
        wallet=wallet,
        utility_fn=utility_fn,
        utility_type=profile.utility_type,
        risk_aversion=profile.risk_aversion,
        eis=profile.eis,
        multi_attribute_weights=profile.weights,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_reasoning.py tests/test_agents.py tests/test_simulation.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/llm/agent_reasoning.py src/agents/base_agent.py src/agents/agent_factory.py tests/test_agent_reasoning.py
git commit -m "feat: add agent-side utility context for LLM prompts"
```

---

## Task 13: Full decision context assembly

**Files:**
- Modify: `src/llm/agent_reasoning.py` (append)
- Modify: `tests/test_agent_reasoning.py` (append)

**Interfaces:**
- Consumes: `AgentUtilityContext` (Task 12), `CurrencyChainOption` (existing, `src.blockchain.routing_engine`), `CurrencyProfile` (Task 5, `src.llm.market_intelligence`), `MacroState` (existing, `src.economy.macro_state`), `NegotiationAction` (Task 11, `src.llm.decision_adapter`).
- Produces: `class TransactionContext(BaseModel)` (`is_cross_border: bool`, `origin_currency: str | None = None`, `destination_currency: str | None = None`, `exchange_rate: float | None = None`, `exchange_rate_volatility: float | None = None`); `class AgentDecisionContext(BaseModel)` (`agent: AgentUtilityContext`, `candidates: list[CurrencyChainOption]`, `currency_profiles: dict[str, CurrencyProfile]`, `objective_macro_state: MacroState`, `perceived_macro_state: MacroState`, `transaction_context: TransactionContext`, `opponent_offer: NegotiationAction | None = None`, `conversation_history: list[str] = []`, `governance_prompt_enabled: bool = False`); `build_decision_context(agent_context: AgentUtilityContext, candidates: list[CurrencyChainOption], currency_profiles: dict[str, CurrencyProfile], objective_macro_state: MacroState, perceived_macro_state: MacroState, transaction_context: TransactionContext, opponent_offer: NegotiationAction | None = None, conversation_history: list[str] | None = None, governance_prompt_enabled: bool = False) -> AgentDecisionContext`.

- [ ] **Step 1: Write the failing test**

Append to the top of `tests/test_agent_reasoning.py`:

```python
from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.agent_reasoning import AgentDecisionContext, TransactionContext, build_decision_context
from src.llm.market_intelligence import load_currency_profile


def _option(**overrides) -> CurrencyChainOption:
    defaults = dict(
        currency_symbol="USDC",
        chain_name="ethereum",
        governance_score=0.95,
        liquidity_score=0.97,
        peg_error=0.0003,
        gas_fee=2.5,
        finality_seconds=12.0,
        genius_compliant=True,
    )
    defaults.update(overrides)
    return CurrencyChainOption(**defaults)
```

Append this test to the end of `tests/test_agent_reasoning.py`:

```python
def test_build_decision_context_filters_profiles_to_candidate_currencies_only():
    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC"), _option(currency_symbol="EURC", governance_score=0.90)]
    profiles = {"USDC": load_currency_profile("USDC"), "USDT": load_currency_profile("USDT")}
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)

    context = build_decision_context(agent_context, candidates, profiles, macro, macro, txn_context)

    assert isinstance(context, AgentDecisionContext)
    # USDT was in the profile corpus passed in, but no candidate proposes USDT --
    # it must not leak into the context (keeps the prompt focused, per the
    # hypothesis -> context-field traceability rule in the design doc).
    assert set(context.currency_profiles.keys()) == {"USDC"}
    assert context.transaction_context.is_cross_border is False
    assert context.conversation_history == []
    assert context.governance_prompt_enabled is False
    assert context.opponent_offer is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: FAIL with `ImportError: cannot import name 'AgentDecisionContext' from 'src.llm.agent_reasoning'`

- [ ] **Step 3: Write minimal implementation**

Append to the top imports of `src/llm/agent_reasoning.py`:

```python
from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.decision_adapter import NegotiationAction
from src.llm.market_intelligence import CurrencyProfile
```

Append to the end of `src/llm/agent_reasoning.py`:

```python
class TransactionContext(BaseModel):
    is_cross_border: bool
    origin_currency: str | None = None
    destination_currency: str | None = None
    exchange_rate: float | None = None
    exchange_rate_volatility: float | None = None


class AgentDecisionContext(BaseModel):
    agent: AgentUtilityContext
    candidates: list[CurrencyChainOption]
    currency_profiles: dict[str, CurrencyProfile] = {}
    objective_macro_state: MacroState
    perceived_macro_state: MacroState
    transaction_context: TransactionContext
    opponent_offer: NegotiationAction | None = None
    conversation_history: list[str] = []
    governance_prompt_enabled: bool = False


def build_decision_context(
    agent_context: AgentUtilityContext,
    candidates: list[CurrencyChainOption],
    currency_profiles: dict[str, CurrencyProfile],
    objective_macro_state: MacroState,
    perceived_macro_state: MacroState,
    transaction_context: TransactionContext,
    opponent_offer: NegotiationAction | None = None,
    conversation_history: list[str] | None = None,
    governance_prompt_enabled: bool = False,
) -> AgentDecisionContext:
    candidate_symbols = {candidate.currency_symbol for candidate in candidates}
    relevant_profiles = {
        symbol: profile for symbol, profile in currency_profiles.items() if symbol in candidate_symbols
    }
    return AgentDecisionContext(
        agent=agent_context,
        candidates=candidates,
        currency_profiles=relevant_profiles,
        objective_macro_state=objective_macro_state,
        perceived_macro_state=perceived_macro_state,
        transaction_context=transaction_context,
        opponent_offer=opponent_offer,
        conversation_history=conversation_history or [],
        governance_prompt_enabled=governance_prompt_enabled,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm/agent_reasoning.py tests/test_agent_reasoning.py
git commit -m "feat: assemble full agent decision context for LLM prompts"
```

---

## Task 14: Prompt templates and rendering

**Files:**
- Modify: `src/llm/prompts/buyer_prompt.txt`
- Modify: `src/llm/prompts/seller_prompt.txt`
- Modify: `src/llm/prompts/investor_prompt.txt`
- Modify: `src/llm/prompts/bank_prompt.txt`
- Modify: `src/llm/agent_reasoning.py` (append)
- Modify: `tests/test_agent_reasoning.py` (append)

**Interfaces:**
- Consumes: `AgentDecisionContext`, `AgentUtilityContext`, `TransactionContext` (Tasks 12-13).
- Produces: `PROMPT_VERSIONS: dict[str, str]`; `prompt_version_for(agent_class: str) -> str`; `render_prompt(agent_class: str, context: AgentDecisionContext, schema_json: str) -> str`; `hash_rendered_prompt(text: str) -> str`.

- [ ] **Step 1: Write the four prompt templates**

```text
# src/llm/prompts/buyer_prompt.txt
You are an AI buyer agent negotiating a purchase in a simulated digital-currency economy. Decide how to pay: which currency, which blockchain, and what price to offer or accept, given your risk preferences and everything you know about the currencies available.

# Your risk and utility profile
{utility_context_block}

# Candidate currencies and chains
{candidates_block}

# Background information on these currencies -- historical, not current market state
{currency_profiles_block}

# Macro-economic conditions
{macro_block}

# Transaction context
{transaction_block}
{governance_block}
# Negotiation so far
{conversation_block}

# Your task
Respond with a single JSON object matching this schema exactly -- no prose before or after the JSON:
{schema_block}

Valid values for "action" are OFFER, COUNTER_OFFER, ACCEPT, REJECT, WALK_AWAY. "proposed_currency" and "proposed_chain" must be one of the candidates listed above.
```

```text
# src/llm/prompts/seller_prompt.txt
You are an AI seller agent in a simulated digital-currency economy. Decide whether to accept, counter, or reject a buyer's offer, and in which currency and blockchain you're willing to settle, given your risk preferences.

# Your risk and utility profile
{utility_context_block}

# Candidate currencies and chains
{candidates_block}

# Background information on these currencies -- historical, not current market state
{currency_profiles_block}

# Macro-economic conditions
{macro_block}

# Transaction context
{transaction_block}
{governance_block}
# Negotiation so far
{conversation_block}

# Your task
Respond with a single JSON object matching this schema exactly -- no prose before or after the JSON:
{schema_block}

Valid values for "action" are OFFER, COUNTER_OFFER, ACCEPT, REJECT, WALK_AWAY. "proposed_currency" and "proposed_chain" must be one of the candidates listed above.
```

```text
# src/llm/prompts/investor_prompt.txt
You are an AI investor agent in a simulated digital-currency economy, allocating funds across currencies with a longer holding horizon than a buyer or seller. Decide which currency and blockchain best serve your risk preferences for this allocation decision.

# Your risk and utility profile
{utility_context_block}

# Candidate currencies and chains
{candidates_block}

# Background information on these currencies -- historical, not current market state
{currency_profiles_block}

# Macro-economic conditions
{macro_block}

# Transaction context
{transaction_block}
{governance_block}
# Negotiation so far
{conversation_block}

# Your task
Respond with a single JSON object matching this schema exactly -- no prose before or after the JSON:
{schema_block}

Valid values for "action" are OFFER, COUNTER_OFFER, ACCEPT, REJECT, WALK_AWAY. "proposed_currency" and "proposed_chain" must be one of the candidates listed above.
```

```text
# src/llm/prompts/bank_prompt.txt
You are an AI bank agent in a simulated digital-currency economy, managing a reserve composition and deciding which settlement currency and blockchain to use for a counterparty transaction, given your risk preferences.

# Your risk and utility profile
{utility_context_block}

# Candidate currencies and chains
{candidates_block}

# Background information on these currencies -- historical, not current market state
{currency_profiles_block}

# Macro-economic conditions
{macro_block}

# Transaction context
{transaction_block}
{governance_block}
# Negotiation so far
{conversation_block}

# Your task
Respond with a single JSON object matching this schema exactly -- no prose before or after the JSON:
{schema_block}

Valid values for "action" are OFFER, COUNTER_OFFER, ACCEPT, REJECT, WALK_AWAY. "proposed_currency" and "proposed_chain" must be one of the candidates listed above.
```

- [ ] **Step 2: Write the failing test**

Append to the top imports of `tests/test_agent_reasoning.py`:

```python
from src.llm.agent_reasoning import prompt_version_for, render_prompt
```

Append these tests to the end of `tests/test_agent_reasoning.py`:

```python
def test_render_prompt_includes_all_context_sections_and_respects_governance_flag():
    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    profiles = {"USDC": load_currency_profile("USDC")}
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)

    baseline_context = build_decision_context(
        agent_context, candidates, profiles, macro, macro, txn_context, governance_prompt_enabled=False
    )
    governance_context = build_decision_context(
        agent_context, candidates, profiles, macro, macro, txn_context, governance_prompt_enabled=True
    )

    baseline_prompt = render_prompt("buyer", baseline_context, "{}")
    governance_prompt = render_prompt("buyer", governance_context, "{}")

    assert "USDC" in baseline_prompt
    assert "Risk aversion" in baseline_prompt
    assert "Governance emphasis" not in baseline_prompt
    assert "Governance emphasis" in governance_prompt


def test_render_prompt_works_for_all_four_agent_classes():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="seller",
        risk_profile="medium",
        utility_type="multi_attribute",
        wallet_balances={"USDC": 500.0},
    )
    candidates = [_option()]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    context = build_decision_context(agent_context, candidates, {}, macro, macro, txn_context)

    for agent_class in ["buyer", "seller", "investor", "bank"]:
        prompt = render_prompt(agent_class, context, "{}")
        assert "USDC" in prompt


def test_prompt_version_for_returns_stable_identifier():
    assert prompt_version_for("buyer") == "buyer_prompt@v1"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_prompt' from 'src.llm.agent_reasoning'`

- [ ] **Step 4: Write minimal implementation**

Append to the top imports of `src/llm/agent_reasoning.py`:

```python
import hashlib
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

PROMPT_VERSIONS: dict[str, str] = {
    "buyer": "buyer_prompt@v1",
    "seller": "seller_prompt@v1",
    "investor": "investor_prompt@v1",
    "bank": "bank_prompt@v1",
}

_GOVERNANCE_EMPHASIS_BLOCK = (
    "# Governance emphasis\n"
    "Pay particular attention to each currency's governance quality: reserve "
    "composition, transparency, issuer risk, and GENIUS Act compliance status "
    "above. These factors should weigh heavily in your decision.\n"
)
```

Append to the end of `src/llm/agent_reasoning.py`:

```python
def prompt_version_for(agent_class: str) -> str:
    return PROMPT_VERSIONS[agent_class]


def hash_rendered_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _format_utility_context_block(agent: AgentUtilityContext) -> str:
    parts = [f"Risk profile: {agent.risk_profile}", f"Utility type: {agent.utility_type}"]
    if agent.risk_aversion is not None:
        parts.append(f"Risk aversion (CRRA/CARA-style gamma): {agent.risk_aversion}")
    if agent.eis is not None:
        parts.append(f"EIS-inspired fee-sensitivity parameter: {agent.eis}")
    if agent.multi_attribute_weights is not None:
        w = agent.multi_attribute_weights
        parts.append(
            f"Multi-attribute weights: governance={w.governance_weight}, liquidity={w.liquidity_weight}, "
            f"gas_fee={w.gas_fee_weight}, volatility={w.volatility_weight}, compliance={w.compliance_weight}"
        )
    return "\n".join(parts)


def _format_candidates_block(candidates: list[CurrencyChainOption]) -> str:
    if not candidates:
        return "(no candidates available)"
    lines = [
        f"- {option.currency_symbol} on {option.chain_name}: governance_score={option.governance_score}, "
        f"liquidity_score={option.liquidity_score}, peg_error={option.peg_error}, gas_fee={option.gas_fee}, "
        f"finality_seconds={option.finality_seconds}, genius_compliant={option.genius_compliant}"
        for option in candidates
    ]
    return "\n".join(lines)


def _format_currency_profiles_block(profiles: dict[str, CurrencyProfile]) -> str:
    if not profiles:
        return "(no background information available for these currencies)"
    sections = [
        f"## {symbol}\n{profile.executive_summary}\nGovernance: {profile.governance}\n"
        f"Reserves/transparency: {profile.reserves_and_transparency}"
        for symbol, profile in profiles.items()
    ]
    return "\n\n".join(sections)


def _format_macro_block(objective: MacroState, perceived: MacroState) -> str:
    return (
        f"Objective state: inflation={objective.inflation}, interest_rate={objective.interest_rate}, "
        f"gold_price={objective.gold_price}, confidence_index={objective.confidence_index}\n"
        f"Your perceived state (may differ from objective): inflation={perceived.inflation}, "
        f"interest_rate={perceived.interest_rate}, gold_price={perceived.gold_price}, "
        f"confidence_index={perceived.confidence_index}"
    )


def _format_transaction_block(txn: TransactionContext) -> str:
    if not txn.is_cross_border:
        return "Domestic transaction."
    return (
        f"Cross-border transaction: {txn.origin_currency} -> {txn.destination_currency}, "
        f"exchange_rate={txn.exchange_rate}, exchange_rate_volatility={txn.exchange_rate_volatility}"
    )


def _format_conversation_block(history: list[str]) -> str:
    return "\n".join(history) if history else "(negotiation has not started yet)"


def render_prompt(agent_class: str, context: AgentDecisionContext, schema_json: str) -> str:
    template = (PROMPTS_DIR / f"{agent_class}_prompt.txt").read_text(encoding="utf-8")
    fields = {
        "utility_context_block": _format_utility_context_block(context.agent),
        "candidates_block": _format_candidates_block(context.candidates),
        "currency_profiles_block": _format_currency_profiles_block(context.currency_profiles),
        "macro_block": _format_macro_block(context.objective_macro_state, context.perceived_macro_state),
        "transaction_block": _format_transaction_block(context.transaction_context),
        "governance_block": _GOVERNANCE_EMPHASIS_BLOCK if context.governance_prompt_enabled else "",
        "conversation_block": _format_conversation_block(context.conversation_history),
        "schema_block": schema_json,
    }
    return template.format(**fields)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/llm/prompts/*.txt src/llm/agent_reasoning.py tests/test_agent_reasoning.py
git commit -m "feat: fill in per-agent-class prompt templates and rendering"
```

---

## Task 15: End-to-end decide() with bounded economic-correction loop

**Files:**
- Modify: `src/llm/agent_reasoning.py` (append)
- Modify: `tests/test_agent_reasoning.py` (append)

**Interfaces:**
- Consumes: `render_prompt`, `AgentDecisionContext` (Task 14); `ModelRosterConfig`, `RetryConfig`, `LLMCallResult`, `call_with_fallback_chain`, `call_model`, `AllModelsFailedError`, `AuthenticationError`, `ModelCallFailedError` (Tasks 8-10); `Decision` (Task 7); `NegotiationAction`, `DecisionValidationError`, `adapt_decision` (Task 11).
- Produces: `class LLMDecisionOutcome(BaseModel)` (`call_result: LLMCallResult | None`, `negotiation_action: NegotiationAction | None`, `used_deterministic_fallback: bool`, `correction_attempts: int`); `decide(agent_class: str, context: AgentDecisionContext, roster: ModelRosterConfig, client: httpx.Client, supported_currencies: set[str], supported_chains: set[str], policy_name: str = "default_reliability_chain", retry_config: RetryConfig | None = None, max_correction_attempts: int = 2, deterministic_fallback: Callable[[], NegotiationAction] | None = None) -> LLMDecisionOutcome`.

This is the tier-3 ("economically invalid") handling from the design doc §4: on an invalid-but-well-formed `Decision`, re-prompt the *same* `actual_model` (not the fallback chain) with the validation error, up to `max_correction_attempts`, then fall back to a caller-supplied deterministic function (e.g. the existing rule-based `choose_best`/`negotiate`).

- [ ] **Step 1: Write the failing test**

Append to the top imports of `tests/test_agent_reasoning.py`:

```python
import json as _json

import httpx

from src.llm.agent_reasoning import LLMDecisionOutcome, decide
from src.llm.decision_adapter import NegotiationAction
from src.llm.decision_schema import DecisionAction
from src.llm.llm_router import OPENROUTER_BASE_URL, RetryConfig, load_model_roster
```

Append these helpers and tests to the end of `tests/test_agent_reasoning.py`:

```python
def _decision_json(action: str = "OFFER", currency: str = "USDC", price: float = 100.0) -> str:
    return _json.dumps(
        {
            "action": action,
            "proposed_currency": currency,
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": price,
            "reasoning": "test",
        }
    )


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _base_decision_context() -> AgentDecisionContext:
    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    return build_decision_context(agent_context, candidates, {}, macro, macro, txn_context)


def test_decide_returns_valid_negotiation_action_on_first_try():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(_decision_json()))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()

    outcome = decide(
        "buyer",
        _base_decision_context(),
        roster,
        client,
        {"USDC"},
        {"ethereum"},
        retry_config=RetryConfig(sleep_fn=lambda s: None),
    )

    assert isinstance(outcome, LLMDecisionOutcome)
    assert outcome.used_deterministic_fallback is False
    assert outcome.negotiation_action.currency_symbol == "USDC"
    assert outcome.correction_attempts == 0
    assert outcome.call_result.actual_model == "anthropic/claude-sonnet-5"


def test_decide_corrects_economically_invalid_decision_with_the_same_model():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, json=_chat_response(_decision_json(currency="NOTACOIN")))
        return httpx.Response(200, json=_chat_response(_decision_json(currency="USDC")))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()

    outcome = decide(
        "buyer",
        _base_decision_context(),
        roster,
        client,
        {"USDC"},
        {"ethereum"},
        retry_config=RetryConfig(sleep_fn=lambda s: None),
    )

    assert outcome.used_deterministic_fallback is False
    assert outcome.correction_attempts == 1
    assert outcome.negotiation_action.currency_symbol == "USDC"


def test_decide_falls_back_deterministically_after_exhausting_correction_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response(_decision_json(currency="NOTACOIN")))

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()
    fallback_action = NegotiationAction(
        action=DecisionAction.WALK_AWAY,
        price=0.0,
        amount=0.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        reasoning="deterministic fallback",
    )

    outcome = decide(
        "buyer",
        _base_decision_context(),
        roster,
        client,
        {"USDC"},
        {"ethereum"},
        retry_config=RetryConfig(sleep_fn=lambda s: None),
        max_correction_attempts=2,
        deterministic_fallback=lambda: fallback_action,
    )

    assert outcome.used_deterministic_fallback is True
    assert outcome.correction_attempts == 2
    assert outcome.negotiation_action is fallback_action


def test_decide_falls_back_when_every_model_in_the_chain_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()
    fallback_action = NegotiationAction(
        action=DecisionAction.WALK_AWAY,
        price=0.0,
        amount=0.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        reasoning="deterministic fallback",
    )

    outcome = decide(
        "buyer",
        _base_decision_context(),
        roster,
        client,
        {"USDC"},
        {"ethereum"},
        retry_config=RetryConfig(max_retries=1, sleep_fn=lambda s: None),
        deterministic_fallback=lambda: fallback_action,
    )

    assert outcome.used_deterministic_fallback is True
    assert outcome.call_result is None
    assert outcome.negotiation_action is fallback_action
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: FAIL with `ImportError: cannot import name 'decide' from 'src.llm.agent_reasoning'`

- [ ] **Step 3: Write minimal implementation**

Append to the top imports of `src/llm/agent_reasoning.py`:

```python
import json
from typing import Callable

import httpx

from src.llm.decision_adapter import DecisionValidationError, adapt_decision
from src.llm.decision_schema import Decision
from src.llm.llm_router import (
    AllModelsFailedError,
    AuthenticationError,
    LLMCallResult,
    ModelCallFailedError,
    ModelRosterConfig,
    RetryConfig,
    call_model,
    call_with_fallback_chain,
)
```

Append to the end of `src/llm/agent_reasoning.py`:

```python
class LLMDecisionOutcome(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    call_result: LLMCallResult | None
    negotiation_action: NegotiationAction | None
    used_deterministic_fallback: bool
    correction_attempts: int


def _model_ids_for_policy(roster: ModelRosterConfig, policy_name: str) -> list[str]:
    if policy_name == "default_reliability_chain":
        chain = roster.routing_policies.default_reliability_chain
        return [roster.resolve(chain.primary)] + [roster.resolve(label) for label in chain.fallbacks]
    if policy_name == "model_comparison":
        return [roster.resolve(label) for label in roster.routing_policies.model_comparison.pinned_models]
    raise ValueError(f"Unknown routing policy: {policy_name}")


def _fall_back(
    deterministic_fallback: Callable[[], NegotiationAction] | None,
    call_result: LLMCallResult | None,
    correction_attempts: int,
) -> LLMDecisionOutcome:
    action = deterministic_fallback() if deterministic_fallback is not None else None
    return LLMDecisionOutcome(
        call_result=call_result,
        negotiation_action=action,
        used_deterministic_fallback=True,
        correction_attempts=correction_attempts,
    )


def decide(
    agent_class: str,
    context: AgentDecisionContext,
    roster: ModelRosterConfig,
    client: httpx.Client,
    supported_currencies: set[str],
    supported_chains: set[str],
    policy_name: str = "default_reliability_chain",
    retry_config: RetryConfig | None = None,
    max_correction_attempts: int = 2,
    deterministic_fallback: Callable[[], NegotiationAction] | None = None,
) -> LLMDecisionOutcome:
    model_ids = _model_ids_for_policy(roster, policy_name)
    schema_json = json.dumps(Decision.model_json_schema())
    prompt = render_prompt(agent_class, context, schema_json)

    try:
        call_result = call_with_fallback_chain(prompt, model_ids, client, retry_config)
    except (AllModelsFailedError, AuthenticationError):
        return _fall_back(deterministic_fallback, call_result=None, correction_attempts=0)

    correction_attempts = 0
    current_call_result = call_result
    current_prompt = prompt

    while True:
        validation_error: DecisionValidationError | None = None
        try:
            action = adapt_decision(
                current_call_result.decision, supported_currencies, supported_chains, context.agent.wallet_balances
            )
        except DecisionValidationError as exc:
            validation_error = exc

        if validation_error is None:
            return LLMDecisionOutcome(
                call_result=current_call_result,
                negotiation_action=action,
                used_deterministic_fallback=False,
                correction_attempts=correction_attempts,
            )

        if correction_attempts >= max_correction_attempts:
            return _fall_back(deterministic_fallback, current_call_result, correction_attempts)

        correction_attempts += 1
        current_prompt = (
            f"{current_prompt}\n\nYour previous proposal was economically invalid: {validation_error.reason}. "
            "Respond again with a corrected JSON decision matching the schema."
        )
        try:
            corrected_decision = call_model(current_prompt, current_call_result.actual_model, client, retry_config)
        except ModelCallFailedError:
            return _fall_back(deterministic_fallback, current_call_result, correction_attempts)

        current_call_result = LLMCallResult(
            requested_model=current_call_result.requested_model,
            actual_model=current_call_result.actual_model,
            fallback_used=current_call_result.fallback_used,
            fallback_reason=current_call_result.fallback_reason,
            model_attempts=current_call_result.model_attempts,
            decision=corrected_decision,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: 10 passed

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -v`
Expected: all previously-passing tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/llm/agent_reasoning.py tests/test_agent_reasoning.py
git commit -m "feat: wire end-to-end decide() with bounded economic-correction loop"
```

---

## Task 16: Live Polygon price snapshot

**Files:**
- Modify: `src/llm/market_intelligence.py` (append)
- Modify: `tests/test_market_intelligence.py` (append)

**Interfaces:**
- Produces: `POLYGON_BASE_URL: str`; `build_polygon_client(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client`; `class LivePriceSnapshot(BaseModel)` (`ticker: str`, `price: float | None`, `retrieval_timestamp: datetime`, `source: str = "polygon"`, `data_window: str | None = None`, `unavailable_reason: str | None = None`); `fetch_live_price(ticker: str, client: httpx.Client) -> LivePriceSnapshot`.

- [ ] **Step 1: Write the failing test**

Append to the top imports of `tests/test_market_intelligence.py`:

```python
import httpx

from src.llm.market_intelligence import POLYGON_BASE_URL, fetch_live_price
```

Append these tests to the end of `tests/test_market_intelligence.py`:

```python
def test_fetch_live_price_returns_price_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"c": 1.0002}]})

    client = httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))

    snapshot = fetch_live_price("X:USDCUSD", client)

    assert snapshot.price == 1.0002
    assert snapshot.unavailable_reason is None
    assert snapshot.ticker == "X:USDCUSD"


def test_fetch_live_price_degrades_gracefully_when_polygon_has_no_data():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))

    snapshot = fetch_live_price("X:NOTATICKER", client)

    assert snapshot.price is None
    assert snapshot.unavailable_reason is not None


def test_fetch_live_price_degrades_gracefully_on_http_error_rather_than_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    client = httpx.Client(base_url=POLYGON_BASE_URL, transport=httpx.MockTransport(handler))

    snapshot = fetch_live_price("X:USDCUSD", client)

    assert snapshot.price is None
    assert snapshot.unavailable_reason is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_market_intelligence.py -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_live_price' from 'src.llm.market_intelligence'`

- [ ] **Step 3: Write minimal implementation**

Append to the top imports of `src/llm/market_intelligence.py`:

```python
from datetime import datetime, timezone

import httpx

POLYGON_BASE_URL = "https://api.polygon.io"
```

Append to the end of `src/llm/market_intelligence.py`:

```python
def build_polygon_client(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(base_url=POLYGON_BASE_URL, params={"apiKey": api_key}, transport=transport, timeout=15.0)


class LivePriceSnapshot(BaseModel):
    ticker: str
    price: float | None
    retrieval_timestamp: datetime
    source: str = "polygon"
    data_window: str | None = None
    unavailable_reason: str | None = None


def fetch_live_price(ticker: str, client: httpx.Client) -> LivePriceSnapshot:
    """Fetch a live crypto aggregate price for `ticker` (e.g. "X:USDCUSD").

    Never raises on a data or network problem -- a market-data outage must
    not crash a negotiation, so failures come back as a snapshot with
    price=None and unavailable_reason set, not an exception.
    """
    now = datetime.now(timezone.utc)
    try:
        response = client.get(f"/v2/aggs/ticker/{ticker}/prev")
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return LivePriceSnapshot(ticker=ticker, price=None, retrieval_timestamp=now, unavailable_reason="no data returned for this ticker")
        return LivePriceSnapshot(
            ticker=ticker, price=results[0]["c"], retrieval_timestamp=now, data_window="previous close"
        )
    except httpx.HTTPError as exc:
        return LivePriceSnapshot(ticker=ticker, price=None, retrieval_timestamp=now, unavailable_reason=str(exc))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_market_intelligence.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm/market_intelligence.py tests/test_market_intelligence.py
git commit -m "feat: fetch optional live price snapshots from Polygon with graceful degradation"
```

---

## Task 17: LLMOffer immutable record

**Files:**
- Create: `src/negotiation/llm_offer.py`
- Test: `tests/test_llm_offer.py`

**Interfaces:**
- Consumes: `DecisionAction` (Task 7), `generate_id` (existing, `src.utils.helpers`).
- Produces: `class LLMOffer(BaseModel)` (`offer_id: str`, `negotiation_id: str`, `previous_offer_id: str | None = None`, `agent_id: str`, `action: DecisionAction`, `price: float`, `currency_symbol: str`, `chain_name: str`, `reasoning: str`, `round: int`, `timestamp: datetime`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_offer.py
from src.llm.decision_schema import DecisionAction
from src.negotiation.llm_offer import LLMOffer


def _offer(**overrides) -> LLMOffer:
    defaults = dict(
        negotiation_id="neg-1",
        agent_id="buyer-1",
        action=DecisionAction.OFFER,
        price=100.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        reasoning="test",
        round=0,
    )
    defaults.update(overrides)
    return LLMOffer(**defaults)


def test_llm_offer_generates_unique_id_and_has_no_previous_by_default():
    offer = _offer()

    assert offer.offer_id.startswith("offer-")
    assert offer.previous_offer_id is None
    assert offer.timestamp is not None


def test_counter_offer_references_previous_offer_id():
    first = _offer(price=90.0)
    counter = _offer(
        previous_offer_id=first.offer_id, agent_id="seller-1", action=DecisionAction.COUNTER_OFFER, price=95.0, round=1
    )

    assert counter.previous_offer_id == first.offer_id
    assert counter.offer_id != first.offer_id


def test_two_offers_never_share_an_id():
    assert _offer().offer_id != _offer().offer_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_offer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.negotiation.llm_offer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/negotiation/llm_offer.py
"""Immutable offer record for the LLM-driven negotiation engine.

Distinct from src.negotiation.offer.Offer (the rule-based engine's offer
type) so nothing here can break Phase 1's tested rule-based negotiation path
-- see llm_negotiation_engine.py. Counter-offers must create new LLMOffer
instances referencing previous_offer_id rather than mutating an existing one.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.llm.decision_schema import DecisionAction
from src.utils.helpers import generate_id


class LLMOffer(BaseModel):
    offer_id: str = Field(default_factory=lambda: generate_id("offer"))
    negotiation_id: str
    previous_offer_id: str | None = None
    agent_id: str
    action: DecisionAction
    price: float
    currency_symbol: str
    chain_name: str
    reasoning: str
    round: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_offer.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/negotiation/llm_offer.py tests/test_llm_offer.py
git commit -m "feat: add immutable LLMOffer record for the LLM negotiation engine"
```

---

## Task 18: LLM-driven negotiation state machine

**Files:**
- Create: `src/negotiation/llm_negotiation_engine.py`
- Test: `tests/test_llm_negotiation_engine.py`

**Interfaces:**
- Consumes: `NegotiationAction` (Task 11), `DecisionAction` (Task 7), `LLMOffer` (Task 17).
- Produces: `class NegotiationStatus(str, Enum)` (`IN_PROGRESS`, `ACCEPTED`, `REJECTED`, `WALKED_AWAY`, `MAX_ROUNDS_REACHED`); `class NegotiationSession(BaseModel)` (`negotiation_id: str`, `buyer_id: str`, `seller_id: str`, `current_round: int`, `max_rounds: int`, `status: NegotiationStatus`, `initial_offer: LLMOffer | None`, `current_offer: LLMOffer | None`, `current_currency: str | None`, `current_blockchain: str | None`, `conversation_history: list[LLMOffer]`, `created_at: datetime`, `completed_at: datetime | None`, methods `record_offer(self, agent_id: str, action: NegotiationAction) -> LLMOffer` and `finalize(self, status: NegotiationStatus) -> None`); `run_llm_negotiation(buyer_id: str, seller_id: str, buyer_decide: Callable[[NegotiationSession], NegotiationAction], seller_decide: Callable[[NegotiationSession], NegotiationAction], max_rounds: int = 10) -> NegotiationSession`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_negotiation_engine.py
from src.llm.decision_adapter import NegotiationAction
from src.llm.decision_schema import DecisionAction
from src.negotiation.llm_negotiation_engine import NegotiationSession, NegotiationStatus, run_llm_negotiation


def _action(action: DecisionAction, price: float = 100.0) -> NegotiationAction:
    return NegotiationAction(
        action=action, price=price, amount=1.0, currency_symbol="USDC", chain_name="ethereum", reasoning="test"
    )


def test_negotiation_accepts_when_seller_accepts_buyers_offer():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.OFFER, price=95.0)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.ACCEPT, price=session.current_offer.price)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)

    assert session.status == NegotiationStatus.ACCEPTED
    assert session.completed_at is not None
    assert len(session.conversation_history) == 2
    assert session.conversation_history[0].agent_id == "buyer-1"
    assert session.conversation_history[1].agent_id == "seller-1"


def test_negotiation_terminates_on_reject():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.OFFER)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.REJECT)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)

    assert session.status == NegotiationStatus.REJECTED


def test_negotiation_terminates_on_walk_away():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.WALK_AWAY)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.OFFER)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)

    assert session.status == NegotiationStatus.WALKED_AWAY
    assert len(session.conversation_history) == 1


def test_negotiation_hits_max_rounds_cap_and_never_loops_forever():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.COUNTER_OFFER)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.COUNTER_OFFER)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=4)

    assert session.status == NegotiationStatus.MAX_ROUNDS_REACHED
    assert len(session.conversation_history) == 4


def test_offers_form_a_previous_offer_id_chain():
    def buyer_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.OFFER)

    def seller_decide(session: NegotiationSession) -> NegotiationAction:
        return _action(DecisionAction.ACCEPT, price=session.current_offer.price)

    session = run_llm_negotiation("buyer-1", "seller-1", buyer_decide, seller_decide, max_rounds=10)

    assert session.conversation_history[0].previous_offer_id is None
    assert session.conversation_history[1].previous_offer_id == session.conversation_history[0].offer_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_negotiation_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.negotiation.llm_negotiation_engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/negotiation/llm_negotiation_engine.py
"""Multi-round LLM-driven negotiation state machine.

Additive alongside src.negotiation.negotiation_engine's rule-based
negotiate() -- that function and its tests are untouched; this is a
separate path used only when a caller opts into it (see
experiments/experiment_007_governance_prompting.py). A hard max_rounds cap
guarantees termination, the same guarantee the rule-based engine already
provides.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from src.llm.decision_adapter import NegotiationAction
from src.llm.decision_schema import DecisionAction
from src.negotiation.llm_offer import LLMOffer
from src.utils.helpers import generate_id


class NegotiationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WALKED_AWAY = "walked_away"
    MAX_ROUNDS_REACHED = "max_rounds_reached"


class NegotiationSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    negotiation_id: str = Field(default_factory=lambda: generate_id("neg"))
    buyer_id: str
    seller_id: str
    current_round: int = 0
    max_rounds: int
    status: NegotiationStatus = NegotiationStatus.IN_PROGRESS
    initial_offer: LLMOffer | None = None
    current_offer: LLMOffer | None = None
    current_currency: str | None = None
    current_blockchain: str | None = None
    conversation_history: list[LLMOffer] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def record_offer(self, agent_id: str, action: NegotiationAction) -> LLMOffer:
        offer = LLMOffer(
            negotiation_id=self.negotiation_id,
            previous_offer_id=self.current_offer.offer_id if self.current_offer else None,
            agent_id=agent_id,
            action=action.action,
            price=action.price,
            currency_symbol=action.currency_symbol,
            chain_name=action.chain_name,
            reasoning=action.reasoning,
            round=self.current_round,
        )
        self.conversation_history.append(offer)
        if self.initial_offer is None:
            self.initial_offer = offer
        self.current_offer = offer
        self.current_currency = offer.currency_symbol
        self.current_blockchain = offer.chain_name
        return offer

    def finalize(self, status: NegotiationStatus) -> None:
        self.status = status
        self.completed_at = datetime.now(timezone.utc)


def run_llm_negotiation(
    buyer_id: str,
    seller_id: str,
    buyer_decide: Callable[[NegotiationSession], NegotiationAction],
    seller_decide: Callable[[NegotiationSession], NegotiationAction],
    max_rounds: int = 10,
) -> NegotiationSession:
    """Alternates buyer_decide/seller_decide turns (buyer opens) until one
    side ACCEPTs, REJECTs, or WALKs_AWAY, or max_rounds is hit.

    buyer_decide/seller_decide are injected callables (typically closures
    around src.llm.agent_reasoning.decide) so this module has no direct
    dependency on the LLM router -- it only knows about NegotiationAction.
    """
    session = NegotiationSession(buyer_id=buyer_id, seller_id=seller_id, max_rounds=max_rounds)
    turns = [(buyer_id, buyer_decide), (seller_id, seller_decide)]

    for round_number in range(max_rounds):
        session.current_round = round_number
        agent_id, decide_fn = turns[round_number % 2]
        action = decide_fn(session)
        session.record_offer(agent_id, action)

        if action.action == DecisionAction.ACCEPT:
            session.finalize(NegotiationStatus.ACCEPTED)
            return session
        if action.action == DecisionAction.REJECT:
            session.finalize(NegotiationStatus.REJECTED)
            return session
        if action.action == DecisionAction.WALK_AWAY:
            session.finalize(NegotiationStatus.WALKED_AWAY)
            return session
        # OFFER / COUNTER_OFFER: continue to the next round.

    session.finalize(NegotiationStatus.MAX_ROUNDS_REACHED)
    return session
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_negotiation_engine.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/negotiation/llm_negotiation_engine.py tests/test_llm_negotiation_engine.py
git commit -m "feat: add LLM-driven multi-round negotiation state machine"
```

---

## Task 19: Hallucination classification and correlated result

**Files:**
- Modify: `src/llm/hallucination_detector.py` (append — `overpayment_pct` already exists and is fully implemented; do not change it)
- Modify: `tests/test_hallucinations.py` (append — the existing 4 tests must not be touched)

**Interfaces:**
- Consumes: nothing new (the existing `overpayment_pct(expected: float, paid: float) -> float` already implements the signed-percentage / `ValueError`-on-nonpositive-expected contract exactly).
- Produces: `class HallucinationDirection(str, Enum)` (`OVERPAYMENT`, `UNDERPAYMENT`, `ACCURATE`); `class HallucinationResult(BaseModel)` (`expected_value: float`, `paid_value: float`, `absolute_error: float`, `percentage_error: float`, `direction: HallucinationDirection`, `currency_symbol: str | None = None`, `chain_name: str | None = None`, `requested_model: str | None = None`, `actual_model: str | None = None`, `agent_type: str | None = None`, `risk_profile: str | None = None`, `economic_scenario: str | None = None`); `detect_hallucination(expected_value: float, paid_value: float, hallucination_threshold: float = 0.20, currency_symbol=None, chain_name=None, requested_model=None, actual_model=None, agent_type=None, risk_profile=None, economic_scenario=None) -> HallucinationResult`.

Verification first — confirm the existing file is exactly as expected before appending:

- [ ] **Step 1: Read the existing file to confirm it's unchanged from the stub**

Run: `cat src/llm/hallucination_detector.py`
Expected output (exactly):

```python
"""Compares expected value vs paid value to quantify over/underpayment.

Pure math, no LLM dependency -- this becomes meaningful once Phase 2 LLM
agents make pricing decisions that can diverge from the market's true price.
Phase 1 rule-based agents never call this in the live simulation loop since
they compute prices deterministically.
"""


def overpayment_pct(expected: float, paid: float) -> float:
    """Positive = overpaid, negative = underpaid, 0 = paid exactly the expected value."""
    if expected <= 0:
        raise ValueError("expected value must be positive")
    return (paid - expected) / expected * 100.0
```

If it differs, stop and reconcile before proceeding — this task only appends to this file.

- [ ] **Step 2: Write the failing test**

Append to the top imports of `tests/test_hallucinations.py` (the existing `import pytest` and `overpayment_pct` import stay as-is):

```python
from src.llm.hallucination_detector import HallucinationDirection, detect_hallucination
```

Append these tests to the end of `tests/test_hallucinations.py`:

```python
def test_detect_hallucination_classifies_overpayment_above_threshold():
    result = detect_hallucination(100.0, 150.0, hallucination_threshold=0.20)

    assert result.direction == HallucinationDirection.OVERPAYMENT
    assert result.absolute_error == pytest.approx(50.0)
    assert result.percentage_error == pytest.approx(50.0)


def test_detect_hallucination_classifies_underpayment_above_threshold():
    result = detect_hallucination(100.0, 50.0, hallucination_threshold=0.20)

    assert result.direction == HallucinationDirection.UNDERPAYMENT


def test_detect_hallucination_classifies_small_deviation_as_accurate():
    result = detect_hallucination(100.0, 105.0, hallucination_threshold=0.20)

    assert result.direction == HallucinationDirection.ACCURATE


def test_detect_hallucination_threshold_is_configurable_not_hardcoded():
    lenient = detect_hallucination(100.0, 115.0, hallucination_threshold=0.20)
    strict = detect_hallucination(100.0, 115.0, hallucination_threshold=0.10)

    assert lenient.direction == HallucinationDirection.ACCURATE
    assert strict.direction == HallucinationDirection.OVERPAYMENT


def test_detect_hallucination_carries_correlated_fields():
    result = detect_hallucination(
        100.0,
        150.0,
        currency_symbol="USDC",
        chain_name="ethereum",
        actual_model="anthropic/claude-sonnet-5",
        agent_type="buyer",
        risk_profile="low",
        economic_scenario="baseline",
    )

    assert result.currency_symbol == "USDC"
    assert result.actual_model == "anthropic/claude-sonnet-5"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_hallucinations.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_hallucination' from 'src.llm.hallucination_detector'`

- [ ] **Step 4: Write minimal implementation**

Append to the top of `src/llm/hallucination_detector.py` (after the module docstring, before `overpayment_pct`):

```python
from enum import Enum

from pydantic import BaseModel
```

Append to the end of `src/llm/hallucination_detector.py`:

```python
class HallucinationDirection(str, Enum):
    OVERPAYMENT = "OVERPAYMENT"
    UNDERPAYMENT = "UNDERPAYMENT"
    ACCURATE = "ACCURATE"


class HallucinationResult(BaseModel):
    expected_value: float
    paid_value: float
    absolute_error: float
    percentage_error: float
    direction: HallucinationDirection
    currency_symbol: str | None = None
    chain_name: str | None = None
    requested_model: str | None = None
    actual_model: str | None = None
    agent_type: str | None = None
    risk_profile: str | None = None
    economic_scenario: str | None = None


def detect_hallucination(
    expected_value: float,
    paid_value: float,
    hallucination_threshold: float = 0.20,
    currency_symbol: str | None = None,
    chain_name: str | None = None,
    requested_model: str | None = None,
    actual_model: str | None = None,
    agent_type: str | None = None,
    risk_profile: str | None = None,
    economic_scenario: str | None = None,
) -> HallucinationResult:
    """Classifies a settled transaction's pricing error, on top of (not
    instead of) the existing signed overpayment_pct -- see the design doc §9
    for why overpayment_pct's contract is not renegotiable."""
    signed_pct = overpayment_pct(expected_value, paid_value)
    percentage_error = abs(signed_pct)

    if percentage_error < hallucination_threshold * 100.0:
        direction = HallucinationDirection.ACCURATE
    elif signed_pct > 0:
        direction = HallucinationDirection.OVERPAYMENT
    else:
        direction = HallucinationDirection.UNDERPAYMENT

    return HallucinationResult(
        expected_value=expected_value,
        paid_value=paid_value,
        absolute_error=abs(paid_value - expected_value),
        percentage_error=percentage_error,
        direction=direction,
        currency_symbol=currency_symbol,
        chain_name=chain_name,
        requested_model=requested_model,
        actual_model=actual_model,
        agent_type=agent_type,
        risk_profile=risk_profile,
        economic_scenario=economic_scenario,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hallucinations.py -v`
Expected: 9 passed (the original 4 plus these 5)

- [ ] **Step 6: Commit**

```bash
git add src/llm/hallucination_detector.py tests/test_hallucinations.py
git commit -m "feat: classify hallucination direction with a configurable threshold"
```

---

## Task 20: Wallet-mutation invariant test

**Files:**
- Test: `tests/test_llm_wallet_invariant.py`

**Interfaces:**
- Consumes: `decide` (Task 15), `build_decision_context`, `AgentUtilityContext`, `TransactionContext` (Tasks 12-13), existing `Wallet` (`src.agents.wallet`), existing `settle`/`Transaction`/`TransactionStatus` (`src.transactions.settlement`, `src.transactions.transaction`).
- Produces: no new production code — this task is pure verification that the invariant from the design doc §1 ("An LLM call can never mutate a wallet, ledger, or transaction directly") holds across the full `decide()` → settlement path.

- [ ] **Step 1: Write the test**

```python
# tests/test_llm_wallet_invariant.py
import json

import httpx

from src.agents.wallet import Wallet
from src.blockchain.routing_engine import CurrencyChainOption
from src.economy.macro_state import MacroState
from src.llm.agent_reasoning import (
    AgentUtilityContext,
    TransactionContext,
    build_decision_context,
    decide,
)
from src.llm.llm_router import OPENROUTER_BASE_URL, RetryConfig, load_model_roster
from src.transactions.settlement import settle
from src.transactions.transaction import Transaction, TransactionStatus


def _decision_json() -> str:
    return json.dumps(
        {
            "action": "ACCEPT",
            "proposed_currency": "USDC",
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 100.0,
            "reasoning": "accepting the offer",
        }
    )


def test_llm_decision_never_mutates_wallet_before_settlement():
    buyer_wallet = Wallet(balances={"USDC": 1000.0})
    seller_wallet = Wallet(balances={"USDC": 0.0})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": _decision_json()}}]})

    client = httpx.Client(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler))
    roster = load_model_roster()

    agent_context = AgentUtilityContext(
        agent_id="buyer-1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances=dict(buyer_wallet.balances),
    )
    candidates = [
        CurrencyChainOption(
            currency_symbol="USDC",
            chain_name="ethereum",
            governance_score=0.95,
            liquidity_score=0.97,
            peg_error=0.0003,
            gas_fee=2.5,
            finality_seconds=12.0,
            genius_compliant=True,
        )
    ]
    macro = MacroState()
    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, TransactionContext(is_cross_border=False)
    )

    outcome = decide(
        "buyer", context, roster, client, {"USDC"}, {"ethereum"}, retry_config=RetryConfig(sleep_fn=lambda s: None)
    )

    # The LLM call and decision-adaptation must never touch the wallet.
    assert buyer_wallet.balances["USDC"] == 1000.0
    assert seller_wallet.balances["USDC"] == 0.0

    # Only the existing deterministic settlement path may move money.
    tx = Transaction(
        buyer_id="buyer-1",
        seller_id="seller-1",
        good_name="cloud_compute",
        currency_symbol=outcome.negotiation_action.currency_symbol,
        chain_name=outcome.negotiation_action.chain_name,
        gas_fee=2.5,
        expected_value=100.0,
        paid_value=outcome.negotiation_action.price,
        timestep=0,
    )
    settle(tx, buyer_wallet, seller_wallet)

    assert tx.status == TransactionStatus.SETTLED
    assert buyer_wallet.balances["USDC"] == 900.0
    assert seller_wallet.balances["USDC"] == 100.0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_llm_wallet_invariant.py -v`
Expected: 1 passed (this test should pass immediately given Tasks 1-19's implementation — if it fails, that indicates a real invariant violation somewhere in the `decide()`/`adapt_decision()` chain that must be fixed before proceeding, not a missing-module error)

- [ ] **Step 3: Commit**

```bash
git add tests/test_llm_wallet_invariant.py
git commit -m "test: verify LLM decisions never mutate a wallet before settlement"
```

---

## Task 21: W&B logging for LLM-path metrics

**Files:**
- Modify: `metrics/wandb_logger.py`
- Test: `tests/test_wandb_llm_metrics.py`

**Interfaces:**
- Consumes: `HallucinationResult`, `HallucinationDirection` (Task 19).
- Produces: `WandbRunLogger.log_llm_metrics(self, hallucination_results: list[HallucinationResult], model_attempts_by_call: list[list[str]], step: int) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wandb_llm_metrics.py
from metrics.wandb_logger import WandbRunLogger
from src.llm.hallucination_detector import HallucinationDirection, HallucinationResult


class _FakeWandb:
    def __init__(self):
        self.logged: list[tuple[dict, int | None]] = []

    def log(self, data: dict, step: int | None = None) -> None:
        self.logged.append((data, step))


def _logger_with_fake_wandb() -> WandbRunLogger:
    # Bypasses __init__ (which calls the real wandb.init) since wandb is an
    # optional dependency this test suite must not require installed.
    logger = WandbRunLogger.__new__(WandbRunLogger)
    logger._wandb = _FakeWandb()
    return logger


def _hallucination(direction: HallucinationDirection) -> HallucinationResult:
    paid = 150.0 if direction == HallucinationDirection.OVERPAYMENT else 100.0
    return HallucinationResult(
        expected_value=100.0, paid_value=paid, absolute_error=abs(paid - 100.0), percentage_error=0.0, direction=direction
    )


def test_log_llm_metrics_computes_hallucination_and_fallback_rates():
    logger = _logger_with_fake_wandb()
    results = [_hallucination(HallucinationDirection.ACCURATE), _hallucination(HallucinationDirection.OVERPAYMENT)]
    attempts = [["model-a"], ["model-a", "model-b"]]

    logger.log_llm_metrics(results, attempts, step=5)

    data, step = logger._wandb.logged[0]
    assert step == 5
    assert data["llm_hallucination_rate"] == 0.5
    assert data["llm_fallback_rate"] == 0.5


def test_log_llm_metrics_handles_empty_inputs_without_dividing_by_zero():
    logger = _logger_with_fake_wandb()

    logger.log_llm_metrics([], [], step=0)

    data, _ = logger._wandb.logged[0]
    assert data["llm_hallucination_rate"] == 0.0
    assert data["llm_fallback_rate"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wandb_llm_metrics.py -v`
Expected: FAIL with `AttributeError: 'WandbRunLogger' object has no attribute 'log_llm_metrics'`

- [ ] **Step 3: Write minimal implementation**

Modify `metrics/wandb_logger.py`: add an import and one new method. Add to the top imports:

```python
from src.llm.hallucination_detector import HallucinationDirection, HallucinationResult
```

Append this method to the `WandbRunLogger` class (after `on_timestep`, before `finish`):

```python
    def log_llm_metrics(
        self,
        hallucination_results: list[HallucinationResult],
        model_attempts_by_call: list[list[str]],
        step: int,
    ) -> None:
        """Additive: logs LLM-path-specific metrics alongside whatever the
        caller already logs via on_timestep(). Only the LLM path calls this;
        Phase 1 rule-based runs never do."""
        total = len(hallucination_results)
        hallucination_rate = (
            sum(1 for result in hallucination_results if result.direction != HallucinationDirection.ACCURATE) / total
            if total
            else 0.0
        )
        fallback_rate = (
            sum(1 for attempts in model_attempts_by_call if len(attempts) > 1) / len(model_attempts_by_call)
            if model_attempts_by_call
            else 0.0
        )
        self._wandb.log(
            {"llm_hallucination_rate": hallucination_rate, "llm_fallback_rate": fallback_rate},
            step=step,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wandb_llm_metrics.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add metrics/wandb_logger.py tests/test_wandb_llm_metrics.py
git commit -m "feat: log LLM hallucination and fallback rates to W&B"
```

---

## Task 22: Live governance-prompting experiment (the "actual results" deliverable)

**Files:**
- Create: `experiments/experiment_007_governance_prompting.py` (this file currently exists as a 0-byte/empty stub)
- Create: `tests/test_experiment_007_live.py`
- Modify: `pyproject.toml` (register the `live` pytest marker)
- Modify: `.env.example` (document `RUN_LIVE_LLM_TESTS`)

**Interfaces:**
- Consumes: everything from Tasks 1-21 — `load_agent_profiles`/`build_agent` (existing), `generate_candidates` (existing), `load_currency_universe`/`load_chain_universe` (existing), `MacroState` (existing), `build_decision_context`/`render_prompt`/`PROMPT_VERSIONS` (Tasks 13-14), `Decision`/`DecisionAction` (Task 7), `detect_hallucination` (Task 19), `build_openrouter_client`/`call_model`/`ModelCallFailedError`/`load_model_roster` (Tasks 8-10).
- Produces: `run_cell(model_id: str, governance_prompt_enabled: bool, client: httpx.Client) -> dict`; `main() -> None` (script entry point).

Per the design doc §12: full factorial of 5 pinned models (via the `model_comparison` routing policy — **no cross-model substitution**, so a model that fails is recorded as excluded, not silently replaced) × 2 conditions (baseline vs. governance-emphasized prompt), with agent profile, risk parameters, market state, currencies, and the transaction held constant across every cell.

- [ ] **Step 1: Register the `live` pytest marker**

Modify `pyproject.toml`'s `[tool.pytest.ini_options]` section from:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

to:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = ["live: hits a real external API (OpenRouter); requires RUN_LIVE_LLM_TESTS=1 to run"]
```

- [ ] **Step 2: Document the live-test opt-in in `.env.example`**

Modify `.env.example`, appending after the existing `OPENROUTER_API_KEY=` line:

```text
# Set to 1 to let the one test marked @pytest.mark.live make a single real
# OpenRouter call (tests/test_experiment_007_live.py). Unset/0 by default so
# `pytest` never hits the network or spends API credits.
RUN_LIVE_LLM_TESTS=0
```

- [ ] **Step 3: Write the experiment script**

```python
# experiments/experiment_007_governance_prompting.py
"""Experiment 007: does governance-prompting shift USDC vs. USDT selection?

Hypothesis 3 (docs/superpowers/specs/2026-07-22-phase2-llm-negotiation-layer-design.md
§1, §12): the more an agent is prompted to reason about governance/compliance,
the more it should favor the better-governed stablecoin (USDC) over the more
liquid but less transparent one (USDT).

Design: 5 pinned models (model_comparison routing policy -- no cross-model
substitution, so "model" stays a clean experimental factor, not confounded
with reliability) x 2 conditions (baseline vs. governance-emphasized
prompt). Agent profile, risk parameters, market state, available currencies,
and the transaction opportunity are held constant across every cell.

Dependent variables are tiered by evidential strength, per the design doc:
primary = observed currency selection; secondary = negotiation
outcome/hallucination rate; exploratory = whether reported_reasoning
mentions governance (never treated as proof of *why* a choice was made).
"""

import json
import os

import httpx
from dotenv import load_dotenv

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.blockchain.chain import load_chain_universe
from src.blockchain.routing_engine import generate_candidates
from src.currencies.currency import load_currency_universe
from src.economy.macro_state import MacroState
from src.llm.agent_reasoning import AgentDecisionContext, TransactionContext, build_decision_context, render_prompt
from src.llm.decision_schema import Decision, DecisionAction
from src.llm.hallucination_detector import detect_hallucination
from src.llm.llm_router import ModelCallFailedError, build_openrouter_client, call_model, load_model_roster
from src.llm.market_intelligence import load_currency_profile
from src.utils.constants import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

GOOD_TRUE_PRICE = 100.0
_COMPARED_CURRENCIES = ("USDC", "USDT")


def _build_context(governance_prompt_enabled: bool) -> AgentDecisionContext:
    """Held constant across every cell: the "consumer" agent profile (fixed
    risk parameters), the currency/chain universe, and the transaction
    opportunity. Only governance_prompt_enabled varies here -- model varies
    in run_cell."""
    profiles = load_agent_profiles()
    agent = build_agent(profiles["consumer"])
    currencies = load_currency_universe()
    chains = load_chain_universe()

    candidates = [
        option
        for option in generate_candidates(agent.wallet.balances, currencies, chains)
        if option.currency_symbol in _COMPARED_CURRENCIES
    ]
    currency_profiles = {
        symbol: profile for symbol in _COMPARED_CURRENCIES if (profile := load_currency_profile(symbol)) is not None
    }
    macro = MacroState()

    return build_decision_context(
        agent.build_llm_context(),
        candidates,
        currency_profiles,
        macro,
        macro,
        TransactionContext(is_cross_border=False),
        governance_prompt_enabled=governance_prompt_enabled,
    )


def run_cell(model_id: str, governance_prompt_enabled: bool, client: httpx.Client) -> dict:
    """Runs one (model, condition) cell. Returns a plain dict rather than a
    pydantic model since this is a script-level result row, not a value
    passed between typed interfaces."""
    context = _build_context(governance_prompt_enabled)
    schema_json = json.dumps(Decision.model_json_schema())
    prompt = render_prompt("buyer", context, schema_json)

    try:
        decision = call_model(prompt, model_id, client)
    except ModelCallFailedError as exc:
        # No cross-model substitution for this experiment: an excluded cell
        # is reported as excluded, never silently backfilled by a different
        # model -- see the design doc §4/§7 on keeping "model" a clean factor.
        return {
            "model_id": model_id,
            "governance_prompt_enabled": governance_prompt_enabled,
            "excluded": True,
            "exclusion_reason": exc.reason,
        }

    hallucination = None
    if decision.action in (DecisionAction.OFFER, DecisionAction.COUNTER_OFFER, DecisionAction.ACCEPT):
        hallucination = detect_hallucination(
            GOOD_TRUE_PRICE, decision.price, currency_symbol=decision.proposed_currency, actual_model=model_id
        )

    return {
        "model_id": model_id,
        "governance_prompt_enabled": governance_prompt_enabled,
        "excluded": False,
        "selected_currency": decision.proposed_currency,
        "action": decision.action.value,
        "price": decision.price,
        "reported_reasoning": decision.reasoning,
        "hallucination_direction": hallucination.direction.value if hallucination else None,
    }


def _print_results_table(results: list[dict]) -> None:
    print(f"{'model':<30} {'condition':<12} {'currency':<8} {'action':<14} {'price':>8}  reasoning")
    for row in results:
        if row["excluded"]:
            print(f"{row['model_id']:<30} EXCLUDED: {row['exclusion_reason']}")
            continue
        condition = "governance" if row["governance_prompt_enabled"] else "baseline"
        print(
            f"{row['model_id']:<30} {condition:<12} {row['selected_currency']:<8} {row['action']:<14} "
            f"{row['price']:>8.2f}  {row['reported_reasoning'][:60]}"
        )

    included = [r for r in results if not r["excluded"]]
    usdc_baseline = sum(1 for r in included if not r["governance_prompt_enabled"] and r["selected_currency"] == "USDC")
    usdc_governance = sum(1 for r in included if r["governance_prompt_enabled"] and r["selected_currency"] == "USDC")
    baseline_total = sum(1 for r in included if not r["governance_prompt_enabled"])
    governance_total = sum(1 for r in included if r["governance_prompt_enabled"])
    print(
        f"\nPrimary outcome -- USDC selection rate: "
        f"baseline {usdc_baseline}/{baseline_total}, governance-emphasized {usdc_governance}/{governance_total}"
    )
    excluded_models = [r["model_id"] for r in results if r["excluded"]]
    if excluded_models:
        print(f"Excluded (no substitution): {excluded_models}")


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set -- see .env.example")

    roster = load_model_roster()
    client = build_openrouter_client(api_key)
    pinned_models = [roster.resolve(label) for label in roster.routing_policies.model_comparison.pinned_models]

    results = [
        run_cell(model_id, governance_prompt_enabled, client)
        for governance_prompt_enabled in (False, True)
        for model_id in pinned_models
    ]

    _print_results_table(results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the live smoke test**

```python
# tests/test_experiment_007_live.py
import os

import pytest

from experiments.experiment_007_governance_prompting import run_cell
from src.llm.llm_router import build_openrouter_client, load_model_roster


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "1", reason="Set RUN_LIVE_LLM_TESTS=1 to run live OpenRouter calls"
)
def test_governance_prompting_cell_runs_against_the_real_api():
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY must be set in .env to run this live test"

    roster = load_model_roster()
    client = build_openrouter_client(api_key)
    primary_model = roster.resolve(roster.routing_policies.default_reliability_chain.primary)

    result = run_cell(primary_model, governance_prompt_enabled=True, client=client)

    assert result["model_id"] == primary_model
    if not result["excluded"]:
        assert result["selected_currency"] in ("USDC", "USDT")
```

- [ ] **Step 5: Run the mocked test suite to confirm nothing broke**

Run: `pytest tests/ -v -m "not live"`
Expected: all tests pass; `test_experiment_007_live.py` is collected but skipped (no `RUN_LIVE_LLM_TESTS` set)

- [ ] **Step 6: Run the live smoke test once, manually, to prove the whole pipeline against the real API**

Run: `RUN_LIVE_LLM_TESTS=1 pytest tests/test_experiment_007_live.py -v -m live`
Expected: PASS (requires a real, funded `OPENROUTER_API_KEY` in `.env`) — if it fails on a specific model slug, that is the runtime preflight check (Task 8's `verify_model_roster`) doing its job; fix the slug in `configs/llm/models.yaml` and re-run.

- [ ] **Step 7: Run the full live experiment and inspect the results table**

Run: `python -m experiments.experiment_007_governance_prompting`
Expected: prints a 10-row results table (5 models x 2 conditions) followed by the primary-outcome USDC-selection-rate summary; no unhandled exceptions. This is the phase's actual-results deliverable — read the table, don't just check it printed something.

- [ ] **Step 8: Commit**

```bash
git add experiments/experiment_007_governance_prompting.py tests/test_experiment_007_live.py pyproject.toml .env.example
git commit -m "feat: implement live governance-prompting experiment across 5 models"
```

---

## Task 23: Wire live price snapshots into the decision context and prompt

**Files:**
- Modify: `src/llm/agent_reasoning.py` (append/extend)
- Modify: `src/llm/prompts/buyer_prompt.txt`, `seller_prompt.txt`, `investor_prompt.txt`, `bank_prompt.txt`
- Modify: `tests/test_agent_reasoning.py` (append)

**Interfaces:**
- Consumes: `LivePriceSnapshot` (Task 16, `src.llm.market_intelligence`).
- Produces: `AgentDecisionContext.live_price_snapshots: dict[str, LivePriceSnapshot] = {}`; `build_decision_context(..., live_price_snapshots: dict[str, LivePriceSnapshot] | None = None)` (new keyword-only-by-convention parameter, appended after the existing ones so every earlier call site keeps working unchanged); `render_prompt` now also fills a `{live_price_block}` placeholder.

Design doc §5/§6 gap this closes: `AgentDecisionContext` was assembled in Task 13 with the static profile corpus only — the "optional live price snapshot per candidate currency" half of §5's market-intelligence bullet wasn't wired in yet because `LivePriceSnapshot` didn't exist until Task 16. This task closes that gap.

- [ ] **Step 1: Write the failing test**

Append to the top imports of `tests/test_agent_reasoning.py`:

```python
from datetime import datetime, timezone

from src.llm.market_intelligence import LivePriceSnapshot
```

Append these tests to the end of `tests/test_agent_reasoning.py`:

```python
def test_build_decision_context_filters_live_price_snapshots_to_candidates_only():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    snapshots = {
        "USDC": LivePriceSnapshot(ticker="X:USDCUSD", price=1.0001, retrieval_timestamp=datetime.now(timezone.utc)),
        "USDT": LivePriceSnapshot(ticker="X:USDTUSD", price=0.9998, retrieval_timestamp=datetime.now(timezone.utc)),
    }

    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, txn_context, live_price_snapshots=snapshots
    )

    assert set(context.live_price_snapshots.keys()) == {"USDC"}


def test_render_prompt_includes_live_price_block():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    snapshots = {"USDC": LivePriceSnapshot(ticker="X:USDCUSD", price=1.0001, retrieval_timestamp=datetime.now(timezone.utc))}
    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, txn_context, live_price_snapshots=snapshots
    )

    prompt = render_prompt("buyer", context, "{}")

    assert "1.0001" in prompt


def test_render_prompt_reports_unavailable_live_price_explicitly_not_silently():
    agent_context = AgentUtilityContext(
        agent_id="a1",
        agent_class="buyer",
        risk_profile="low",
        utility_type="crra",
        risk_aversion=3.0,
        wallet_balances={"USDC": 1000.0},
    )
    candidates = [_option(currency_symbol="USDC")]
    macro = MacroState()
    txn_context = TransactionContext(is_cross_border=False)
    snapshots = {
        "USDC": LivePriceSnapshot(
            ticker="X:USDCUSD",
            price=None,
            retrieval_timestamp=datetime.now(timezone.utc),
            unavailable_reason="no data returned for this ticker",
        )
    }
    context = build_decision_context(
        agent_context, candidates, {}, macro, macro, txn_context, live_price_snapshots=snapshots
    )

    prompt = render_prompt("buyer", context, "{}")

    assert "unavailable" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: FAIL with `TypeError: build_decision_context() got an unexpected keyword argument 'live_price_snapshots'`

- [ ] **Step 3: Write minimal implementation**

Modify the top imports of `src/llm/agent_reasoning.py` — change the existing `from src.llm.market_intelligence import CurrencyProfile` line to:

```python
from src.llm.market_intelligence import CurrencyProfile, LivePriceSnapshot
```

Modify `AgentDecisionContext` (add one field after `currency_profiles`):

```python
class AgentDecisionContext(BaseModel):
    agent: AgentUtilityContext
    candidates: list[CurrencyChainOption]
    currency_profiles: dict[str, CurrencyProfile] = {}
    live_price_snapshots: dict[str, LivePriceSnapshot] = {}
    objective_macro_state: MacroState
    perceived_macro_state: MacroState
    transaction_context: TransactionContext
    opponent_offer: NegotiationAction | None = None
    conversation_history: list[str] = []
    governance_prompt_enabled: bool = False
```

Modify `build_decision_context` (full new version):

```python
def build_decision_context(
    agent_context: AgentUtilityContext,
    candidates: list[CurrencyChainOption],
    currency_profiles: dict[str, CurrencyProfile],
    objective_macro_state: MacroState,
    perceived_macro_state: MacroState,
    transaction_context: TransactionContext,
    opponent_offer: NegotiationAction | None = None,
    conversation_history: list[str] | None = None,
    governance_prompt_enabled: bool = False,
    live_price_snapshots: dict[str, LivePriceSnapshot] | None = None,
) -> AgentDecisionContext:
    candidate_symbols = {candidate.currency_symbol for candidate in candidates}
    relevant_profiles = {
        symbol: profile for symbol, profile in currency_profiles.items() if symbol in candidate_symbols
    }
    relevant_snapshots = {
        symbol: snapshot
        for symbol, snapshot in (live_price_snapshots or {}).items()
        if symbol in candidate_symbols
    }
    return AgentDecisionContext(
        agent=agent_context,
        candidates=candidates,
        currency_profiles=relevant_profiles,
        live_price_snapshots=relevant_snapshots,
        objective_macro_state=objective_macro_state,
        perceived_macro_state=perceived_macro_state,
        transaction_context=transaction_context,
        opponent_offer=opponent_offer,
        conversation_history=conversation_history or [],
        governance_prompt_enabled=governance_prompt_enabled,
    )
```

Add a new formatting helper and wire it into `render_prompt`. Append this function near the other `_format_*` helpers in `src/llm/agent_reasoning.py`:

```python
def _format_live_price_block(snapshots: dict[str, LivePriceSnapshot]) -> str:
    if not snapshots:
        return "(no live price data available)"
    lines = []
    for symbol, snapshot in snapshots.items():
        if snapshot.price is None:
            lines.append(f"- {symbol}: live price unavailable ({snapshot.unavailable_reason})")
        else:
            lines.append(
                f"- {symbol}: {snapshot.price} (retrieved {snapshot.retrieval_timestamp.isoformat()}, {snapshot.source})"
            )
    return "\n".join(lines)
```

Modify `render_prompt`'s `fields` dict to add one entry (insert after `"currency_profiles_block"`):

```python
        "live_price_block": _format_live_price_block(context.live_price_snapshots),
```

Modify all four prompt templates, inserting a new section right after the "Background information" section and before "Macro-economic conditions" (shown here for `buyer_prompt.txt`; apply the identical insertion to `seller_prompt.txt`, `investor_prompt.txt`, and `bank_prompt.txt`):

```text
# Live price data -- current, not historical
{live_price_block}

# Macro-economic conditions
```

(i.e. the line `# Macro-economic conditions` that already exists in each file is now preceded by the two new lines above it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_reasoning.py -v`
Expected: 13 passed

- [ ] **Step 5: Run the full test suite to confirm no regression**

Run: `pytest tests/ -v -m "not live"`
Expected: all passed (the new `live_price_snapshots` parameter defaults to `None`/`{}`, so every earlier call site in Tasks 12-22 that doesn't pass it keeps working unchanged)

- [ ] **Step 6: Commit**

```bash
git add src/llm/agent_reasoning.py src/llm/prompts/*.txt tests/test_agent_reasoning.py
git commit -m "feat: wire optional live price snapshots into the LLM prompt context"
```

---

## Task 24: Persist experiment_007 decisions to the database

**Files:**
- Modify: `experiments/experiment_007_governance_prompting.py`
- Test: `tests/test_experiment_007_persistence.py`

**Interfaces:**
- Consumes: `LLMDecisionRepository`, `LLMDecisionLogEntry` (Task 6); `run_cell` (Task 22).
- Produces: `run_cell(model_id: str, governance_prompt_enabled: bool, client: httpx.Client, repository: "LLMDecisionRepository | None" = None) -> dict` (new optional keyword-only-by-convention parameter; existing calls without it are unaffected).

Closes the remaining design-doc gap: Task 6 built `LLMDecisionRecord`/`LLMDecisionRepository` and Task 22 built the experiment, but nothing yet calls `repository.record(...)` with the experiment's actual decisions — every completed cell must be persisted, not just printed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment_007_persistence.py
import json

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, LLMDecisionRecord
from database.repository import LLMDecisionRepository
from experiments.experiment_007_governance_prompting import run_cell


def _decision_json() -> str:
    return json.dumps(
        {
            "action": "OFFER",
            "proposed_currency": "USDC",
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 100.0,
            "reasoning": "USDC has stronger governance",
        }
    )


def test_run_cell_persists_a_decision_record_when_given_a_repository():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": _decision_json()}}]})

    client = httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    repository = LLMDecisionRepository(session)

    result = run_cell("anthropic/claude-sonnet-5", governance_prompt_enabled=True, client=client, repository=repository)
    session.commit()

    assert result["excluded"] is False
    rows = session.query(LLMDecisionRecord).all()
    assert len(rows) == 1
    assert rows[0].actual_model == "anthropic/claude-sonnet-5"
    assert rows[0].governance_prompt_enabled is True
    assert rows[0].currency == "USDC"


def test_run_cell_without_a_repository_still_works_and_persists_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": _decision_json()}}]})

    client = httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))

    result = run_cell("anthropic/claude-sonnet-5", governance_prompt_enabled=False, client=client)

    assert result["excluded"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_experiment_007_persistence.py -v`
Expected: FAIL with `TypeError: run_cell() got an unexpected keyword argument 'repository'`

- [ ] **Step 3: Write minimal implementation**

Modify `experiments/experiment_007_governance_prompting.py`: add an import and update `run_cell` and `main`.

Add to the top imports:

```python
from database.repository import LLMDecisionLogEntry, LLMDecisionRepository
from src.llm.agent_reasoning import PROMPT_VERSIONS, hash_rendered_prompt
from src.utils.helpers import generate_id
```

Replace the `run_cell` function with:

```python
def run_cell(
    model_id: str,
    governance_prompt_enabled: bool,
    client: httpx.Client,
    repository: LLMDecisionRepository | None = None,
) -> dict:
    """Runs one (model, condition) cell. Returns a plain dict rather than a
    pydantic model since this is a script-level result row, not a value
    passed between typed interfaces. If a repository is supplied, also
    persists the decision -- callers that only want the printed table (e.g.
    tests/test_experiment_007_live.py) may omit it."""
    context = _build_context(governance_prompt_enabled)
    schema_json = json.dumps(Decision.model_json_schema())
    prompt = render_prompt("buyer", context, schema_json)

    try:
        decision = call_model(prompt, model_id, client)
    except ModelCallFailedError as exc:
        return {
            "model_id": model_id,
            "governance_prompt_enabled": governance_prompt_enabled,
            "excluded": True,
            "exclusion_reason": exc.reason,
        }

    hallucination = None
    if decision.action in (DecisionAction.OFFER, DecisionAction.COUNTER_OFFER, DecisionAction.ACCEPT):
        hallucination = detect_hallucination(
            GOOD_TRUE_PRICE, decision.price, currency_symbol=decision.proposed_currency, actual_model=model_id
        )

    if repository is not None:
        repository.record(
            LLMDecisionLogEntry(
                decision_id=generate_id("dec"),
                simulation_id="experiment_007_governance_prompting",
                timestep=0,
                agent_id=context.agent.agent_id,
                agent_type=context.agent.agent_class,
                requested_model=model_id,
                actual_model=model_id,
                fallback_used=False,
                fallback_reason=None,
                model_attempts=[model_id],
                prompt_version=PROMPT_VERSIONS["buyer"],
                rendered_prompt_hash=hash_rendered_prompt(prompt),
                action=decision.action.value,
                currency=decision.proposed_currency,
                chain=decision.proposed_chain,
                amount=decision.amount,
                price=decision.price,
                reported_reasoning=decision.reasoning,
                negotiation_id=None,
                round=0,
                risk_profile=context.agent.risk_profile,
                utility_type=context.agent.utility_type,
                utility_parameters={"risk_aversion": context.agent.risk_aversion, "eis": context.agent.eis},
                scenario="experiment_007_governance_prompting",
                domestic_or_cross_border="cross_border" if context.transaction_context.is_cross_border else "domestic",
                governance_prompt_enabled=governance_prompt_enabled,
            )
        )

    return {
        "model_id": model_id,
        "governance_prompt_enabled": governance_prompt_enabled,
        "excluded": False,
        "selected_currency": decision.proposed_currency,
        "action": decision.action.value,
        "price": decision.price,
        "reported_reasoning": decision.reasoning,
        "hallucination_direction": hallucination.direction.value if hallucination else None,
    }
```

Modify `main()` to open a real database session and pass the repository through:

```python
def main() -> None:
    from database.session import create_all_tables, new_session

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set -- see .env.example")

    roster = load_model_roster()
    client = build_openrouter_client(api_key)
    pinned_models = [roster.resolve(label) for label in roster.routing_policies.model_comparison.pinned_models]

    create_all_tables()
    session = new_session()
    repository = LLMDecisionRepository(session)

    results = [
        run_cell(model_id, governance_prompt_enabled, client, repository)
        for governance_prompt_enabled in (False, True)
        for model_id in pinned_models
    ]
    session.commit()

    _print_results_table(results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_experiment_007_persistence.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full mocked test suite one final time**

Run: `pytest tests/ -v -m "not live"`
Expected: all tests across all 24 tasks pass

- [ ] **Step 6: Commit**

```bash
git add experiments/experiment_007_governance_prompting.py tests/test_experiment_007_persistence.py
git commit -m "feat: persist experiment_007 decisions to the database"
```

---

## Known scope trade-offs (read before executing)

- **`HallucinationRecord` (existing table) is intentionally not populated by this plan.** It requires a `transaction_id` referencing a real settled `Transaction` row, but `experiment_007_governance_prompting.py` (Tasks 22/24) deliberately makes one `call_model` decision per cell rather than running a full `run_llm_negotiation` (Task 18) → settlement round-trip, to keep the flagship experiment tractable. `HallucinationResult` objects are still computed per cell (Task 19) and surfaced two ways: the printed results table (Task 22) and the aggregate `llm_hallucination_rate` logged to W&B (Task 21). Wiring a full negotiation-and-settlement version of this experiment (or a new one) that populates `HallucinationRecord` with a real `transaction_id` is natural follow-up work, not part of this plan.
- **`run_llm_negotiation` (Task 18) is not exercised by `experiment_007`.** It's a general-purpose building block, unit-tested on its own; `experiment_007` uses the single-call shortcut described above. A future experiment (e.g. one of `008`-`011`, or a new one) can call `run_llm_negotiation` with `decide()`-backed closures for `buyer_decide`/`seller_decide` to get a full multi-round negotiation.
- **`experiments/008`-`011` remain empty stubs.** Only `007` is implemented, per the design doc §12's explicit scope decision.

## Self-Review Notes

- **Spec coverage:** every numbered section of `docs/superpowers/specs/2026-07-22-phase2-llm-negotiation-layer-design.md` (§1-§14) maps to at least one task above; the two gaps found during review (live price snapshots not wired into the prompt context, and no repository call actually persisting a decision) were closed by adding Tasks 23 and 24 rather than left as silent omissions. The `HallucinationRecord`/`run_llm_negotiation` scope trade-offs above are the two remaining gaps, called out explicitly rather than silently dropped.
- **Placeholder scan:** no `TBD`/`TODO`/"implement later" markers; every step includes complete, runnable code.
- **Type consistency:** verified `LLMDecisionLogEntry` (Task 6) fields match `LLMDecisionRecord` ORM columns 1:1; `MarketSnapshotLogEntry` matches `MarketSnapshotRecord` 1:1; `render_prompt`'s `fields` dict keys match every prompt template's placeholders across Tasks 14 and 23; `build_agent`'s new keyword arguments (Task 12) are introduced in the same task that adds the corresponding required fields to `BaseAgent`, so no intermediate task leaves a required field unpopulated; test-count assertions in each task's "Step 4" were recomputed against the cumulative test list in the shared files (`tests/test_llm_router.py`, `tests/test_agent_reasoning.py`) and one miscount (Task 23) was corrected from 15 to 13.
