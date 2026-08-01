"""Per-sandbox synthetic shock schedules for the 12 factor-isolation sandbox
matrix cells.

Bug this fixes: `master_simulation.yaml`'s 18 shocks target real-universe
currency symbols (USDC, PAXG, EURT, ...) exclusively for every
currency-targeted shock. None of those symbols exist in any sandbox's
synthetic 2-currency universe (`src.currencies.sandbox_currencies`), so
`apply_currency_shock`/`TrustLedger.update`'s per-currency offsets silently
no-op for all 13 currency-targeted shocks in every one of the 12 sandbox
cells -- only the 5 macro-level shocks (no `target_currency`) still apply.
That means H4 (crisis proximity -> gold preference) is untestable in any
sandbox, and none of the 6 sandboxes' own currency pairs ever experience a
governance_downgrade/depeg_event/liquidity_crunch/etc. shock to stress-test
the very factor each sandbox isolates.

`build_sandbox_scenario` fixes this parametrically (NOT via 6 hand-authored
YAML files -- the sandbox symbols are Python-constructed, per
`SANDBOX_CURRENCY_PAIRS`, not loaded from YAML): it copies
`master_simulation`'s 365-day macro backdrop and its 5 macro-level shocks
(which already apply fine to any currency universe, real or synthetic) and
adds, per sandbox:

1. One `crisis_warning` -> `depeg_event` pair targeting one of that
   sandbox's two currency symbols, giving every sandbox a genuine
   crisis-proximity shock (exercising the shock mechanism and TrustLedger
   dynamics) that stress-tests its own currency pair. H4's gold-preference
   DIRECTION specifically, however, is only meaningfully testable in the
   two sandboxes that actually contain a gold-backed option --
   `asset_backing_vs_liquidity` and `asset_backing_vs_stability` -- since
   only there is there a gold currency for agents to shift TOWARD. In
   those two sandboxes the pair deliberately targets the NON-gold option,
   so a flight-to-gold response is observable; the other 4 sandboxes have
   no gold option at all, so their crisis/depeg pairs test the general
   crisis-proximity mechanism and each sandbox's own isolated factor, not
   H4's gold-preference direction.
2. One additional currency-targeted shock relevant to the OTHER factor that
   sandbox isolates (e.g. `liquidity_crunch` on the lower-liquidity option
   in a liquidity-vs-X sandbox), so both of a sandbox's isolated dimensions
   see some stress, not just one.

Per-sandbox target/gap choices are a documented judgment call (see
`_SANDBOX_SHOCK_PLANS` below and this task's own commit message) --
`SANDBOX_CURRENCY_PAIRS`'s "option_a"/"option_b" don't share a single
"weaker side" convention across all 6 pairs (which dimension a given
config's "option_a" favors differs sandbox to sandbox), so each pick is
reasoned about individually rather than mechanically.

Day layout: the H4 pair sits at day 110 (`_CRISIS_WARNING_DAY`) plus a
0/5/10/20-day gap (mirroring master_simulation.yaml's own H4 sweep), and the
additional shock sits at day 160 (`_ADDITIONAL_SHOCK_DAY`). Both are >=15
days clear of master_simulation's last macro shock (day 90) and of each
other even at the widest gap (20), matching Task 12's non-confounding
spacing convention -- the only shocks deliberately close together are the
crisis_warning/depeg_event pair itself, since proximity is exactly what H4
tests.
"""

from dataclasses import dataclass

from src.currencies.currency import CurrencyConfig
from src.economy.shocks import ScenarioConfig, ShockEvent, ShockType

_CRISIS_WARNING_DAY = 110
_ADDITIONAL_SHOCK_DAY = 160
_CRISIS_WARNING_MAGNITUDE = 0.05
_DEPEG_MAGNITUDE = 0.15


@dataclass(frozen=True)
class _SandboxShockPlan:
    depeg_target: str  # "a" or "b" -- which of the sandbox's two CurrencyConfigs gets the H4 pair
    depeg_gap_days: int  # 0/5/10/20, matching master_simulation.yaml's own H4 sweep gaps
    additional_shock_type: ShockType
    additional_shock_target: str  # "a" or "b"
    additional_shock_magnitude: float


# Judgment calls, one per sandbox (documented individually -- see each
# comment for the reasoning behind which option gets which shock):
_SANDBOX_SHOCK_PLANS: dict[str, _SandboxShockPlan] = {
    # option_a=SBX1_HILIQ_LOGOV (governance 0.55, the weaker-governance
    # side), option_b=SBX1_HIGOV_LOLIQ (governance 0.95, liquidity 0.90).
    # Depeg target: option_a, the lower-governance/non-compliant side.
    # Additional shock: liquidity_crunch on option_b, this pair's own
    # lower-liquidity side -- stresses the sandbox's OTHER isolated
    # dimension too.
    "liquidity_vs_governance": _SandboxShockPlan(
        depeg_target="a",
        depeg_gap_days=0,
        additional_shock_type=ShockType.LIQUIDITY_CRUNCH,
        additional_shock_target="b",
        additional_shock_magnitude=0.20,
    ),
    # option_a=SBX2_HIGOV_LOSTAB (governance 0.95), option_b=SBX2_LOGOV_HISTAB
    # (governance 0.55, non-compliant). Depeg target: option_b, the
    # lower-governance side. Additional shock: governance_downgrade on the
    # same symbol, reinforcing direct stress on this sandbox's governance
    # axis (and giving a clean `apply_currency_shock`-mutation test case).
    "governance_vs_stability": _SandboxShockPlan(
        depeg_target="b",
        depeg_gap_days=5,
        additional_shock_type=ShockType.GOVERNANCE_DOWNGRADE,
        additional_shock_target="b",
        additional_shock_magnitude=0.15,
    ),
    # option_a=SBX3_HILIQ_LOSTAB (peg_error 0.04, the worse-stability side),
    # option_b=SBX3_LOLIQ_HISTAB (liquidity 0.75, the lower-liquidity side).
    # governance_score is held constant in this pair, so depeg targets the
    # weaker-STABILITY side (option_a) instead. Additional shock:
    # liquidity_crunch on option_b, the lower-liquidity side.
    "liquidity_vs_stability": _SandboxShockPlan(
        depeg_target="a",
        depeg_gap_days=10,
        additional_shock_type=ShockType.LIQUIDITY_CRUNCH,
        additional_shock_target="b",
        additional_shock_magnitude=0.20,
    ),
    # option_a=SBX4_GOLD_LOLIQ (the GOLD-BACKED option, liquidity 0.70),
    # option_b=SBX4_STABLE_HILIQ (liquidity 0.99). governance_score/peg_error
    # are held constant. Depeg target: option_b, the NON-gold option --
    # this sandbox is one of the two (with asset_backing_vs_stability) that
    # actually contains a gold-backed currency, so it's where H4 (crisis
    # proximity -> gold preference) is meaningfully testable: the
    # crisis_warning/depeg_event pair must stress the currency agents would
    # flee FROM, not the gold option they're predicted to flee TOWARD.
    # Targeting option_a (gold itself) would put the crisis on gold and make
    # a flight-to-gold response unmeasurable -- there'd be no non-gold
    # currency left to flee. Additional shock: liquidity_crunch on option_a
    # (gold, the lower-liquidity side), directly stressing this sandbox's
    # isolated liquidity dimension.
    "asset_backing_vs_liquidity": _SandboxShockPlan(
        depeg_target="b",
        depeg_gap_days=20,
        additional_shock_type=ShockType.LIQUIDITY_CRUNCH,
        additional_shock_target="a",
        additional_shock_magnitude=0.20,
    ),
    # option_a=SBX5_GOLD_LOSTAB (the GOLD-BACKED option, peg_error 0.015, the
    # worse-stability side), option_b=SBX5_DEPOSIT_HISTAB (a bank-issued
    # tokenized deposit). governance_score/liquidity_score are held
    # constant. Depeg target: option_b, the NON-gold option -- the other of
    # the two sandboxes where H4 is meaningfully testable (see
    # asset_backing_vs_liquidity above for the full reasoning): the
    # crisis/depeg pair must stress the non-gold deposit token so a shift
    # toward option_a (gold) is an observable flight-to-gold response, not
    # target gold itself and make that response unmeasurable. Additional
    # shock: regulatory_enforcement on option_b (the same non-gold deposit
    # token) -- thematically apt for a bank-issued deposit, and (like
    # governance_downgrade) `apply_currency_shock` actually mutates
    # issuer_risk for this shock type, giving a clean end-to-end mutation
    # test.
    "asset_backing_vs_stability": _SandboxShockPlan(
        depeg_target="b",
        depeg_gap_days=0,
        additional_shock_type=ShockType.REGULATORY_ENFORCEMENT,
        additional_shock_target="b",
        additional_shock_magnitude=0.20,
    ),
    # option_a=SBX6_DEPOSIT_BANKRISK (governance 0.75 but issuer_risk 0.25,
    # the higher-issuer-risk side), option_b=SBX6_STABLE_ALGO (governance
    # 0.70, the lower-governance side but issuer_risk only 0.20). This
    # pair's governance_score and issuer_risk are inversely ordered, so
    # depeg targets the higher-ISSUER-RISK side (option_a) -- issuer/
    # counterparty risk is the more direct driver of depeg fragility.
    # Additional shock: governance_downgrade on option_b, the
    # lower-governance side, isolating this sandbox's other named
    # dimension.
    "asset_backing_vs_governance": _SandboxShockPlan(
        depeg_target="a",
        depeg_gap_days=10,
        additional_shock_type=ShockType.GOVERNANCE_DOWNGRADE,
        additional_shock_target="b",
        additional_shock_magnitude=0.15,
    ),
}


def build_sandbox_scenario(
    sandbox_key: str,
    option_a: CurrencyConfig,
    option_b: CurrencyConfig,
    base_scenario: ScenarioConfig,
) -> ScenarioConfig:
    """Build a sandbox-specific `ScenarioConfig`: `base_scenario`'s
    `duration_days`/`initial_state` and its 5 macro-level shocks (no
    `target_currency`), plus this sandbox's own H4 crisis-proximity pair and
    one additional currency-targeted shock -- both targeting one of
    `option_a`/`option_b`'s actual symbols, per `_SANDBOX_SHOCK_PLANS`.

    `sandbox_key` must be one of `SANDBOX_CURRENCY_PAIRS`'s keys (raises
    `KeyError` otherwise, via the `_SANDBOX_SHOCK_PLANS` lookup).
    """
    plan = _SANDBOX_SHOCK_PLANS[sandbox_key]
    symbol_by_side = {"a": option_a.symbol, "b": option_b.symbol}

    macro_shocks = [shock for shock in base_scenario.shocks if shock.target_currency is None]

    depeg_symbol = symbol_by_side[plan.depeg_target]
    crisis_pair = [
        ShockEvent(
            day=_CRISIS_WARNING_DAY,
            type=ShockType.CRISIS_WARNING,
            magnitude=_CRISIS_WARNING_MAGNITUDE,
            target_currency=depeg_symbol,
        ),
        ShockEvent(
            day=_CRISIS_WARNING_DAY + plan.depeg_gap_days,
            type=ShockType.DEPEG_EVENT,
            magnitude=_DEPEG_MAGNITUDE,
            target_currency=depeg_symbol,
        ),
    ]

    additional_symbol = symbol_by_side[plan.additional_shock_target]
    additional_shock = ShockEvent(
        day=_ADDITIONAL_SHOCK_DAY,
        type=plan.additional_shock_type,
        magnitude=plan.additional_shock_magnitude,
        target_currency=additional_symbol,
    )

    return base_scenario.model_copy(
        update={
            "name": f"{sandbox_key}_sandbox",
            "shocks": [*macro_shocks, *crisis_pair, additional_shock],
        },
        deep=True,
    )
