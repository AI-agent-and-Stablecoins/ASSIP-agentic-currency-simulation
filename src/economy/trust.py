"""Dynamic per-currency trust ledger.

governance_score in configs/currencies/*.yaml is a static structural prior
-- it never changes during a run. trust_score is the missing dynamic
counterpart: it starts at governance_score and moves with lived simulation
experience (decaying fast on a shock, recovering slowly on quiet days),
per docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md
Sec 3.2. peg_error_offset/liquidity_offset (Task 4) reuse the identical
mechanism for temporary shock effects -- see that task for why one ledger
tracks three quantities instead of three separate mechanisms.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from src.currencies.currency import CurrencyConfig
from src.economy.shocks import ShockEvent, ShockType
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT
from src.utils.helpers import clamp

TRUST_PARAMS_PATH = CONFIG_ROOT / "economy" / "trust_params.yaml"


class TrustParams(BaseModel):
    lambda_shock: float
    lambda_recover: float
    lambda_contagion: float
    rolling_window_days: int


def load_trust_params(path: Path = TRUST_PARAMS_PATH) -> TrustParams:
    return load_yaml_as(path, TrustParams)


class _CurrencyLedgerState(BaseModel):
    trust_score: float
    trust_history: list[float] = Field(default_factory=list)
    peg_error_offset: float = 0.0
    liquidity_offset: float = 0.0


class TrustLedger:
    _PEG_OFFSET_SHOCKS = {ShockType.DEPEG_EVENT, ShockType.FX_VOLATILITY_SHOCK}
    _LIQUIDITY_OFFSET_SHOCKS = {ShockType.LIQUIDITY_CRUNCH, ShockType.REGULATORY_ENFORCEMENT, ShockType.CAPITAL_CONTROLS}

    def __init__(self, currencies: dict[str, CurrencyConfig], params: TrustParams):
        self._params = params
        self._asset_class_of = {symbol: cfg.asset_class for symbol, cfg in currencies.items()}
        self._baseline_governance = {symbol: cfg.governance_score for symbol, cfg in currencies.items()}
        self._state: dict[str, _CurrencyLedgerState] = {
            symbol: _CurrencyLedgerState(trust_score=cfg.governance_score, trust_history=[cfg.governance_score])
            for symbol, cfg in currencies.items()
        }

    def trust_score(self, symbol: str) -> float:
        return self._state[symbol].trust_score

    def history(self, symbol: str, days: int) -> list[float]:
        return self._state[symbol].trust_history[-days:]

    def peg_error_offset(self, symbol: str) -> float:
        return self._state[symbol].peg_error_offset

    def liquidity_offset(self, symbol: str) -> float:
        return self._state[symbol].liquidity_offset

    def effective_peg_error(self, symbol: str, baseline: float) -> float:
        return max(0.0, baseline + self.peg_error_offset(symbol))

    def effective_liquidity_score(self, symbol: str, baseline: float) -> float:
        return clamp(baseline + self.liquidity_offset(symbol), 0.0, 1.0)

    def update(
        self,
        fired_shocks: list[ShockEvent],
        currencies: dict[str, CurrencyConfig] | None = None,
    ) -> None:
        if currencies is not None:
            # A permanent structural mutation (e.g. governance_downgrade via
            # apply_currency_shock) must be reflected in the recovery
            # baseline immediately, or a currency would "self-heal" on quiet
            # days back toward its stale, higher original governance_score.
            for symbol, cfg in currencies.items():
                self._baseline_governance[symbol] = cfg.governance_score

        severity_by_currency: dict[str, float] = {}
        peg_shock_by_currency: dict[str, float] = {}
        liquidity_shock_by_currency: dict[str, float] = {}

        for shock in fired_shocks:
            if shock.target_currency is None:
                continue
            severity = min(1.0, shock.magnitude)
            severity_by_currency[shock.target_currency] = max(
                severity_by_currency.get(shock.target_currency, 0.0), severity
            )
            if shock.type in self._PEG_OFFSET_SHOCKS:
                peg_shock_by_currency[shock.target_currency] = (
                    peg_shock_by_currency.get(shock.target_currency, 0.0) + shock.magnitude
                )
            if shock.type in self._LIQUIDITY_OFFSET_SHOCKS:
                liquidity_shock_by_currency[shock.target_currency] = (
                    liquidity_shock_by_currency.get(shock.target_currency, 0.0) - abs(shock.magnitude)
                )

        for symbol, state in self._state.items():
            severity = severity_by_currency.get(symbol)
            if severity is not None:
                state.trust_score = max(0.0, state.trust_score - self._params.lambda_shock * severity * state.trust_score)
            else:
                contagion_severity = 0.0
                for other, other_severity in severity_by_currency.items():
                    if other != symbol and self._asset_class_of.get(other) == self._asset_class_of.get(symbol):
                        contagion_severity = max(contagion_severity, other_severity)
                if contagion_severity > 0:
                    state.trust_score = max(
                        0.0, state.trust_score - self._params.lambda_contagion * contagion_severity * state.trust_score
                    )
                else:
                    baseline = self._baseline_governance[symbol]
                    state.trust_score = state.trust_score + self._params.lambda_recover * (baseline - state.trust_score)

            state.trust_history.append(state.trust_score)
            max_history = self._params.rolling_window_days * 3
            if len(state.trust_history) > max_history:
                state.trust_history = state.trust_history[-max_history:]

            state.peg_error_offset += (
                -self._params.lambda_recover * state.peg_error_offset + peg_shock_by_currency.get(symbol, 0.0)
            )
            state.liquidity_offset += (
                -self._params.lambda_recover * state.liquidity_offset + liquidity_shock_by_currency.get(symbol, 0.0)
            )
