"""Daily stochastic EUR/USD dynamics.

Real currency pairs move day-to-day even absent a deliberate scheduled
shock (`ShockType.FX_RATE_SHOCK`/`fx_volatility_shock`). Modeled as a
simple daily percentage return, `Normal(0, daily_volatility)`, applied to
`MacroState.peg_reference_rates["EUR"]` -- the REAL accounting rate (used
by `ExchangeRateTable` for every currency conversion, not just a cosmetic
log value), per the user's decision that H5's volatility regressor should
reflect what agents actually experienced/reacted to, not a value decoupled
from real settlement.

Deliberately seeded from `day` alone (`random.Random(day)`), NOT from
`run_timestep`'s own `rng: random.Random` parameter: `rng` is already
consumed downstream by `agent_activation_order`, and every existing
deterministic test built against a fixed `random.Random(seed)` sequence
would shift unpredictably if this drew from that same stream first. Using
`day` as the sole seed makes the EUR/USD path identical across every seed
of the same scenario -- a deliberate simplification, consistent with how
the shock SCHEDULE itself is already fixed per scenario regardless of
seed (only agent behavior/model assignment varies by seed).
"""

import random

from src.economy.macro_state import MacroState

DEFAULT_DAILY_VOLATILITY = 0.004  # ~0.4% daily std -- roughly real-world EUR/USD historical daily volatility


def advance_eur_usd_rate(
    state: MacroState, day: int, daily_volatility: float = DEFAULT_DAILY_VOLATILITY
) -> MacroState:
    """Returns a new `MacroState` with `peg_reference_rates["EUR"]`
    advanced by one day's stochastic return. Applied every day regardless
    of whether a scheduled `fx_rate_shock` also fires that day -- this
    models ambient day-to-day noise; shocks model deliberate discrete
    events, and the two compose multiplicatively (whichever order they're
    applied in that day)."""
    updated = state.model_copy(deep=True)
    daily_return = random.Random(day).gauss(0.0, daily_volatility)
    updated.peg_reference_rates["EUR"] = updated.peg_reference_rates["EUR"] * (1 + daily_return)
    return updated
