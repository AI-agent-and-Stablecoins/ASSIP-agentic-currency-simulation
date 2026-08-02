"""Loss-driven CARA-coefficient adaptation (Phase 3 Plan 4, Task 7).

Agents holding a CARA (or risk-neutral-nominal-CARA) utility function become
more risk-averse after realizing a loss in real (inflation-adjusted)
purchasing power: a_next = min(a_max, a + eta_risk * Loss_t / W_real_t),
where Loss_t = max(0, W_real_before - W_real_after). Gains never reduce a.

Only the 55 CARA-eligible agents (consumer/bank/investor -- built with a
nominal `cara_coefficient`, whether currently "cara" or "risk_neutral")
adapt. merchant/institution agents (`multi_attribute`, `cara_coefficient is
None`) are untouched -- this is a no-op for them.

Follows the same load_yaml_as/CONFIG_ROOT pattern as Plan 2's trust.py and
Task 6's fx_tax.py.
"""

from pathlib import Path

from pydantic import BaseModel

from src.agents.agent_factory import cara_utility_fields
from src.agents.base_agent import BaseAgent
from src.utility.utility_factory import build_utility_function
from src.utils.config_loader import load_yaml_as
from src.utils.constants import CONFIG_ROOT

RISK_ADAPTATION_PARAMS_PATH = CONFIG_ROOT / "economy" / "risk_adaptation_params.yaml"


class RiskAdaptationParams(BaseModel):
    eta_risk: float
    a_max: float


def load_risk_adaptation_params(path: Path = RISK_ADAPTATION_PARAMS_PATH) -> RiskAdaptationParams:
    return load_yaml_as(path, RiskAdaptationParams)


def adapt_cara_coefficient(
    agent: BaseAgent, w_real_before: float, w_real_after: float, params: RiskAdaptationParams
) -> None:
    if agent.cara_coefficient is None:
        return

    loss = max(0.0, w_real_before - w_real_after)

    if w_real_after <= 0.0:
        # A wallet can legitimately be drained to exactly 0.0 nominal value,
        # which deflates to 0.0 real purchasing power -- Loss_t / W_real_t is
        # then undefined (0/0, no loss this step) or infinite (loss > 0).
        # Rather than dividing by zero, treat a realized loss into a fully
        # drained wallet as the maximal possible loss ratio (clamp to
        # a_max); with no loss, leave the coefficient unchanged.
        a_next = params.a_max if loss > 0.0 else agent.cara_coefficient
    else:
        a_next = min(params.a_max, agent.cara_coefficient + params.eta_risk * loss / w_real_after)

    agent.cara_coefficient = a_next
    agent.utility_type, agent.risk_aversion = cara_utility_fields(a_next)
    agent.utility_fn = build_utility_function(
        agent.utility_type, agent.risk_aversion, agent.multi_attribute_weights, agent.eis
    )
