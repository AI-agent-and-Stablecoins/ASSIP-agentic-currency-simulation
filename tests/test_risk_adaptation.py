import pytest

from src.agents.agent_factory import build_agent, load_agent_profiles
from src.economy.risk_adaptation import RiskAdaptationParams, adapt_cara_coefficient, load_risk_adaptation_params
from src.utility.cara import CARAUtility
from src.utility.risk_neutral import RiskNeutralUtility


def _params() -> RiskAdaptationParams:
    return RiskAdaptationParams(eta_risk=1.0, a_max=5.0)


def test_load_risk_adaptation_params_reads_the_real_config():
    params = load_risk_adaptation_params()
    assert params.eta_risk == 1.0
    assert params.a_max == 5.0


def test_adapt_cara_coefficient_is_a_noop_for_non_cara_eligible_agents():
    profile = load_agent_profiles()["merchant"]
    agent = build_agent(profile)  # cara_coefficient is None
    original_utility_fn = agent.utility_fn

    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=800.0, params=_params())

    assert agent.cara_coefficient is None
    assert agent.utility_fn is original_utility_fn


def test_adapt_cara_coefficient_increases_a_after_a_realized_loss():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, cara_override=("cara", 1.0))

    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=800.0, params=_params())
    # Loss_t = 200, W_real_t = 800, eta_risk=1.0 -> a_next = 1.0 + 1.0 * 200/800 = 1.25

    assert agent.cara_coefficient == pytest.approx(1.25)
    assert agent.risk_aversion == pytest.approx(1.25)
    assert isinstance(agent.utility_fn, CARAUtility)


def test_adapt_cara_coefficient_never_decreases_on_a_gain():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, cara_override=("cara", 1.0))

    adapt_cara_coefficient(agent, w_real_before=800.0, w_real_after=1000.0, params=_params())

    assert agent.cara_coefficient == pytest.approx(1.0)  # unchanged, gains don't reduce a


def test_adapt_cara_coefficient_clamps_at_a_max():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, cara_override=("cara", 4.9))

    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=1.0, params=_params())
    # Loss_t = 999, W_real_t = 1.0 -> raw a_next would be huge; must clamp at a_max=5.0

    assert agent.cara_coefficient == pytest.approx(5.0)


def test_adapt_cara_coefficient_switches_to_cara_when_a_crosses_above_zero():
    profile = load_agent_profiles()["bank"]
    agent = build_agent(profile, cara_override=("risk_neutral", None))  # nominal a starts at 0.0

    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=500.0, params=_params())
    # Loss_t = 500, W_real_t = 500, eta_risk=1.0 -> a_next = 0.0 + 1.0*500/500 = 1.0 (crosses above 0)

    assert agent.cara_coefficient == pytest.approx(1.0)
    assert agent.utility_type == "cara"
    assert isinstance(agent.utility_fn, CARAUtility)


def test_adapt_cara_coefficient_guards_against_zero_w_real_after():
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, cara_override=("cara", 1.0))

    # A catastrophic shock could plausibly drive real wealth to exactly zero;
    # dividing by w_real_after must not raise ZeroDivisionError.
    adapt_cara_coefficient(agent, w_real_before=1000.0, w_real_after=0.0, params=_params())

    assert agent.cara_coefficient == pytest.approx(5.0)  # clamped at a_max
