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
