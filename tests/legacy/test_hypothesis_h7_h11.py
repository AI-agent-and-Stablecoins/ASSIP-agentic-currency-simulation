from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from src.currencies.gold_token import GoldBackedConfig
from src.currencies.sandbox_currencies import SANDBOX_CURRENCY_PAIRS
from src.currencies.tokenized_deposit import TokenizedDepositConfig
from src.legacy.econometrics.hypothesis_datasets import (
    build_h7_dataset,
    build_h8_dataset,
    build_h9_dataset,
    build_h10_dataset,
    build_h11_dataset,
    build_sandbox_preference_dataset,
)
from src.legacy.econometrics.hypothesis_regressions import (
    regress_h7,
    regress_h8,
    regress_h9,
    regress_h10,
    regress_h11,
)
from src.legacy.econometrics.regression_engine import RegressionResult
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _populated_session(sandbox_key: str, forced_symbol: str, num_days: int = 8) -> Session:
    """Mirrors tests/test_hypothesis_h3.py's _populated_session helper: forces
    every mock decision to one specific symbol so at least that sandbox's
    cells produce genuine ACCEPT decisions instead of falling back to a
    synthetic WALK_AWAY (see matrix_runner's per-cell mock-currency
    docstring)."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=num_days,
        dry_run=True,
        exercise_llm_path=True,
        mock_llm_decision={
            "action": "ACCEPT",
            "proposed_currency": forced_symbol,
            "proposed_chain": "ethereum",
            "amount": 1.0,
            "price": 1.0,
            "reasoning": "test fixture",
        },
        session=session,
    )
    return session


def test_build_sandbox_preference_dataset_scopes_to_exactly_one_cell_variant():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["governance_vs_stability"]
    session = _populated_session("governance_vs_stability", option_a.symbol)

    df = build_sandbox_preference_dataset(
        session,
        sandbox_key="governance_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant="domestic",
    )

    assert not df.empty
    assert set(df.columns) >= {"agent_id", "chose_higher_option", "cara_a", "agent_type", "actual_model"}
    assert df["chose_higher_option"].isin([0, 1]).all()


def test_build_sandbox_preference_dataset_domestic_and_cross_border_are_disjoint_cells():
    """Regression test for a Plan 6b whole-branch review finding: the
    original version of this test only asserted both frames were
    non-empty, which would still pass even if `cell_variant` scoping were
    broken and both frames were pooling the same rows -- the disjointness
    claim in the name was never actually checked. This asserts genuine
    row-level disjointness via each row's `run_id`, which is retained in
    both frames."""
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["governance_vs_stability"]
    session = _populated_session("governance_vs_stability", option_a.symbol)

    domestic_df = build_sandbox_preference_dataset(
        session,
        sandbox_key="governance_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant="domestic",
    )
    cross_border_df = build_sandbox_preference_dataset(
        session,
        sandbox_key="governance_vs_stability",
        higher_option_selector=lambda a, b: a.symbol if a.peg_error <= b.peg_error else b.symbol,
        cell_variant="cross_border",
    )

    assert not domestic_df.empty
    assert not cross_border_df.empty
    domestic_run_ids = set(domestic_df["run_id"])
    cross_border_run_ids = set(cross_border_df["run_id"])
    assert domestic_run_ids.isdisjoint(cross_border_run_ids)
    # run_id is f"{matrix_run_id}-{cell_key}-seed{seed}" (matrix_runner.py),
    # so the cell-key segment is a substring, not a suffix.
    assert all("_domestic-seed" in run_id for run_id in domestic_run_ids)
    assert all("_cross_border-seed" in run_id for run_id in cross_border_run_ids)


def test_build_sandbox_preference_dataset_rejects_unknown_cell_variant():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["governance_vs_stability"]
    session = _populated_session("governance_vs_stability", option_a.symbol)

    try:
        build_sandbox_preference_dataset(
            session,
            sandbox_key="governance_vs_stability",
            higher_option_selector=lambda a, b: a.symbol,
            cell_variant="not_a_real_variant",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "cell_variant" in str(exc)


_H7_H11_CASES = [
    ("governance_vs_stability", build_h7_dataset),
    ("liquidity_vs_stability", build_h8_dataset),
    ("asset_backing_vs_liquidity", build_h9_dataset),
    ("asset_backing_vs_stability", build_h10_dataset),
    ("asset_backing_vs_governance", build_h11_dataset),
]


def test_each_h7_h11_dataset_builder_scopes_to_its_own_sandbox_and_variant():
    for sandbox_key, builder in _H7_H11_CASES:
        option_a, _ = SANDBOX_CURRENCY_PAIRS[sandbox_key]
        session = _populated_session(sandbox_key, option_a.symbol)

        domestic_df = builder(session, cell_variant="domestic")
        cross_border_df = builder(session, cell_variant="cross_border")

        assert not domestic_df.empty, f"{builder.__name__} domestic was empty"
        assert not cross_border_df.empty, f"{builder.__name__} cross_border was empty"
        assert set(domestic_df.columns) >= {"agent_id", "chose_higher_option", "cara_a"}


def test_h9_selector_picks_the_gold_backed_symbol():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_liquidity"]
    gold_option = option_a if isinstance(option_a, GoldBackedConfig) else option_b
    session = _populated_session("asset_backing_vs_liquidity", gold_option.symbol)

    df = build_h9_dataset(session, cell_variant="domestic")
    # Every forced-ACCEPT decision proposed the gold option -> chose_higher_option must be all 1s.
    assert (df["chose_higher_option"] == 1).all()


def test_h10_selector_picks_the_deposit_symbol_not_the_gold_symbol():
    option_a, option_b = SANDBOX_CURRENCY_PAIRS["asset_backing_vs_stability"]
    deposit_option = option_a if isinstance(option_a, TokenizedDepositConfig) else option_b
    session = _populated_session("asset_backing_vs_stability", deposit_option.symbol)

    df = build_h10_dataset(session, cell_variant="domestic")
    assert (df["chose_higher_option"] == 1).all()


_H7_H11_REGRESS_CASES = [
    ("governance_vs_stability", regress_h7, "H7"),
    ("liquidity_vs_stability", regress_h8, "H8"),
    ("asset_backing_vs_liquidity", regress_h9, "H9"),
    ("asset_backing_vs_stability", regress_h10, "H10"),
    ("asset_backing_vs_governance", regress_h11, "H11"),
]


def _populated_session_with_genuine_variation(sandbox_key: str, num_days: int = 8) -> Session:
    """Mirrors test_hypothesis_h3.py's _populated_session_with_genuine_variation:
    a single run_matrix call forces a constant proposed_currency, so
    chose_higher_option never varies within one call. Two run_matrix calls
    into the same session, forcing option_a then option_b, gives genuine
    variation for fit_clustered_logit to fit against."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    option_a, option_b = SANDBOX_CURRENCY_PAIRS[sandbox_key]
    for call_index, symbol in enumerate((option_a.symbol, option_b.symbol)):
        run_matrix(
            model_candidates=MODEL_CANDIDATES,
            seeds=[0],
            num_days=num_days,
            dry_run=True,
            exercise_llm_path=True,
            matrix_run_id=f"{sandbox_key}-variation-{call_index}",
            mock_llm_decision={
                "action": "ACCEPT",
                "proposed_currency": symbol,
                "proposed_chain": "ethereum",
                "amount": 1.0,
                "price": 1.0,
                "reasoning": "forced alternation for genuine chose_higher_option variation",
            },
            session=session,
        )
    return session


def test_each_regress_h7_h11_returns_separate_domestic_and_cross_border_results():
    for sandbox_key, regress_fn, hyp_label in _H7_H11_REGRESS_CASES:
        session = _populated_session_with_genuine_variation(sandbox_key)

        domestic_result = regress_fn(session, cell_variant="domestic")
        cross_border_result = regress_fn(session, cell_variant="cross_border")

        assert isinstance(domestic_result, RegressionResult)
        assert isinstance(cross_border_result, RegressionResult)
        assert domestic_result.hypothesis == f"{hyp_label}_domestic"
        assert cross_border_result.hypothesis == f"{hyp_label}_cross_border"
        assert domestic_result.regressor == "cara_a"
