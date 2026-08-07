from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base
from dashboard.queries import get_progress_for_run
from src.simulation.matrix_runner import run_matrix

MODEL_CANDIDATES = ["vendor/fake-model"]


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_get_progress_for_run_reports_current_day_and_decision_count_per_cell_seed():
    session = _session()
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=3,
        dry_run=True,
        session=session,
        matrix_run_id="progress-test",
        cell_keys=["master"],
    )

    progress = get_progress_for_run(session, "progress-test")

    assert len(progress) == 1
    row = progress[0]
    assert row.cell_key == "master"
    assert row.seed == 0
    assert row.run_id == "progress-test-master-seed0"
    assert row.current_day == 2  # 3 days completed -> timesteps 0, 1, 2
    assert row.total_llm_decisions == 0  # dry_run without exercise_llm_path never calls the LLM router


def test_get_progress_for_run_returns_empty_list_for_an_unknown_matrix_run_id():
    session = _session()
    progress = get_progress_for_run(session, "no-such-matrix-run")
    assert progress == []


def test_get_progress_for_run_only_includes_this_matrix_run_ids_rows():
    session = _session()
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        session=session,
        matrix_run_id="run-a",
        cell_keys=["master"],
    )
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        session=session,
        matrix_run_id="run-b",
        cell_keys=["master"],
    )

    progress_a = get_progress_for_run(session, "run-a")
    assert len(progress_a) == 1
    assert progress_a[0].run_id == "run-a-master-seed0"


def test_get_progress_for_run_escapes_like_wildcards():
    """Verify that % and _ characters in matrix_run_id are treated as literals,
    not SQL LIKE wildcards. Without escaping, a matrix_run_id like 'test_%run'
    would match 'test-arun' due to _ being a LIKE wildcard."""
    session = _session()
    # Create a run with underscore in its ID
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        session=session,
        matrix_run_id="test_%run",
        cell_keys=["master"],
    )
    # Create another run whose ID would match the naive wildcard interpretation
    # (test_ matches "test_", then % matches any characters "a", then run matches "run")
    run_matrix(
        model_candidates=MODEL_CANDIDATES,
        seeds=[0],
        num_days=1,
        dry_run=True,
        session=session,
        matrix_run_id="test-arun",
        cell_keys=["master"],
    )

    # Query for the run with wildcards in its name
    progress_with_wildcard = get_progress_for_run(session, "test_%run")
    # Should only get 1 result from the first run, not both
    assert len(progress_with_wildcard) == 1
    assert progress_with_wildcard[0].run_id == "test_%run-master-seed0"

    # Query for the second run to confirm it was created
    progress_second = get_progress_for_run(session, "test-arun")
    assert len(progress_second) == 1
    assert progress_second[0].run_id == "test-arun-master-seed0"
