"""Streamlit dashboard for the Phase 3 experiment matrix.

Run with: streamlit run dashboard/app.py

Reads live progress directly from the project's own database
(dashboard/queries.py) and control-plane state from a small status file
(dashboard/status_store.py); starts/stops the actual simulation as a
separate OS process (dashboard/process_control.py) that survives this
Streamlit script's own rerun-on-every-interaction execution model.
"""

import streamlit as st
from sqlalchemy.exc import OperationalError

from dashboard import status_store
from dashboard.process_control import RunConfig, is_alive, resume, start, stop
from dashboard.queries import get_progress_for_run
from database.session import new_session
from src.simulation.matrix_runner import _build_cell_specs
from src.utils.helpers import generate_id

st.set_page_config(page_title="Phase 3 Matrix Dashboard", layout="wide")
st.title("Phase 3 Experiment Matrix Dashboard")

ALL_CELL_KEYS = [spec.key for spec in _build_cell_specs()]

if "matrix_run_id" not in st.session_state:
    st.session_state.matrix_run_id = status_store.get_active_run_id() or generate_id("matrix")

st.header("Run configuration")
matrix_run_id = st.text_input("matrix_run_id", value=st.session_state.matrix_run_id)
st.session_state.matrix_run_id = matrix_run_id

col1, col2 = st.columns(2)
with col1:
    cell_scope = st.radio("Cell scope", ["All 13 cells", "Specific cells"])
    selected_cells = None
    if cell_scope == "Specific cells":
        selected_cells = st.multiselect("Cells", ALL_CELL_KEYS)
    seeds_text = st.text_input("Seeds (comma-separated)", value="0")
    num_days = st.number_input("Days", min_value=1, value=1, step=1)
with col2:
    distributed = st.checkbox("Distributed (multiple processes)", value=False)
    num_processes = st.number_input("num_processes", min_value=1, value=4, step=1, disabled=not distributed)
    dry_run = st.checkbox("Dry run (no real spend)", value=True)


def _build_config() -> RunConfig:
    return RunConfig(
        matrix_run_id=matrix_run_id,
        cell_keys=selected_cells if cell_scope == "Specific cells" else None,
        seeds=[int(s.strip()) for s in seeds_text.split(",") if s.strip()],
        num_days=int(num_days),
        dry_run=dry_run,
        distributed=distributed,
        num_processes=int(num_processes),
    )


try:
    status = status_store.read_status(matrix_run_id)
    status_read_error = False
except status_store.StatusFileCorruptedError:
    status = None
    status_read_error = True

st.header("Controls")
control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
    if dry_run:
        if st.button("Start"):
            start(_build_config())
            status_store.set_active_run_id(matrix_run_id)
            st.rerun()
    else:
        st.warning("Real launch selected -- type the matrix_run_id below to confirm.")
        confirmation = st.text_input("Type the matrix_run_id above to confirm a REAL (paid) launch")
        if st.button("Confirm real launch"):
            if confirmation == matrix_run_id:
                start(_build_config())
                status_store.set_active_run_id(matrix_run_id)
                st.rerun()
            else:
                st.error("Typed text does not match matrix_run_id -- launch not started.")

with control_col2:
    if st.button("Stop"):
        stop(matrix_run_id)
        st.rerun()

with control_col3:
    if st.button("Pause"):
        st.info("Pause behaves the same as Stop -- there is no mid-execution suspend. Safe to Resume afterward.")
        stop(matrix_run_id)
        st.rerun()

with control_col4:
    resume_disabled = status is None or status.get("state") not in ("stopped", "failed")
    if st.button("Resume", disabled=resume_disabled):
        resume(matrix_run_id)
        st.rerun()

st.header("Status")
if status_read_error:
    st.error("Status file for this matrix_run_id is unreadable. Live progress below still works from the database.")

if status is None:
    st.info("No run has been started under this matrix_run_id yet.")
else:
    pid = status.get("pid")
    reported_state = status.get("state", "unknown")
    if reported_state == "running" and pid is not None and not is_alive(pid):
        st.error(f"Process (pid {pid}) is no longer running but never reported completion -- likely crashed.")
    else:
        st.write(f"State: **{reported_state}**")

    usage = status.get("cumulative_usage")
    if usage:
        u1, u2, u3 = st.columns(3)
        u1.metric("Prompt tokens", usage.get("prompt_tokens", 0))
        u2.metric("Completion tokens", usage.get("completion_tokens", 0))
        u3.metric("Total tokens", usage.get("total_tokens", 0))

    failures = status.get("failures")
    if failures:
        st.subheader("Failures")
        st.table(failures)
    elif status.get("state") == "running":
        st.caption("Failures are only known once each cell/seed's run completes -- not live mid-run.")

    error = status.get("error")
    if error:
        st.error(f"Error: {error}")

st.header("Live progress")
session = new_session()
try:
    progress_rows = get_progress_for_run(session, matrix_run_id)
except OperationalError:
    st.warning(
        "Database schema not initialized yet -- no tables exist. Start a run to initialize it, "
        "or run database.session.create_all_tables() directly."
    )
    progress_rows = []
if not progress_rows:
    st.info("No progress recorded yet for this matrix_run_id.")
else:
    st.dataframe(
        [
            {
                "Cell": row.cell_key,
                "Seed": row.seed,
                "Day": row.current_day,
                "LLM decisions": row.total_llm_decisions,
            }
            for row in progress_rows
        ],
        use_container_width=True,
    )
