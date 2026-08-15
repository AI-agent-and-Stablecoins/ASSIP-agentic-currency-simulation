"""Generates New info.pdf's paper-style tables (H1's equilibrium-holdings
table; H2-H11's compensation tables) from a completed run_hypothesis_matrix
run, for every cell that has data. Prints each table to the console and
writes it to reports/<matrix_run_id>/<track>/<cell_key>.csv.

Usage:
    DATABASE_URL="sqlite:///./assip.db" .venv/bin/python \
        scripts/generate_hypothesis_report.py <matrix_run_id> [seed] [track]

`track` is "real" (default) or "synthetic" -- must match whichever
`run_hypothesis_matrix(track=...)` produced the run being reported on.
"""

import sys

from database.session import new_session
from src.economy.hypothesis_scenarios import build_hypothesis_cell_specs
from src.economy.synthetic_hypothesis_scenarios import build_synthetic_hypothesis_cell_specs
from src.reporting.hypothesis_tables import build_compensation_tables, build_equilibrium_holdings_table
from src.utils.constants import REPO_ROOT


def _is_empty(table) -> bool:
    return bool((table == "").all().all())


def _report_holdings(session, matrix_run_id, spec, seed, track, out_dir):
    table = build_equilibrium_holdings_table(session, matrix_run_id, cell_key=spec.key, seed=seed, track=track)
    if _is_empty(table):
        return 0
    path = out_dir / f"{spec.key}.csv"
    table.to_csv(path)
    print(f"\n=== {spec.key} (equilibrium holdings) ===")
    print(table.to_string())
    print(f"-> {path}")
    return 1


def _report_compensation(session, matrix_run_id, spec, seed, track, out_dir):
    tables = build_compensation_tables(
        session, matrix_run_id, spec.hypothesis, cell_key=spec.key, seed=seed, track=track
    )
    written = 0
    for title, table in tables.items():
        if _is_empty(table):
            continue
        safe_title = title.replace(" ", "_").replace("/", "-")
        path = out_dir / f"{spec.key}__{safe_title}.csv"
        table.to_csv(path)
        print(f"\n=== {spec.key}: {title} ===")
        print(table.to_string())
        print(f"-> {path}")
        written += 1
    return written


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: generate_hypothesis_report.py <matrix_run_id> [seed] [track]")
    matrix_run_id = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    track = sys.argv[3] if len(sys.argv) > 3 else "real"
    if track not in ("real", "synthetic"):
        raise SystemExit(f"track must be 'real' or 'synthetic', got {track!r}")

    session = new_session()
    out_dir = REPO_ROOT / "reports" / matrix_run_id / track
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    if track == "real":
        for spec in build_hypothesis_cell_specs():
            if spec.hypothesis == "H1":
                written += _report_holdings(session, matrix_run_id, spec, seed, track, out_dir)
            else:
                written += _report_compensation(session, matrix_run_id, spec, seed, track, out_dir)
    else:
        # Synthetic track: every hypothesis gets a holdings table; every
        # hypothesis except H1 (medium-alone, no second dimension to vary)
        # additionally gets a compensation table.
        for spec in build_synthetic_hypothesis_cell_specs():
            written += _report_holdings(session, matrix_run_id, spec, seed, track, out_dir)
            if spec.hypothesis != "H1":
                written += _report_compensation(session, matrix_run_id, spec, seed, track, out_dir)

    if written == 0:
        print(
            f"No data found for matrix_run_id={matrix_run_id!r} seed={seed} track={track!r} "
            "-- has this run completed yet?"
        )
    else:
        print(f"\n{written} table(s) written to {out_dir}")


if __name__ == "__main__":
    main()
