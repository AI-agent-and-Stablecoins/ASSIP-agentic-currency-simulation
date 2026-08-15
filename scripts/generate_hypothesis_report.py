"""Generates New info.pdf's paper-style tables (H1's equilibrium-holdings
table; H2-H11's compensation tables) from a completed run_hypothesis_matrix
run, for every cell that has data. Prints each table to the console and
writes it to reports/<matrix_run_id>/<cell_key>.csv.

Usage:
    DATABASE_URL="sqlite:///./assip.db" .venv/bin/python \
        scripts/generate_hypothesis_report.py <matrix_run_id> [seed]
"""

import sys

from database.session import new_session
from src.economy.hypothesis_scenarios import build_hypothesis_cell_specs
from src.reporting.hypothesis_tables import build_compensation_tables, build_equilibrium_holdings_table
from src.utils.constants import REPO_ROOT


def _is_empty(table) -> bool:
    return bool((table == "").all().all())


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: generate_hypothesis_report.py <matrix_run_id> [seed]")
    matrix_run_id = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    session = new_session()
    out_dir = REPO_ROOT / "reports" / matrix_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for spec in build_hypothesis_cell_specs():
        if spec.hypothesis == "H1":
            table = build_equilibrium_holdings_table(session, matrix_run_id, cell_key=spec.key, seed=seed)
            if _is_empty(table):
                continue
            path = out_dir / f"{spec.key}.csv"
            table.to_csv(path)
            print(f"\n=== {spec.key} (equilibrium holdings) ===")
            print(table.to_string())
            print(f"-> {path}")
            written += 1
        else:
            tables = build_compensation_tables(session, matrix_run_id, spec.hypothesis, cell_key=spec.key, seed=seed)
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

    if written == 0:
        print(f"No data found for matrix_run_id={matrix_run_id!r} seed={seed} -- has this run completed yet?")
    else:
        print(f"\n{written} table(s) written to {out_dir}")


if __name__ == "__main__":
    main()
