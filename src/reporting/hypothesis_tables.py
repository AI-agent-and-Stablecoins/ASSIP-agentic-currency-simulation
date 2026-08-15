"""Assembles New info.pdf's own paper-table shapes -- H1's "50% USD, 30%
Euro, 20% gold"-style equilibrium-holdings table, and the H2-H11 "X-Y
compensation" tables -- from a completed run_hypothesis_matrix run's
persisted `CohortHoldingsRecord`/`IndifferencePointRecord` rows.

One run_hypothesis_matrix call produces one column's worth of data per
utility_type (each utility_type is its own cell/seed/utility_type run,
sharing the same matrix_run_id); this module is the "trivial glue" every
prior sub-project's own design spec deferred to whoever builds the final
report -- combining those per-utility_type runs into one 3-column table.
"""

import pandas as pd
from sqlalchemy.orm import Session

from database.models import CohortHoldingsRecord, IndifferencePointRecord
from src.agents.population import HYPOTHESIS_UTILITY_TYPES, RISK_AVERSION_COHORTS
from src.economy.equivalence_framework import EQUIVALENCE_COMPARISONS

RISK_AVERSION_LABELS: dict[float, str] = {
    0.0: "Risk Neutral (a=0)",
    2.0: "Moderate risk averse (a=2)",
    4.0: "More Risk averse (a=4)",
    6.0: "Most Risk Averse (a=6)",
}

UTILITY_TYPE_LABELS: dict[str, str] = {
    "crra": "CRRA",
    "cara": "CARA",
    "epstein_zin_proxy": "Epstein Zin",
}

# H1's currencies are fixed (HYPOTHESIS_CURRENCIES["H1"] = ("USDC", "EURC",
# "PAXG")) -- one real symbol per zone New info.pdf's own H1 table names.
_H1_ZONE_LABELS: dict[str, str] = {"USDC": "USD", "EURC": "Euro", "PAXG": "gold"}


def _run_id(matrix_run_id: str, cell_key: str, utility_type: str, seed: int) -> str:
    return f"{matrix_run_id}-{cell_key}-{utility_type}-seed{seed}"


def _pivot(columns: dict[str, dict[float, str]]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            label: [per_cohort.get(cohort, "") for cohort in RISK_AVERSION_COHORTS]
            for label, per_cohort in columns.items()
        },
        index=[RISK_AVERSION_LABELS[cohort] for cohort in RISK_AVERSION_COHORTS],
    )
    df.index.name = "Risk Aversion Level"
    return df


def build_equilibrium_holdings_table(
    session: Session, matrix_run_id: str, cell_key: str = "H1", seed: int = 0
) -> pd.DataFrame:
    """New info.pdf's H1 "Baseline model" table: rows are risk-aversion
    levels, columns are utility functions, cells are "50% USD, 30% Euro,
    20% gold"-style holdings breakdowns. `cell_key` selects which of H1's
    variants to report ("H1" baseline, "H1_cb" cross-border, "H1_depeg_event"
    / "H1_bank_failure" event-based)."""
    columns: dict[str, dict[float, str]] = {}
    for utility_type in sorted(HYPOTHESIS_UTILITY_TYPES):
        run_id = _run_id(matrix_run_id, cell_key, utility_type, seed)
        rows = session.query(CohortHoldingsRecord).filter(CohortHoldingsRecord.run_id == run_id).all()

        by_cohort: dict[float, dict[str, float]] = {}
        for row in rows:
            by_cohort.setdefault(row.risk_aversion_cohort, {})[row.currency_symbol] = row.pct_of_wealth

        per_cohort_label: dict[float, str] = {}
        for cohort, pct_by_symbol in by_cohort.items():
            # New info.pdf's own H1 example omits a zero-rounding currency
            # entirely ("80% USD, 20% Euro", no "0% gold") rather than
            # listing it at 0% -- matching its own aside that "only the
            # risk seeking agent wants gold".
            ordered = sorted(pct_by_symbol.items(), key=lambda item: item[1], reverse=True)
            per_cohort_label[cohort] = ", ".join(
                f"{pct * 100:.0f}% {_H1_ZONE_LABELS.get(symbol, symbol)}"
                for symbol, pct in ordered
                if round(pct * 100) > 0
            )

        columns[UTILITY_TYPE_LABELS[utility_type]] = per_cohort_label

    return _pivot(columns)


def _format_compensation(varied_field: str, compensation: float) -> str:
    if varied_field == "gas_fee":
        return f"{compensation:+.4f}"
    return f"{compensation * 100:+.2f}%"


def build_compensation_tables(
    session: Session, matrix_run_id: str, hypothesis: str, cell_key: str | None = None, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """New info.pdf's H3/H4-style "X-Y" compensation tables: rows are
    risk-aversion levels, columns are utility functions, cells are the
    compensation (X-Y) an agent needs to switch. Returns one table per
    `EQUIVALENCE_COMPARISONS[hypothesis]` entry (H2 has two, one per
    varied_currency; every other hypothesis has exactly one), keyed by a
    human-readable title. `cell_key` selects which variant to report
    (defaults to `hypothesis` itself, e.g. "H3" -- the baseline cell)."""
    resolved_cell_key = cell_key or hypothesis
    tables: dict[str, pd.DataFrame] = {}

    for comparison in EQUIVALENCE_COMPARISONS[hypothesis]:
        columns: dict[str, dict[float, str]] = {}
        for utility_type in sorted(HYPOTHESIS_UTILITY_TYPES):
            run_id = _run_id(matrix_run_id, resolved_cell_key, utility_type, seed)
            rows = (
                session.query(IndifferencePointRecord)
                .filter(
                    IndifferencePointRecord.run_id == run_id,
                    IndifferencePointRecord.varied_currency == comparison.varied_currency,
                )
                .all()
            )
            columns[UTILITY_TYPE_LABELS[utility_type]] = {
                row.risk_aversion_cohort: _format_compensation(row.varied_field, row.compensation) for row in rows
            }

        title = (
            f"Switch from {comparison.fixed_currency} to {comparison.varied_currency} "
            f"(equivalent change in {comparison.varied_field} needed for indifference) [X-Y]"
        )
        tables[title] = _pivot(columns)

    return tables
