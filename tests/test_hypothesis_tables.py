from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import Base, CohortHoldingsRecord, IndifferencePointRecord, SimulationRunRecord
from src.reporting.hypothesis_tables import build_compensation_tables, build_equilibrium_holdings_table

MATRIX_RUN_ID = "report-test"


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _register_run(session: Session, run_id: str) -> None:
    session.add(
        SimulationRunRecord(
            run_id=run_id,
            scenario_name="master_simulation",
            research_mode="factual",
            random_seed=0,
            model_roster_summary="vendor/model",
            prompt_version_hash="hash",
            git_commit_hash="hash",
            config_hash="hash",
            created_at=datetime.now(timezone.utc),
        )
    )


def test_build_equilibrium_holdings_table_matches_new_info_pdfs_h1_shape():
    session = _session()
    for utility_type, splits in {
        "crra": {0.0: {"USDC": 0.5, "EURC": 0.3, "PAXG": 0.2}, 6.0: {"USDC": 0.8, "EURC": 0.2, "PAXG": 0.0}},
        "cara": {0.0: {"USDC": 0.5, "EURC": 0.3, "PAXG": 0.2}},
        "epstein_zin_proxy": {0.0: {"USDC": 0.5, "EURC": 0.3, "PAXG": 0.2}},
    }.items():
        run_id = f"{MATRIX_RUN_ID}-H1-{utility_type}-seed0"
        _register_run(session, run_id)
        for cohort, pct_by_symbol in splits.items():
            for symbol, pct in pct_by_symbol.items():
                session.add(
                    CohortHoldingsRecord(
                        run_id=run_id, risk_aversion_cohort=cohort, currency_symbol=symbol, pct_of_wealth=pct
                    )
                )
    session.commit()

    table = build_equilibrium_holdings_table(session, MATRIX_RUN_ID, cell_key="H1", seed=0)

    assert list(table.columns) == ["CARA", "CRRA", "Epstein Zin"]
    assert table.index.name == "Risk Aversion Level"
    assert table.loc["Risk Neutral (a=0)", "CRRA"] == "50% USD, 30% Euro, 20% gold"
    assert table.loc["Most Risk Averse (a=6)", "CRRA"] == "80% USD, 20% Euro"
    assert table.loc["Most Risk Averse (a=6)", "CARA"] == ""


def test_build_compensation_tables_matches_new_info_pdfs_h3_shape():
    session = _session()
    for utility_type, compensation_by_cohort in {
        "crra": {0.0: 0.0001, 6.0: 0.0004},
        "cara": {0.0: 0.00015},
        "epstein_zin_proxy": {0.0: 0.0002},
    }.items():
        run_id = f"{MATRIX_RUN_ID}-H3-{utility_type}-seed0"
        _register_run(session, run_id)
        for cohort, compensation in compensation_by_cohort.items():
            session.add(
                IndifferencePointRecord(
                    run_id=run_id,
                    hypothesis="H3",
                    fixed_currency="USDT",
                    varied_currency="TDUSD",
                    varied_field="liquidity_score",
                    risk_aversion_cohort=cohort,
                    compensation=compensation,
                )
            )
    session.commit()

    tables = build_compensation_tables(session, MATRIX_RUN_ID, "H3", seed=0)

    assert set(tables.keys()) == {
        "Switch from USDT to TDUSD (equivalent change in liquidity_score needed for indifference) [X-Y]"
    }
    table = next(iter(tables.values()))
    assert table.loc["Risk Neutral (a=0)", "CRRA"] == "+0.01%"
    assert table.loc["Most Risk Averse (a=6)", "CRRA"] == "+0.04%"


def test_build_compensation_tables_keys_h2_by_varied_currency():
    session = _session()
    for utility_type in ("crra", "cara", "epstein_zin_proxy"):
        run_id = f"{MATRIX_RUN_ID}-H2-{utility_type}-seed0"
        _register_run(session, run_id)
        session.add(
            IndifferencePointRecord(
                run_id=run_id,
                hypothesis="H2",
                fixed_currency="USDT",
                varied_currency="EURC",
                varied_field="governance_score",
                risk_aversion_cohort=0.0,
                compensation=0.01,
            )
        )
        session.add(
            IndifferencePointRecord(
                run_id=run_id,
                hypothesis="H2",
                fixed_currency="USDT",
                varied_currency="PAXG",
                varied_field="governance_score",
                risk_aversion_cohort=0.0,
                compensation=0.02,
            )
        )
    session.commit()

    tables = build_compensation_tables(session, MATRIX_RUN_ID, "H2", seed=0)

    assert len(tables) == 2
    assert any("EURC" in title for title in tables)
    assert any("PAXG" in title for title in tables)
