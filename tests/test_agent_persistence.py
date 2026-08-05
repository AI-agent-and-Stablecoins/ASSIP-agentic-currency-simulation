from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.models import AgentRecord, Base, WalletRecord
from database.repository import AgentRepository
from src.agents.agent_factory import build_agent, load_agent_profiles


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_upsert_agent_persists_population_fields():
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)
    agent.currency_zone = "EUR"
    agent.assigned_model = "anthropic/claude-sonnet-5"
    agent.cara_coefficient = 2.0

    repo.upsert_agent(agent, "run-a")
    session.commit()

    row = session.get(AgentRecord, {"run_id": "run-a", "id": agent.agent_id})
    assert row.currency_zone == "EUR"
    assert row.assigned_model == "anthropic/claude-sonnet-5"
    assert row.cara_coefficient == 2.0


def test_upsert_agent_allows_none_population_fields():
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile)  # currency_zone/assigned_model/cara_coefficient all None

    repo.upsert_agent(agent, "run-a")
    session.commit()

    row = session.get(AgentRecord, {"run_id": "run-a", "id": agent.agent_id})
    assert row.currency_zone is None
    assert row.assigned_model is None
    assert row.cara_coefficient is None


def test_upsert_agent_scopes_identity_by_run_id_so_two_runs_sharing_an_agent_id_do_not_collide():
    """The core regression this schema change exists for.

    `src/agents/population.py`'s agent ids are a pure function of
    `(profile_name, seed, slot_index)`, and `run_matrix` runs all 13 matrix
    cells with the SAME seeds -- so every cell generates the identical 100
    agent ids. With the old bare `agents.id` primary key, the second run's
    upsert found the first run's row and left the first run's field values in
    place (sequential runs), or raced to INSERT the same id across processes
    and crashed with `UNIQUE constraint failed: agents.id`.

    Two runs writing the SAME agent id must now produce two independent
    rows, each holding its OWN field values.
    """
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]

    agent_a = build_agent(profile, agent_id="consumer-seed0-000")
    agent_a.currency_zone = "USD"
    agent_a.assigned_model = "vendor/model-a"
    repo.upsert_agent(agent_a, "run-cell-1")

    agent_b = build_agent(profile, agent_id="consumer-seed0-000")
    agent_b.currency_zone = "EUR"
    agent_b.assigned_model = "vendor/model-b"
    repo.upsert_agent(agent_b, "run-cell-2")

    session.commit()

    rows = session.query(AgentRecord).filter(AgentRecord.id == "consumer-seed0-000").all()
    assert len(rows) == 2
    by_run = {row.run_id: row for row in rows}
    assert by_run["run-cell-1"].currency_zone == "USD"
    assert by_run["run-cell-1"].assigned_model == "vendor/model-a"
    # The second run did NOT overwrite the first, and the first did NOT
    # shadow the second (the old code's actual behavior: run-cell-2's values
    # were silently dropped because a row with that id already existed).
    assert by_run["run-cell-2"].currency_zone == "EUR"
    assert by_run["run-cell-2"].assigned_model == "vendor/model-b"


def test_sync_wallet_scopes_by_run_id_so_one_run_does_not_clobber_anothers_wallet_rows():
    """`_sync_wallet` deletes-then-reinserts an agent's wallet rows on EVERY
    simulated day. Scoped by `agent_id` alone (the old behavior), cell 2's
    first day wiped cell 1's committed wallet snapshot for the same shared
    agent id -- silently, with no concurrency involved at all, so by the end
    of a 13-cell matrix run `wallets` held only whichever cell happened to
    write last. Both runs' rows must now survive independently.
    """
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]

    agent_a = build_agent(profile, agent_id="consumer-seed0-000")
    agent_a.wallet.balances = {"USDC": 100.0}
    repo.upsert_agent(agent_a, "run-cell-1")
    session.commit()

    agent_b = build_agent(profile, agent_id="consumer-seed0-000")
    agent_b.wallet.balances = {"EURC": 7.0}
    repo.upsert_agent(agent_b, "run-cell-2")
    session.commit()

    run_1 = {(w.currency_symbol, w.balance) for w in session.query(WalletRecord).filter_by(run_id="run-cell-1")}
    run_2 = {(w.currency_symbol, w.balance) for w in session.query(WalletRecord).filter_by(run_id="run-cell-2")}
    assert run_1 == {("USDC", 100.0)}
    assert run_2 == {("EURC", 7.0)}


def test_sync_wallet_still_replaces_rather_than_accumulates_within_one_run():
    """Run-scoping must not turn the day-over-day wallet mirror into an
    append-only log: within ONE run, a later day's balances still REPLACE
    the earlier day's rows for that agent (that is the table's contract --
    per-day history lives in `agent_states`, not here).
    """
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, agent_id="consumer-seed0-000")

    agent.wallet.balances = {"USDC": 100.0}
    repo.upsert_agent(agent, "run-cell-1")
    session.commit()

    agent.wallet.balances = {"USDC": 42.0, "DAI": 5.0}
    repo.upsert_agent(agent, "run-cell-1")
    session.commit()

    rows = {(w.currency_symbol, w.balance) for w in session.query(WalletRecord).filter_by(run_id="run-cell-1")}
    assert rows == {("USDC", 42.0), ("DAI", 5.0)}
