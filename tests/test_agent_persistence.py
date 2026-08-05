import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
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


def test_wallets_is_keyed_by_its_true_natural_key():
    """`wallets` rows are "the latest known balance of ONE currency for ONE
    agent in ONE run", so `(run_id, agent_id, currency_symbol)` IS the
    identity of a row -- not a surrogate autoincrement id (round-2 review
    finding I3). Declaring the natural key makes a duplicated
    run/agent/currency triple impossible at the database level instead of
    merely unlikely because `_sync_wallet` happens to rewrite carefully.
    """
    primary_key_columns = {column.name for column in WalletRecord.__table__.primary_key.columns}
    assert primary_key_columns == {"run_id", "agent_id", "currency_symbol"}


def test_wallets_rejects_a_duplicate_run_agent_currency_row():
    """The guarantee the composite key buys: the same currency cannot be
    recorded twice for one agent in one run."""
    session = _session()
    session.add(WalletRecord(run_id="run-cell-1", agent_id="a", currency_symbol="USDC", balance=1.0))
    session.commit()

    session.add(WalletRecord(run_id="run-cell-1", agent_id="a", currency_symbol="USDC", balance=2.0))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    # The same currency for a DIFFERENT run or a DIFFERENT agent is still a
    # distinct, legal row -- the key is composite, not global.
    session.add(WalletRecord(run_id="run-cell-2", agent_id="a", currency_symbol="USDC", balance=3.0))
    session.add(WalletRecord(run_id="run-cell-1", agent_id="b", currency_symbol="USDC", balance=4.0))
    session.commit()
    assert session.query(WalletRecord).count() == 3


def test_sync_wallet_survives_many_days_on_one_long_lived_session():
    """The scenario that made the first implementation reach for a surrogate
    key: `matrix_runner` uses ONE long-lived session for a whole cell/seed
    and `_sync_wallet` rewrites the same agent's rows every simulated day, so
    with a natural primary key each day re-derives identity keys the session
    has already seen. Eleven days of churn -- balances changing, currencies
    appearing and disappearing and re-appearing -- must neither raise nor
    accumulate.
    """
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, agent_id="consumer-seed0-000")

    for day in range(11):
        # USDC persists every day (same identity key, re-derived 11 times);
        # DAI appears and disappears on alternating days (deleted then
        # re-inserted under an identity key the session saw two days ago) and
        # is present again on the final, even-numbered day.
        agent.wallet.balances = {"USDC": 100.0 + day}
        if day % 2 == 0:
            agent.wallet.balances["DAI"] = float(day)
        repo.upsert_agent(agent, "run-cell-1")
        session.commit()

    rows = {(w.currency_symbol, w.balance) for w in session.query(WalletRecord).all()}
    assert rows == {("USDC", 110.0), ("DAI", 10.0)}


def test_sync_wallet_survives_repeated_rewrites_without_an_intervening_commit():
    """Harsher variant of the test above: several days' worth of rewrites
    inside ONE uncommitted unit of work. Nothing in production does this
    today (`persist_full_timestep` commits per day), but it is the exact
    shape that would blow up on a stale identity-map entry, so it is worth
    pinning down rather than leaving to chance.
    """
    session = _session()
    repo = AgentRepository(session)
    profile = load_agent_profiles()["consumer"]
    agent = build_agent(profile, agent_id="consumer-seed0-000")

    agent.wallet.balances = {"USDC": 1.0}
    repo.upsert_agent(agent, "run-cell-1")
    agent.wallet.balances = {"USDC": 2.0, "DAI": 9.0}
    repo.upsert_agent(agent, "run-cell-1")
    agent.wallet.balances = {"USDC": 3.0}
    repo.upsert_agent(agent, "run-cell-1")
    session.commit()

    rows = {(w.currency_symbol, w.balance) for w in session.query(WalletRecord).all()}
    assert rows == {("USDC", 3.0)}
