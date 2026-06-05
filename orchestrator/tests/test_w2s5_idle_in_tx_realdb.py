"""PRD-142 Wave 2 · WS-G · W2-S5 — G1: real-DB idle-in-transaction regression.

W1-S9 (PRD-142 Wave 1) fixed the connection leak where a long-lived session
issues a SELECT — the documented 9-hour idle SELECT on ``agents`` (PRD-135) —
and then sits ``idle in transaction`` across a long ``await`` (an LLM call or an
``asyncio.gather`` of agent coroutines), holding row locks and blocking DDL.

The structural fix is ``end_open_transaction(db)`` — a ``commit`` that ends the
open transaction — called immediately before each such await on the four hot
surfaces, plus a ``rollback`` in ``get_db``'s ``finally`` so a request handler
never returns a connection to the pool still ``idle in transaction``.

W1-S9's own tests proved the *ordering* (select < commit < await) with recording
fakes — no real database. They could not prove that a real Postgres backend
actually ends up ``idle in transaction`` after the read, nor that the commit
actually clears it. **This is that missing real-DB regression.** It connects to
the real test Postgres, hydrates an ``Agent`` (the documented SELECT-on-agents),
and reads the subject backend's ``state`` from ``pg_stat_activity`` via a
*second* connection.

Revert guard (the AC): ``test_end_open_transaction_clears_idle_in_tx_across_await``
**fails** if ``end_open_transaction`` is reverted to a no-op (or removed from the
hot surfaces) — the post-commit state stays ``idle in transaction`` across the
await and the final assert trips.

Marked ``integration``: it needs a live Postgres (the CI ``test.yml`` service,
or a local one). When no database is reachable the module *skips* — it never
turns the fast, DB-less mock suite red.
"""
import asyncio
import os

import pytest
from sqlalchemy import text

# Importing core.database.database builds the SQLAlchemy engine at module load,
# which needs POSTGRES_* creds. setdefault means a real .env / CI service env
# still wins; this only covers a bare shell so the import doesn't crash before
# the skip-guard below can run.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test_db",
}.items():
    os.environ.setdefault(_k, _v)

from core.database.database import (  # noqa: E402
    SessionLocal,
    end_open_transaction,
    engine,
    get_db,
)
from core.models.core import Agent  # noqa: E402

pytestmark = pytest.mark.integration

IDLE_IN_TX = "idle in transaction"


@pytest.fixture(scope="module")
def live_db():
    """Skip the whole module unless the real Postgres engine is reachable.

    Keeps the fast, DB-less local run green (skip, not error) while running for
    real against the CI Postgres service.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # OperationalError and friends
        pytest.skip(f"no reachable Postgres for integration test: {exc}")
    return engine


@pytest.fixture
def observer(live_db):
    """A second, independent connection used only to read ``pg_stat_activity``.

    AUTOCOMMIT so each read sees live backend state and the observer never holds
    a transaction of its own (which would itself show as idle-in-tx). It is a
    different pooled backend from the subject session, so it can watch the
    subject's state without perturbing it.
    """
    conn = live_db.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        yield conn
    finally:
        conn.close()


def _backend_state(observer_conn, pid):
    """The ``pg_stat_activity.state`` of backend ``pid`` (None if it is gone)."""
    return observer_conn.execute(
        text("SELECT state FROM pg_stat_activity WHERE pid = :pid"),
        {"pid": pid},
    ).scalar()


def _hydrate_agent_and_pid(db):
    """Reproduce the hot-surface read: capture the backend pid, then hydrate an
    ``Agent`` (the documented SELECT-on-agents). The first statement opens the
    transaction; ``.first()`` runs inside it. Returns the backend pid.

    ``.first()`` may be ``None`` on an empty table — the transaction opens either
    way, which is all the leak needs.
    """
    pid = db.execute(text("SELECT pg_backend_pid()")).scalar()
    db.query(Agent).first()
    return pid


# ── The leak precondition (proves the regression has teeth) ──────────────────

def test_agents_select_leaves_backend_idle_in_tx(observer):
    """A hydrate-Agent read really does leave the backend ``idle in transaction``.

    This is the condition the W1-S9 fix exists to clear. Asserting it explicitly
    on real Postgres stops the fix tests below from passing vacuously if the
    transaction semantics ever change.
    """
    db = SessionLocal()
    try:
        pid = _hydrate_agent_and_pid(db)
        assert _backend_state(observer, pid) == IDLE_IN_TX
    finally:
        db.rollback()
        db.close()


# ── The fix clears it across the await (the revert guard) ────────────────────

@pytest.mark.asyncio
async def test_end_open_transaction_clears_idle_in_tx_across_await(observer):
    """``end_open_transaction`` must leave the backend ``idle`` — not idle-in-tx —
    for the duration of the long await it precedes.

    Revert guard: if ``end_open_transaction`` is reverted to a no-op (or removed
    from the hot surfaces), the backend stays ``idle in transaction`` across the
    await and the final assert fails. This is the real-DB proof W1-S9 lacked.
    """
    db = SessionLocal()
    try:
        pid = _hydrate_agent_and_pid(db)
        assert _backend_state(observer, pid) == IDLE_IN_TX  # the leak, pre-commit

        end_open_transaction(db)  # THE FIX: commit ends the open transaction

        await asyncio.sleep(0.05)  # the long await the connection used to span

        state = _backend_state(observer, pid)
        assert state == "idle", f"expected backend 'idle' after commit, got {state!r}"
    finally:
        db.close()


# ── get_db returns a clean connection to the pool ────────────────────────────

def test_get_db_request_does_not_leak_idle_in_tx(observer):
    """A ``get_db`` request that reads data must not return its connection to the
    pool ``idle in transaction``.

    Drives the real ``get_db`` generator: a handler hydrates an Agent (opening a
    transaction), then request teardown runs ``get_db``'s ``finally`` (rollback +
    close). Asserts the production invariant that the backend ends not-idle-in-tx.

    (``get_db``'s explicit rollback and the QueuePool's reset-on-return are
    belt-and-suspenders here, so this locks the end-to-end request invariant
    rather than that single line; the strong revert guard for the commit
    mechanism is the test above.)
    """
    gen = get_db()
    db = next(gen)
    pid = _hydrate_agent_and_pid(db)
    assert _backend_state(observer, pid) == IDLE_IN_TX  # handler left a tx open

    # Finish the request: exhausting the generator runs `finally: rollback/close`.
    try:
        next(gen)
    except StopIteration:
        pass

    assert _backend_state(observer, pid) != IDLE_IN_TX
