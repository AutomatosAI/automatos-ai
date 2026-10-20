"""PRD-265 US-002 / US-005 against real Postgres (``@integration``).

- The same source event recorded twice changes nothing: one set of rows, one
  balance move.
- The balance moves in the same transaction as the ledger rows: a rollback takes
  both back.
- A debit that spans buckets writes one row per bucket, and the balance matches
  the ledger's sums.
- The migration's trial conversion: the dollars left × 100, floored, keyed once
  per owner; a second run changes nothing.

Skips cleanly when no Postgres with the ``prd265_credit_ledger`` schema is
reachable (CI runs it).
"""
from __future__ import annotations

import importlib.util
import json
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.database.database import get_database_url  # noqa: E402
from services.credits import ledger  # noqa: E402
from services.credits.ledger import Entry  # noqa: E402

pytestmark = pytest.mark.integration

_MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "prd265_credit_ledger.py"


def _migration():
    spec = importlib.util.spec_from_file_location("_prd265_credit_ledger_db", _MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("workspaces", "credit_ledger", "credit_balances"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
            c.execute(text("SELECT ask_id FROM llm_usage LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"credit ledger suite needs Postgres with prd265_credit_ledger applied: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspaces(engine, new_session):
    made: list[str] = []

    def make(owner_id=None, trial=None) -> str:
        ws = str(uuid.uuid4())
        s = new_session()
        onboarding = {"stage": "not_started", **({"trial": trial} if trial is not None else {})}
        s.execute(
            text("INSERT INTO workspaces (id, name, slug, owner_id, onboarding) "
                 "VALUES (CAST(:id AS uuid), :n, :n, :owner, CAST(:ob AS jsonb))"),
            {"id": ws, "n": f"credits-{ws[:8]}", "owner": owner_id, "ob": json.dumps(onboarding)},
        )
        s.commit()
        made.append(ws)
        return ws

    yield make
    s = new_session.sweep()
    for ws in made:
        s.execute(text("DELETE FROM credit_ledger WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
        s.execute(text("DELETE FROM credit_balances WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()


@pytest.fixture
def owner(engine, new_session):
    tag = uuid.uuid4().hex[:10]
    s = new_session()
    uid = s.execute(
        text("INSERT INTO users (username, email, clerk_user_id, is_active) "
             "VALUES (:e, :e, :cid, true) RETURNING id"),
        {"e": f"credits-{tag}@example.test", "cid": f"user_credits_{tag}"},
    ).scalar()
    s.commit()
    yield uid
    s = new_session.sweep()
    s.execute(text("DELETE FROM users WHERE id = :id"), {"id": uid})
    s.commit()


def _rows(s, ws):
    return s.execute(
        text("SELECT kind, bucket, credits FROM credit_ledger WHERE workspace_id = CAST(:ws AS uuid) ORDER BY id"),
        {"ws": ws},
    ).fetchall()


def _grant(ref: str, credits: int, bucket: str = ledger.BUCKET_TRIAL) -> Entry:
    return Entry(kind=ledger.KIND_GRANT, source_type="test_grant", source_ref=ref, credits=credits, bucket=bucket)


def test_the_same_source_event_twice_changes_nothing(workspaces, new_session):
    ws = workspaces()
    ref = f"g-{ws}"
    s = new_session()
    assert ledger.record(s, ws, _grant(ref, 500)) is True
    s.commit()
    assert ledger.record(s, ws, _grant(ref, 500)) is False
    s.commit()
    assert len(_rows(s, ws)) == 1
    assert ledger.read_balance(s, ws).trial == 500


def test_the_balance_moves_in_the_same_transaction_as_the_rows(workspaces, new_session):
    ws = workspaces()
    s = new_session()
    ledger.record(s, ws, _grant(f"g-{ws}", 200))
    s.rollback()
    assert _rows(s, ws) == []
    assert ledger.read_balance(s, ws).trial == 0


def test_a_debit_across_buckets_writes_one_row_per_bucket(workspaces, new_session):
    ws = workspaces()
    s = new_session()
    ledger.record(s, ws, _grant(f"t-{ws}", 3, ledger.BUCKET_TRIAL))
    ledger.record(s, ws, _grant(f"m-{ws}", 10, ledger.BUCKET_MONTHLY))
    ledger.record(s, ws, Entry(kind=ledger.KIND_DEBIT, source_type="ask", source_ref=f"ask-{ws}",
                               credits=-5, band="standard"))
    s.commit()
    debits = [r for r in _rows(s, ws) if r[0] == ledger.KIND_DEBIT]
    assert sorted((r[1], r[2]) for r in debits) == [("monthly", -2), ("trial", -3)]
    bal = ledger.read_balance(s, ws)
    assert (bal.trial, bal.monthly, bal.topup) == (0, 8, 0)
    assert ledger.entry_rows(s, "ask", f"ask-{ws}") == [("trial", -3), ("monthly", -2)]


def test_the_trial_conversion_keeps_the_dollars_left_once_per_owner(owner, workspaces, new_session):
    first = workspaces(owner_id=owner, trial={"granted_usd": 5.0, "spent_usd": 1.237, "state": "active"})
    second = workspaces(owner_id=owner, trial={"granted_usd": 5.0, "spent_usd": 0, "state": "active"})
    spent = workspaces(trial={"granted_usd": 5.0, "spent_usd": 5.4, "state": "exhausted"})
    mig = _migration()
    for _ in range(2):  # a second run changes nothing
        s = new_session()
        s.execute(text(mig.CONVERT_TRIALS_SQL))
        s.execute(text(mig.REBUILD_BALANCES_SQL))
        s.commit()
    s = new_session()
    assert _rows(s, first) == [("grant", "trial", 376)]  # ($5 - $1.237) x 100, floored
    assert _rows(s, second) == []  # the owner already held a trial
    assert ledger.read_balance(s, first).trial == 376
    assert _rows(s, spent) == [("grant", "trial", 0)]  # held, and nothing left
    assert ledger.read_balance(s, spent).trial == 0
