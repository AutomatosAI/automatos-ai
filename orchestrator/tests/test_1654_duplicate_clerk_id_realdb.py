"""#1654: two ``users`` rows with one Clerk user ID name one person, the lowest ``users.id``.

The model declares ``users.clerk_user_id`` unique, but no migration creates the constraint, so a
database built by migrations may hold a duplicate. Every Clerk-lane read names the person with a
scalar subquery (``CLERK_PERSON_PK_SQL``, ``clerk_person_pk_expression``); over two rows a bare
one makes Postgres raise "more than one row returned by a subquery", and that person's every
request fails. Each read now takes the lowest ``users.id``, the original row.

The statement checks need no database. The rest run on real Postgres (``@integration``) and skip
when none is reachable (CI runs them). The test database is built by ``create_all``, which does
create the unique constraint, so the test drops it (and any unique index on the column alone)
inside its own transaction, the shape of a database built by migrations, makes the duplicate,
reads, and rolls the whole transaction back: Postgres DDL is transactional, so the constraint
returns untouched and nothing is committed.
"""
from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.auth import clerk_cutover, clerk_person  # noqa: E402
from core.auth import hybrid  # noqa: E402
from core.auth import person as people  # noqa: E402
from core.auth.user_link import user_id_for_clerk_id  # noqa: E402
from core.auth.workspace_permission import resolve_workspace_role  # noqa: E402
from core.database.database import get_database_url  # noqa: E402
from core.models.core import User  # noqa: E402
from modules.tools.discovery.platform_executor import _workspace_role_for_clerk  # noqa: E402

# Drops every unique constraint, then every unique index, on users.clerk_user_id alone: the
# shape of a database whose migrations never created one. Runs inside the test's transaction.
_DROP_CLERK_ID_UNIQUENESS = """
DO $$
DECLARE r record;
BEGIN
  FOR r IN SELECT c.conname FROM pg_constraint c
           JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
           WHERE c.conrelid = 'users'::regclass AND c.contype = 'u'
             AND cardinality(c.conkey) = 1 AND a.attname = 'clerk_user_id'
  LOOP EXECUTE format('ALTER TABLE users DROP CONSTRAINT %I', r.conname); END LOOP;
  FOR r IN SELECT i.relname FROM pg_index x
           JOIN pg_class i ON i.oid = x.indexrelid
           JOIN pg_attribute a ON a.attrelid = x.indrelid AND a.attnum = x.indkey[0]
           WHERE x.indrelid = 'users'::regclass AND x.indisunique
             AND x.indnatts = 1 AND a.attname = 'clerk_user_id'
  LOOP EXECUTE format('DROP INDEX %I', r.relname); END LOOP;
END $$
"""
_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '5s'"


# --------------------------------------------------------------------------- statements, no database


def test_the_clerk_person_subquery_takes_the_lowest_id_and_one_row():
    assert clerk_person.CLERK_PERSON_PK_SQL == (
        "(SELECT id FROM users WHERE clerk_user_id = :cid ORDER BY id LIMIT 1)"
    )


def test_the_orm_subquery_takes_the_lowest_id_and_one_row():
    sql = " ".join(str(select(clerk_person.clerk_person_pk_expression("c"))).split())
    assert "ORDER BY users.id LIMIT :param_1" in sql


def test_the_row_reads_take_the_lowest_id():
    for param, statement in clerk_person._PK_BY_CLERK_ID.items():
        assert str(statement) == f"SELECT id FROM users WHERE clerk_user_id = :{param} ORDER BY id LIMIT 1"
    query = " ".join(str(clerk_person.clerk_person_query(Session(), User.id, "c")).split())
    assert "WHERE users.id = (SELECT users.id FROM users WHERE users.clerk_user_id = " in query
    assert query.endswith("ORDER BY users.id LIMIT :param_1)")


def test_the_cutover_reads_the_same_lowest_id_row():
    seen = []

    class _Db:
        def execute(self, statement, params):
            seen.append(str(statement))
            return SimpleNamespace(fetchone=lambda: (3, "c", "a@example.test"))

    assert clerk_cutover._app_user(_Db(), "c").id == 3
    assert seen == ["SELECT id, clerk_user_id, email FROM users WHERE clerk_user_id = :cid ORDER BY id LIMIT 1"]


# --------------------------------------------------------------------------- the reads, on Postgres


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for probe in ("SELECT clerk_user_id FROM users LIMIT 1", "SELECT 1 FROM workspaces LIMIT 1"):
                c.execute(text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the duplicate Clerk ID suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


def _user(db, clerk_id: str, label: str) -> int:
    return db.execute(
        text("INSERT INTO users (username, email, clerk_user_id, is_active) VALUES (:e, :e, :c, true) RETURNING id"),
        {"e": f"issue1654-{label}-{uuid.uuid4().hex[:10]}@example.test", "c": clerk_id},
    ).scalar_one()


def _workspace(db, owner_pk: int, role: str) -> str:
    """A workspace this user owns, with their active member row in ``role``."""
    ws = str(uuid.uuid4())
    db.execute(text("INSERT INTO workspaces (id, name, owner_id, is_active) "
                    "VALUES (CAST(:id AS uuid), 'Issue 1654', :owner, true)"), {"id": ws, "owner": owner_pk})
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, :role, true)"), {"ws": ws, "uid": owner_pk, "role": role})
    return ws


@pytest.fixture
def duplicated(engine):
    """One open, never-committed session holding two users rows with one Clerk user ID.

    The original owns ``ws`` (member row ``owner``); the duplicate owns ``ws_dup`` (``admin``). The original row is then touched,
    as a sign-in stamp would, so it sits after the duplicate on disk and a read with no order
    meets the duplicate first.
    """
    db = Session(bind=engine)
    try:
        db.execute(text(_LOCK_TIMEOUT))
        db.execute(text(_DROP_CLERK_ID_UNIQUENESS))
        cid = f"user_issue1654_{uuid.uuid4().hex[:10]}"
        original, duplicate = _user(db, cid, "original"), _user(db, cid, "duplicate")
        ws, ws_dup = _workspace(db, original, "owner"), _workspace(db, duplicate, "admin")
        db.execute(text("UPDATE users SET last_sign_in = now() WHERE id = :id"), {"id": original})
        yield SimpleNamespace(db=db, cid=cid, original=original, duplicate=duplicate, ws=ws, ws_dup=ws_dup)
    finally:
        db.rollback()
        db.close()


@pytest.mark.integration
def test_the_request_lane_reads_the_original_row(duplicated):
    d = duplicated
    assert d.original < d.duplicate
    person = people.by_clerk_id(d.cid)
    assert hybrid._user_has_workspace_access(d.db, d.cid, uuid.UUID(d.ws)) is True
    assert people.may_open_workspace(d.db, person, d.ws_dup) is False
    assert str(people.default_workspace(d.db, person)) == d.ws
    for ws, role in ((d.ws, "owner"), (d.ws_dup, None)):
        ctx = SimpleNamespace(user=SimpleNamespace(clerk_user_id=d.cid), workspace_id=ws)
        assert resolve_workspace_role(d.db, ctx) == role
        assert _workspace_role_for_clerk(d.db, ws, d.cid) == role


@pytest.mark.integration
def test_every_clerk_person_read_names_the_original_row(duplicated):
    d = duplicated
    assert d.db.execute(select(clerk_person.clerk_person_pk_expression(d.cid))).scalar_one() == d.original
    assert clerk_person.clerk_person_row(d.db, d.cid)[0] == d.original
    assert clerk_person.clerk_person_row(d.db, d.cid, param=clerk_person.PRINCIPAL_PARAM)[0] == d.original
    assert clerk_person.clerk_person_query(d.db, User.id, d.cid).first()[0] == d.original
    assert user_id_for_clerk_id(d.db, d.cid) == d.original
