"""PRD-266 US-010b — push tickets and digest posts: the models and the revision describe two tables.

Expo's tickets wait in ``mobile_push_tickets`` until the receipts job reads them (§4.4);
``mobile_digest_posts`` makes the daily digest once per workspace, person and local date
(§4.5, K5). Proven here:

* ``core.models.mobile_push`` and the Alembic revision ``prd266_w2_push`` declare the same
  columns, in the same order, with the same nullability, primary keys, foreign keys (the
  device, the workspace and the person CASCADE), defaults and index; the push-type CHECK
  matches the newest revision that sets it, ``prd268_w2_delivery`` (PRD-268 K19).
* ``PUSH_TYPES`` is the one list the CHECK is built from.
* The revision chains onto the old head, creates only what is absent (every CREATE is
  ``IF NOT EXISTS``), alters no other table, and its downgrade drops only these two.
* On Postgres: the upgrade survives a ``create_all`` database (it puts back only a missing
  index); ``INSERT … ON CONFLICT DO NOTHING`` claims a digest post once per
  (workspace, person, date); deleting the device deletes its tickets; deleting the
  workspace deletes both kinds of row; an unknown push type is refused.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import uuid
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
MIGRATION = ORCH / "alembic" / "versions" / "prd266_w2_push.py"
# The newest revision that sets the push-type CHECK: PRD-268 W2 widens it to suggestion (K19).
CHECK_MIGRATION = ORCH / "alembic" / "versions" / "prd268_w2_delivery.py"
OLD_HEAD = "prd261_revocation_feed"
TABLES = ("mobile_push_tickets", "mobile_digest_posts")

TICKET_COLUMNS = ("ticket_id", "device_id", "workspace_id", "push_type", "created_at")
DIGEST_COLUMNS = ("workspace_id", "user_id", "local_date", "posted_at")
PUSH_TYPES = ("needs_answer", "review", "auto_message", "suggestion")
TICKET_INDEX = ("ix_mobile_push_tickets_created_at", ["created_at"])
TICKET_CHECK = "ck_mobile_push_tickets_push_type"
DAY = date(2026, 10, 24)


def _ticket_table():
    from core.models.mobile_push import MobilePushTicket

    return MobilePushTicket.__table__


def _digest_table():
    from core.models.mobile_push import MobileDigestPost

    return MobileDigestPost.__table__


def _ddl_lines(table: str) -> list[str]:
    pattern = rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\s*\)\n"
    (ddl,) = re.findall(pattern, MIGRATION.read_text(), re.S)
    return [line.strip().rstrip(",") for line in ddl.strip().splitlines()]


def _ddl_columns(table: str) -> dict[str, str]:
    return {
        line.split()[0]: line for line in _ddl_lines(table)
        if not line.startswith(("CONSTRAINT", "PRIMARY KEY"))
    }


def _statements(step: str, path: Path = MIGRATION) -> list[str]:
    spec = importlib.util.spec_from_file_location(f"{path.stem}_{step}", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements: list[str] = []
    migration.op = SimpleNamespace(execute=statements.append)
    getattr(migration, step)()
    return statements


def _fks(table) -> dict[str, tuple[str, str]]:
    return {fk.parent.name: (fk.target_fullname, fk.ondelete) for fk in table.foreign_keys}


# --------------------------------------------------------------------------- one shape


def test_the_models_and_the_revision_declare_every_column_in_one_order():
    assert tuple(column.name for column in _ticket_table().columns) == TICKET_COLUMNS
    assert tuple(_ddl_columns("mobile_push_tickets")) == TICKET_COLUMNS
    assert tuple(column.name for column in _digest_table().columns) == DIGEST_COLUMNS
    assert tuple(_ddl_columns("mobile_digest_posts")) == DIGEST_COLUMNS


def test_every_column_is_not_null_on_both_sides():
    for table, name in ((_ticket_table(), "mobile_push_tickets"), (_digest_table(), "mobile_digest_posts")):
        assert all(not column.nullable for column in table.columns), name
        lines = _ddl_columns(name)
        assert all("NOT NULL" in line or "PRIMARY KEY" in line for line in lines.values()), name


def test_a_ticket_is_keyed_by_expos_id_and_a_digest_post_by_workspace_person_and_date():
    assert [column.name for column in _ticket_table().primary_key.columns] == ["ticket_id"]
    assert "ticket_id VARCHAR(64) PRIMARY KEY" in _ddl_lines("mobile_push_tickets")
    assert [column.name for column in _digest_table().primary_key.columns] == [
        "workspace_id", "user_id", "local_date",
    ]
    assert "PRIMARY KEY (workspace_id, user_id, local_date)" in _ddl_lines("mobile_digest_posts")


def test_the_device_the_workspace_and_the_person_cascade():
    assert _fks(_ticket_table()) == {
        "device_id": ("mobile_devices.id", "CASCADE"),
        "workspace_id": ("workspaces.id", "CASCADE"),
    }
    assert _fks(_digest_table()) == {
        "workspace_id": ("workspaces.id", "CASCADE"),
        "user_id": ("users.id", "CASCADE"),
    }
    tickets, digests = _ddl_columns("mobile_push_tickets"), _ddl_columns("mobile_digest_posts")
    assert "REFERENCES mobile_devices(id) ON DELETE CASCADE" in tickets["device_id"]
    assert "REFERENCES workspaces(id) ON DELETE CASCADE" in tickets["workspace_id"]
    assert "REFERENCES workspaces(id) ON DELETE CASCADE" in digests["workspace_id"]
    assert "REFERENCES users(id) ON DELETE CASCADE" in digests["user_id"]


def test_the_push_type_check_is_built_from_push_types_and_matches_the_revision():
    from sqlalchemy import CheckConstraint

    from core.models import mobile_push as model

    assert model.PUSH_TYPES == PUSH_TYPES
    checks = {c.name: str(c.sqltext) for c in _ticket_table().constraints if isinstance(c, CheckConstraint)}
    assert set(checks) == {TICKET_CHECK}
    assert tuple(re.findall(r"'([a-z_]+)'", checks[TICKET_CHECK])) == PUSH_TYPES
    (created,) = [line for line in _ddl_lines("mobile_push_tickets") if line.startswith(f"CONSTRAINT {TICKET_CHECK} ")]
    assert tuple(re.findall(r"'([a-z_]+)'", created)) == PUSH_TYPES[:3]  # prd266_w2_push's three
    newest = f"ADD CONSTRAINT {TICKET_CHECK} CHECK ({checks[TICKET_CHECK]});"
    assert any(newest in statement for statement in _statements("upgrade", CHECK_MIGRATION))
    assert not [c for c in _digest_table().constraints if isinstance(c, CheckConstraint)]


def test_created_at_is_indexed_and_both_timestamps_default_to_now():
    name, columns = TICKET_INDEX
    assert {index.name: [c.name for c in index.columns] for index in _ticket_table().indexes} == {name: columns}
    assert not _digest_table().indexes
    assert f"CREATE INDEX IF NOT EXISTS {name} " in MIGRATION.read_text()
    assert "ON mobile_push_tickets (created_at);" in MIGRATION.read_text()
    assert _ticket_table().c.created_at.server_default is not None
    assert _digest_table().c.posted_at.server_default is not None
    assert "DEFAULT now()" in _ddl_columns("mobile_push_tickets")["created_at"]
    assert "DEFAULT now()" in _ddl_columns("mobile_digest_posts")["posted_at"]


def test_the_models_are_registered_for_create_all():
    import core.models as models
    from core.database.base import Base

    assert Base.metadata.tables["mobile_push_tickets"] is models.MobilePushTicket.__table__
    assert Base.metadata.tables["mobile_digest_posts"] is models.MobileDigestPost.__table__
    assert models.PUSH_TYPES == PUSH_TYPES


# --------------------------------------------------------------------------- the revision


def _assigned(name: str):
    tree = ast.parse(MIGRATION.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{MIGRATION.name} assigns no {name}")


def test_the_revision_chains_onto_the_old_head():
    assert _assigned("revision") == "prd266_w2_push"
    assert _assigned("down_revision") == OLD_HEAD


def test_every_create_is_if_not_exists_and_no_other_table_is_touched():
    statements = _statements("upgrade")
    assert len(statements) == 3
    for statement in statements:
        sql = " ".join(statement.split())
        assert re.match(r"CREATE (TABLE|INDEX) IF NOT EXISTS ", sql), sql
        assert not re.search(r"\b(ALTER TABLE|DROP|UPDATE \w+ SET|DELETE FROM|INSERT INTO)\b", sql), sql
    created = [re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", s) for s in statements]
    assert [m.group(1) for m in created if m] == list(TABLES)
    assert re.search(r"\bON mobile_push_tickets \(", statements[1])
    assert "ALTER TABLE" not in MIGRATION.read_text()


def test_the_downgrade_drops_only_these_two_tables():
    assert _statements("downgrade") == [
        "DROP TABLE IF EXISTS mobile_digest_posts;",
        "DROP TABLE IF EXISTS mobile_push_tickets;",
    ]


# --------------------------------------------------------------------------- Postgres (CI)


def _seed_user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-{tag}", "e": f"prd266-{tag}@t.test"},
    ).scalar()


def _seed_device(db, workspace_id: str, user_id: int) -> str:
    device_id = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO mobile_devices (id, workspace_id, user_id, kind, status) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :u, 'clerk', 'active')"),
        {"id": device_id, "ws": workspace_id, "u": user_id})
    return device_id


def _ticket(db, device_id: str, workspace_id: str, push_type: str = "needs_answer") -> str:
    ticket_id = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO mobile_push_tickets (ticket_id, device_id, workspace_id, push_type) "
        "VALUES (:t, CAST(:d AS uuid), CAST(:ws AS uuid), :p)"),
        {"t": ticket_id, "d": device_id, "ws": workspace_id, "p": push_type})
    return ticket_id


def _claim_digest(db, workspace_id: str, user_id: int, local_date: date) -> bool:
    claimed = db.execute(text(
        "INSERT INTO mobile_digest_posts (workspace_id, user_id, local_date) "
        "VALUES (CAST(:ws AS uuid), :u, :d) ON CONFLICT DO NOTHING RETURNING user_id"),
        {"ws": workspace_id, "u": user_id, "d": local_date}).scalar()
    return claimed is not None


def _index_names(db) -> set[str]:
    return set(db.execute(text(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'mobile_push_tickets'")).scalars())


def test_the_upgrade_keeps_create_all_tables_and_adds_only_the_missing_index(db_session):
    """create_all built both tables; the upgrade runs over them and puts back the dropped
    index. Everything runs inside db_session's transaction and is rolled back."""
    name, _ = TICKET_INDEX
    assert name in _index_names(db_session)
    db_session.execute(text(f"DROP INDEX {name}"))
    assert name not in _index_names(db_session)

    for statement in _statements("upgrade"):
        db_session.execute(text(statement))

    assert name in _index_names(db_session)
    for table, columns in (("mobile_push_tickets", TICKET_COLUMNS), ("mobile_digest_posts", DIGEST_COLUMNS)):
        found = db_session.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t "
            "ORDER BY ordinal_position"), {"t": table}).scalars().all()
        assert tuple(found) == columns


def _count(db, table: str, workspace_id: str) -> int:
    assert table in TABLES
    return db.execute(
        text(f"SELECT count(*) FROM {table} WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": workspace_id},
    ).scalar()


def test_a_digest_post_is_claimed_once_per_workspace_person_and_date(db_session, seed_workspace):
    ws, other_ws, user, other_user = seed_workspace(), seed_workspace(), _seed_user(db_session), _seed_user(db_session)
    assert _claim_digest(db_session, ws, user, DAY) is True
    assert _claim_digest(db_session, ws, user, DAY) is False
    assert _claim_digest(db_session, ws, user, date(2026, 10, 25)) is True
    assert _claim_digest(db_session, ws, other_user, DAY) is True
    assert _claim_digest(db_session, other_ws, user, DAY) is True
    assert _count(db_session, "mobile_digest_posts", ws) == 3
    posted_at = db_session.execute(text(
        "SELECT posted_at FROM mobile_digest_posts WHERE workspace_id = CAST(:ws AS uuid) AND user_id = :u "
        "AND local_date = :d"), {"ws": ws, "u": user, "d": DAY}).scalar()
    assert posted_at is not None


def test_every_push_type_is_kept_and_its_created_at_defaults(db_session, seed_workspace):
    ws, user = seed_workspace(), _seed_user(db_session)
    device = _seed_device(db_session, ws, user)
    for push_type in PUSH_TYPES:
        _ticket(db_session, device, ws, push_type)
    rows = db_session.execute(text(
        "SELECT push_type, created_at FROM mobile_push_tickets WHERE device_id = CAST(:d AS uuid)"),
        {"d": device}).all()
    assert sorted(row.push_type for row in rows) == sorted(PUSH_TYPES)
    assert all(row.created_at is not None for row in rows)


def test_an_unknown_push_type_is_refused(db_session, seed_workspace):
    from sqlalchemy.exc import IntegrityError

    ws, user = seed_workspace(), _seed_user(db_session)
    device = _seed_device(db_session, ws, user)
    with pytest.raises(IntegrityError, match=TICKET_CHECK):
        with db_session.begin_nested():
            _ticket(db_session, device, ws, "credit_low")


def test_deleting_the_device_deletes_its_tickets_only(db_session, seed_workspace):
    ws, user = seed_workspace(), _seed_user(db_session)
    gone, kept = _seed_device(db_session, ws, user), _seed_device(db_session, ws, user)
    _ticket(db_session, gone, ws)
    kept_ticket = _ticket(db_session, kept, ws, "review")
    db_session.execute(text("DELETE FROM mobile_devices WHERE id = CAST(:id AS uuid)"), {"id": gone})
    left = db_session.execute(text(
        "SELECT ticket_id FROM mobile_push_tickets WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws}).scalars().all()
    assert left == [kept_ticket]


def test_deleting_the_workspace_deletes_its_tickets_and_digest_posts(db_session, seed_workspace):
    ws, other_ws, user = seed_workspace(), seed_workspace(), _seed_user(db_session)
    _ticket(db_session, _seed_device(db_session, ws, user), ws, "auto_message")
    _claim_digest(db_session, ws, user, DAY)
    _ticket(db_session, _seed_device(db_session, other_ws, user), other_ws)
    _claim_digest(db_session, other_ws, user, DAY)

    db_session.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})

    assert (_count(db_session, "mobile_push_tickets", ws), _count(db_session, "mobile_digest_posts", ws)) == (0, 0)
    assert (_count(db_session, "mobile_push_tickets", other_ws),
            _count(db_session, "mobile_digest_posts", other_ws)) == (1, 1)


def test_deleting_the_person_deletes_their_digest_posts(db_session, seed_workspace):
    ws, user, other = seed_workspace(), _seed_user(db_session), _seed_user(db_session)
    _claim_digest(db_session, ws, user, DAY)
    _claim_digest(db_session, ws, other, DAY)
    db_session.execute(text("DELETE FROM users WHERE id = :id"), {"id": user})
    assert _count(db_session, "mobile_digest_posts", ws) == 1
