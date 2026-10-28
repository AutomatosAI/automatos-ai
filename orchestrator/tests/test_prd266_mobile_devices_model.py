"""PRD-266 US-002 — the mobile_devices table: the model and the revision describe one table.

One row per phone (or paired browser) a person uses with one workspace (§4.2). Proven here:

* ``core.models.mobile_devices.MobileDevice`` and the Alembic revision
  ``prd266_mobile_devices`` declare the same columns, in the same order, with the same
  nullability (``install_id`` is null on a pending row), foreign keys (the workspace and
  the person CASCADE), uniques, CHECKs, defaults and indexes. A column a later revision
  appends (``push_categories``, ``prd268_w2_delivery``, PRD-268 K19) is read from it.
* ``to_dict()`` never carries a code or token hash, the push key or the push token.
* The revision chains onto the old head, creates only what is absent (a ``create_all``
  database survives it and gains only a missing index), alters no other table, and its
  downgrade drops only this one.
* On Postgres the table refuses what the code must never write: a clerk row holding a
  token hash (§8), an unknown kind, status or platform, a second row for one phone, and
  a pairing code, token hash or push token on two devices. Deleting the person or the workspace
  deletes their devices.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
MIGRATION = ORCH / "alembic" / "versions" / "prd266_mobile_devices.py"
OLD_HEAD = "clerk_user_relinks_20261019"
# The later revisions that append a column (ALTER TABLE … ADD COLUMN IF NOT EXISTS), oldest first.
ADDED_BY = (ORCH / "alembic" / "versions" / "prd268_w2_delivery.py",)
ADD_COLUMN = re.compile(r"ALTER TABLE mobile_devices ADD COLUMN IF NOT EXISTS ((\w+) [^;\"]+);")

COLUMNS = (
    "id", "workspace_id", "user_id", "kind", "install_id", "name", "platform", "app_version",
    "locale", "timezone", "digest_time", "status", "pairing_code_hash", "pairing_expires_at",
    "token_hash", "token_rotated_at", "prev_token_hash", "prev_token_valid_until",
    "expo_push_token", "push_enabled", "push_disabled_reason", "push_key", "last_seen_at",
    "paired_at", "revoked_at", "revoked_by", "events", "created_at", "updated_at", "push_categories",
)
NOT_NULL = {"id", "workspace_id", "user_id", "kind", "status", "push_enabled", "events",
            "created_at", "updated_at", "push_categories"}
UNIQUES = {
    "uq_mobile_devices_workspace_user_install": ["workspace_id", "user_id", "install_id"],
    "uq_mobile_devices_token_hash": ["token_hash"],
    "uq_mobile_devices_expo_push_token": ["expo_push_token"],
    "uq_mobile_devices_pairing_code_hash": ["pairing_code_hash"],
}
INDEXES = {
    "ix_mobile_devices_workspace_status": ["workspace_id", "status"],
    "ix_mobile_devices_prev_token_hash": ["prev_token_hash"],
}
SECRETS = ("pairing_code_hash", "token_hash", "prev_token_hash", "push_key")
CODE_HASH = "a" * 64
TOKEN_HASH = "b" * 64
PREV_HASH = "c" * 64
PUSH_KEY = "push-key-prd266-test-never-shown"
PUSH_TOKEN = "ExponentPushToken[prd266-test-token]"
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _table():
    from core.models.mobile_devices import MobileDevice

    return MobileDevice.__table__


def _ddl() -> str:
    (ddl,) = re.findall(r"CREATE TABLE IF NOT EXISTS mobile_devices \((.*?)\n\s*\)\n", MIGRATION.read_text(), re.S)
    return ddl


def _ddl_lines() -> list[str]:
    return [line.strip().rstrip(",") for line in _ddl().strip().splitlines()]


def _ddl_columns() -> dict[str, str]:
    created = {line.split()[0]: line for line in _ddl_lines() if not line.startswith("CONSTRAINT")}
    appended = {name: line for path in ADDED_BY for line, name in ADD_COLUMN.findall(path.read_text())}
    return {**created, **appended}


def _statements(step: str) -> list[str]:
    spec = importlib.util.spec_from_file_location(f"prd266_mobile_devices_{step}", MIGRATION)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements: list[str] = []
    migration.op = SimpleNamespace(execute=statements.append)
    getattr(migration, step)()
    return statements


# --------------------------------------------------------------------------- one shape


def test_the_model_and_the_revision_declare_every_column_in_one_order():
    assert tuple(column.name for column in _table().columns) == COLUMNS
    assert tuple(_ddl_columns()) == COLUMNS


def test_only_identity_state_and_bookkeeping_are_not_null_and_install_id_may_be():
    table = _table()
    assert {column.name for column in table.columns if not column.nullable} == NOT_NULL
    assert table.c.install_id.nullable is True
    ddl_not_null = {name for name, line in _ddl_columns().items() if "NOT NULL" in line or "PRIMARY KEY" in line}
    assert ddl_not_null == NOT_NULL


def test_the_workspace_and_the_person_cascade_and_the_revoker_becomes_null():
    table = _table()
    fks = {fk.parent.name: (fk.target_fullname, fk.ondelete) for fk in table.foreign_keys}
    assert fks == {
        "workspace_id": ("workspaces.id", "CASCADE"),
        "user_id": ("users.id", "CASCADE"),
        "revoked_by": ("users.id", "SET NULL"),
    }
    columns = _ddl_columns()
    assert "REFERENCES workspaces(id) ON DELETE CASCADE" in columns["workspace_id"]
    assert "REFERENCES users(id) ON DELETE CASCADE" in columns["user_id"]
    assert "REFERENCES users(id) ON DELETE SET NULL" in columns["revoked_by"]


def test_the_four_uniques_are_named_alike_on_both_sides():
    from sqlalchemy import UniqueConstraint

    table = _table()
    uniques = {c.name: [col.name for col in c.columns] for c in table.constraints if isinstance(c, UniqueConstraint)}
    assert uniques == UNIQUES
    assert not any(column.unique for column in table.columns)
    for name, columns in UNIQUES.items():
        assert f"CONSTRAINT {name} UNIQUE ({', '.join(columns)})" in _ddl_lines()


def test_the_two_indexes_are_named_alike_on_both_sides():
    table = _table()
    assert {index.name: [c.name for c in index.columns] for index in table.indexes} == INDEXES
    src = MIGRATION.read_text()
    for name, columns in INDEXES.items():
        assert f"CREATE INDEX IF NOT EXISTS {name} " in src
        assert f"ON mobile_devices ({', '.join(columns)});" in src


def test_the_checks_hold_kind_status_and_platform_and_keep_tokens_off_clerk_rows():
    from sqlalchemy import CheckConstraint

    from core.models import mobile_devices as model

    checks = {c.name: str(c.sqltext) for c in _table().constraints if isinstance(c, CheckConstraint)}
    assert set(checks) == {"ck_mobile_devices_kind", "ck_mobile_devices_status",
                           "ck_mobile_devices_platform", "ck_mobile_devices_clerk_holds_no_token"}
    assert set(re.findall(r"'([a-z]+)'", checks["ck_mobile_devices_kind"])) == {"paired", "clerk"}
    assert {k.value for k in model.MobileDeviceKind} == {"paired", "clerk"}
    assert set(re.findall(r"'([a-z]+)'", checks["ck_mobile_devices_status"])) == {"pending", "active", "revoked"}
    assert {s.value for s in model.MobileDeviceStatus} == {"pending", "active", "revoked"}
    assert model.MOBILE_DEVICE_PLATFORMS == ("ios", "android", "web")
    assert checks["ck_mobile_devices_platform"].startswith("platform IS NULL OR ")
    for hash_column in ("token_hash", "prev_token_hash", "pairing_code_hash"):
        assert f"{hash_column} IS NULL" in checks["ck_mobile_devices_clerk_holds_no_token"]
    for name, sqltext in checks.items():
        assert f"CONSTRAINT {name} CHECK ({sqltext})" in _ddl_lines()


def test_a_new_device_is_pending_with_push_off_and_no_events_on_both_sides():
    from sqlalchemy.dialects import postgresql

    table = _table()
    assert table.c.status.server_default.arg == "pending"
    assert str(table.c.push_enabled.server_default.arg.compile(dialect=postgresql.dialect())) == "false"
    assert table.c.events.server_default.arg.text == "'[]'"
    columns = _ddl_columns()
    assert "DEFAULT 'pending'" in columns["status"]
    assert "DEFAULT false" in columns["push_enabled"]
    assert "DEFAULT '[]'" in columns["events"]


def test_the_model_is_registered_for_create_all():
    import core.models as models
    from core.database.base import Base

    assert models.MobileDevice.__tablename__ == "mobile_devices"
    assert Base.metadata.tables["mobile_devices"] is models.MobileDevice.__table__
    assert {models.MobileDeviceKind.PAIRED.value, models.MobileDeviceStatus.ACTIVE.value} == {"paired", "active"}


# --------------------------------------------------------------------------- to_dict


def test_to_dict_never_carries_a_hash_the_push_key_or_the_push_token():
    from core.models.mobile_devices import SECRET_COLUMNS, MobileDevice

    assert SECRET_COLUMNS == SECRETS
    device = MobileDevice(
        id=uuid.uuid4(), workspace_id=uuid.uuid4(), user_id=7, kind="paired", status="active",
        install_id="install-1", platform="ios", pairing_code_hash=CODE_HASH, token_hash=TOKEN_HASH,
        prev_token_hash=PREV_HASH, prev_token_valid_until=NOW, push_key=PUSH_KEY,
        expo_push_token=PUSH_TOKEN, push_enabled=True, events=[{"event": "enrolled"}], paired_at=NOW,
    )
    shown = device.to_dict()
    for hidden in (*SECRETS, "expo_push_token"):
        assert hidden not in shown
    body = json.dumps(shown)
    for value in (CODE_HASH, TOKEN_HASH, PREV_HASH, PUSH_KEY, PUSH_TOKEN):
        assert value not in body
    assert shown["push_registered"] is True and shown["push_enabled"] is True
    assert (shown["status"], shown["install_id"], shown["paired_at"]) == ("active", "install-1", NOW.isoformat())


def test_to_dict_returns_its_own_events_list():
    from core.models.mobile_devices import MobileDevice

    events = [{"event": "enrolled"}]
    shown = MobileDevice(id=uuid.uuid4(), workspace_id=uuid.uuid4(), user_id=1, kind="clerk", events=events).to_dict()
    shown["events"].append({"event": "tampered"})
    assert events == [{"event": "enrolled"}]
    assert shown["push_registered"] is False


# --------------------------------------------------------------------------- the revision


def _assigned(name: str):
    tree = ast.parse(MIGRATION.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{MIGRATION.name} assigns no {name}")


def test_the_revision_chains_onto_the_old_head():
    assert _assigned("revision") == "prd266_mobile_devices"
    assert _assigned("down_revision") == OLD_HEAD


def test_every_create_is_if_not_exists_and_no_other_table_is_touched():
    statements = _statements("upgrade")
    assert len(statements) == 3
    for statement in statements:
        sql = " ".join(statement.split())
        assert re.match(r"CREATE (TABLE|INDEX) IF NOT EXISTS ", sql), sql
        assert not re.search(r"\b(ALTER TABLE|DROP|UPDATE \w+ SET|DELETE FROM|INSERT INTO)\b", sql), sql
    for statement in statements[1:]:
        assert re.search(r"\bON mobile_devices \(", statement)
    assert "ALTER TABLE" not in MIGRATION.read_text()


def test_the_downgrade_drops_only_this_table():
    assert _statements("downgrade") == ["DROP TABLE IF EXISTS mobile_devices;"]


# --------------------------------------------------------------------------- Postgres (CI)


def _seed_user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-{tag}", "e": f"prd266-{tag}@t.test"},
    ).scalar()


def _device(db, workspace_id, user_id, **fields):
    from core.models.mobile_devices import MobileDevice

    row = MobileDevice(workspace_id=uuid.UUID(str(workspace_id)), user_id=user_id,
                       kind=fields.pop("kind", "paired"), **fields)
    db.add(row)
    db.flush()
    return row


def _refused(db, constraint: str, workspace_id, user_id, **fields) -> None:
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError, match=constraint):
        with db.begin_nested():
            _device(db, workspace_id, user_id, **fields)


def _index_names(db) -> set[str]:
    return set(db.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = 'mobile_devices'")).scalars())


def test_the_upgrade_keeps_a_create_all_table_and_adds_only_its_missing_indexes(db_session):
    """create_all built the table; the upgrade runs over it and puts back the dropped
    indexes. Everything runs inside db_session's transaction and is rolled back."""
    assert set(INDEXES) <= _index_names(db_session)
    for name in INDEXES:
        db_session.execute(text(f"DROP INDEX {name}"))
    assert not set(INDEXES) & _index_names(db_session)

    for statement in _statements("upgrade"):
        db_session.execute(text(statement))

    assert set(INDEXES) <= _index_names(db_session)
    columns = db_session.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'mobile_devices' "
        "ORDER BY ordinal_position")).scalars().all()
    assert tuple(columns) == COLUMNS


def test_a_new_row_is_pending_with_push_off_and_an_empty_audit(db_session, seed_workspace):
    ws, user, device_id = seed_workspace(), _seed_user(db_session), str(uuid.uuid4())
    db_session.execute(text(
        "INSERT INTO mobile_devices (id, workspace_id, user_id, kind) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :u, 'paired')"),
        {"id": device_id, "ws": ws, "u": user})
    status, push_enabled, events = db_session.execute(text(
        "SELECT status, push_enabled, events FROM mobile_devices WHERE id = CAST(:id AS uuid)"),
        {"id": device_id}).one()
    assert (status, push_enabled, events) == ("pending", False, [])


def test_a_clerk_row_never_holds_a_code_or_token_hash(db_session, seed_workspace):
    ws, user = seed_workspace(), _seed_user(db_session)
    for hash_column in ("token_hash", "prev_token_hash", "pairing_code_hash"):
        _refused(db_session, "ck_mobile_devices_clerk_holds_no_token", ws, user,
                 kind="clerk", install_id=f"clerk-{hash_column}", **{hash_column: TOKEN_HASH})
    clerk = _device(db_session, ws, user, kind="clerk", install_id="clerk-phone", status="active",
                    expo_push_token=PUSH_TOKEN)
    paired = _device(db_session, ws, user, kind="paired", install_id="paired-phone", status="active",
                     token_hash=TOKEN_HASH, prev_token_hash=PREV_HASH,
                     prev_token_valid_until=NOW + timedelta(minutes=10))
    assert clerk.token_hash is None and paired.token_hash == TOKEN_HASH


@pytest.mark.parametrize("bad, constraint", [
    ({"kind": "oidc"}, "ck_mobile_devices_kind"),
    ({"status": "paired"}, "ck_mobile_devices_status"),
    ({"platform": "windows"}, "ck_mobile_devices_platform"),
])
def test_an_unknown_kind_status_or_platform_is_refused(db_session, seed_workspace, bad, constraint):
    _refused(db_session, constraint, seed_workspace(), _seed_user(db_session), **bad)


def test_one_row_per_phone_and_person_while_pending_rows_have_no_phone(db_session, seed_workspace):
    ws, user, other = seed_workspace(), _seed_user(db_session), _seed_user(db_session)
    first, second = _device(db_session, ws, user), _device(db_session, ws, user)
    assert first.install_id is None and second.install_id is None
    _device(db_session, ws, user, install_id="phone-1", platform="web")
    _refused(db_session, "uq_mobile_devices_workspace_user_install", ws, user, install_id="phone-1")
    _device(db_session, ws, other, install_id="phone-1")
    _device(db_session, seed_workspace(), user, install_id="phone-1")


def test_a_code_a_token_hash_or_a_push_token_belongs_to_one_device(db_session, seed_workspace):
    ws, user = seed_workspace(), _seed_user(db_session)
    _device(db_session, ws, user, install_id="one", token_hash=TOKEN_HASH, expo_push_token=PUSH_TOKEN)
    _refused(db_session, "uq_mobile_devices_token_hash", ws, user, install_id="two", token_hash=TOKEN_HASH)
    _refused(db_session, "uq_mobile_devices_expo_push_token", ws, user, install_id="three",
             expo_push_token=PUSH_TOKEN)
    _device(db_session, ws, user, pairing_code_hash=CODE_HASH)
    _refused(db_session, "uq_mobile_devices_pairing_code_hash", seed_workspace(), _seed_user(db_session),
             pairing_code_hash=CODE_HASH)


def _devices_of(db, user_id: int) -> int:
    return db.execute(text("SELECT count(*) FROM mobile_devices WHERE user_id = :u"), {"u": user_id}).scalar()


def test_deleting_the_person_or_the_workspace_deletes_their_devices(db_session, seed_workspace):
    ws, user = seed_workspace(), _seed_user(db_session)
    revoker = _seed_user(db_session)
    kept = _device(db_session, ws, user, install_id="kept", status="revoked", revoked_at=NOW, revoked_by=revoker)
    db_session.execute(text("DELETE FROM users WHERE id = :id"), {"id": revoker})
    assert db_session.execute(text("SELECT revoked_by FROM mobile_devices WHERE id = CAST(:id AS uuid)"),
                              {"id": str(kept.id)}).scalar() is None

    db_session.execute(text("DELETE FROM users WHERE id = :id"), {"id": user})
    assert _devices_of(db_session, user) == 0

    other_ws, other_user = seed_workspace(), _seed_user(db_session)
    _device(db_session, other_ws, other_user, install_id="by-workspace")
    db_session.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": other_ws})
    assert _devices_of(db_session, other_user) == 0
