"""PRD-268 US-C19a (§6.5.3, §6.12.5, K6) — the wave's one migration: the suggestion push type and push_categories.

``prd268_w2_delivery`` widens ``ck_mobile_push_tickets_push_type`` to the fourth push type
and adds ``mobile_devices.push_categories`` (the optional push categories the phone's app
registered). Proven here:

* The revision chains onto the base head (``prd268_w1_proactive``) and only
  ``prd270_w1_cloud_sync`` (PRD-270 W1) chains onto it; the single head is pinned by
  test_prd209_alembic_single_head.py.
* The model's CHECK (built from ``PUSH_TYPES``) equals the revision's; the downgrade puts
  back the three-type CHECK.
* ``push_categories`` is declared the same way on both sides: JSONB, NOT NULL, default
  ``[]``, the model's last column (the revision appends it), and ``to_dict`` lists it.
* The upgrade touches only those two tables, each step ``IF EXISTS`` / ``IF NOT EXISTS``;
  the downgrade deletes only ``suggestion`` tickets.
* On Postgres (inside a rolled-back transaction): the widened CHECK accepts ``suggestion``
  and refuses another value; ``push_categories`` defaults to ``[]``; upgrading a
  ``create_all``-built database, twice, changes nothing; the downgrade drops the
  ``suggestion`` tickets, the fourth type and the column; and the upgrade from that
  pre-revision state gives back exactly the schema ``create_all`` builds.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
MIGRATION = VERSIONS / "prd268_w2_delivery.py"
REVISION = "prd268_w2_delivery"
BASE_HEAD = "prd268_w1_proactive"

PUSH_TYPES = ("needs_answer", "review", "auto_message", "suggestion")
TICKET_CHECK = "ck_mobile_push_tickets_push_type"
CATEGORIES_LINE = "push_categories JSONB NOT NULL DEFAULT '[]'"
TABLES = ("mobile_push_tickets", "mobile_devices")


def _assigned(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(getattr(t, "id", None) == name for t in targets):
            return ast.literal_eval(node.value)
    return None


def _statements(step: str) -> list[str]:
    spec = importlib.util.spec_from_file_location(f"{REVISION}_{step}", MIGRATION)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements: list[str] = []
    migration.op = SimpleNamespace(execute=statements.append)
    getattr(migration, step)()
    return statements


def _check_in(statement: str) -> str:
    (found,) = re.findall(rf"ADD CONSTRAINT {TICKET_CHECK} CHECK \((.*)\);$", statement)
    return found


def _model_check() -> str:
    from sqlalchemy import CheckConstraint

    from core.models.mobile_push import MobilePushTicket

    checks = {c.name: str(c.sqltext) for c in MobilePushTicket.__table__.constraints
              if isinstance(c, CheckConstraint)}
    return checks[TICKET_CHECK]


# --------------------------------------------------------------------------- the chain


def test_the_revision_chains_onto_the_base_head_and_prd270_w1_chains_onto_it():
    assert _assigned(MIGRATION, "revision") == REVISION
    assert _assigned(MIGRATION, "down_revision") == BASE_HEAD
    assert f"Revision ID: {REVISION}\nRevises: {BASE_HEAD}\n" in MIGRATION.read_text(encoding="utf-8")
    parents = {path.name: _assigned(path, "down_revision") for path in VERSIONS.glob("*.py")}

    def revises(parent: str) -> list[str]:
        return sorted(name for name, down in parents.items()
                      if down == parent or (isinstance(down, tuple) and parent in down))

    assert revises(BASE_HEAD) == [MIGRATION.name]
    assert revises(REVISION) == ["prd268_w3_follow_ups.py"]  # 2026-10-29: W3's revision (PRD-268 US-C11a)


# --------------------------------------------------------------------------- one shape


def test_the_models_check_is_built_from_push_types_and_equals_the_revisions():
    from core.models import mobile_push

    assert mobile_push.PUSH_TYPES == PUSH_TYPES
    assert _model_check() == mobile_push.PUSH_TYPE_CHECK
    upgrade, _ = _statements("upgrade")
    assert _check_in(upgrade) == _model_check()
    assert tuple(re.findall(r"'([a-z_]+)'", _check_in(upgrade))) == PUSH_TYPES


def test_the_downgrade_restores_the_three_type_check():
    _, restore, _ = _statements("downgrade")
    assert tuple(re.findall(r"'([a-z_]+)'", _check_in(restore))) == PUSH_TYPES[:3]
    assert _check_in(restore) == "push_type IN ('needs_answer', 'review', 'auto_message')"


def test_push_categories_is_declared_the_same_way_on_both_sides():
    from sqlalchemy.dialects import postgresql

    from core.models.mobile_devices import MobileDevice

    table = MobileDevice.__table__
    column = table.c.push_categories
    assert list(table.columns)[-1] is column  # the revision appends it: last on both sides
    assert column.type.compile(dialect=postgresql.dialect()) == "JSONB"
    assert column.nullable is False
    assert column.server_default.arg.text == "'[]'"
    assert column.default.is_callable  # default=list: a fresh list per row, never a shared one
    _, add_column = _statements("upgrade")
    assert add_column == f"ALTER TABLE mobile_devices ADD COLUMN IF NOT EXISTS {CATEGORIES_LINE};"


def test_to_dict_lists_push_categories_as_its_own_list():
    from core.models.mobile_devices import MobileDevice

    registered = ["suggestion"]
    device = MobileDevice(id=uuid.uuid4(), workspace_id=uuid.uuid4(), user_id=1, kind="clerk",
                          push_categories=registered)
    shown = device.to_dict()
    assert shown["push_categories"] == ["suggestion"]
    shown["push_categories"].append("tampered")
    assert registered == ["suggestion"]
    unset = MobileDevice(id=uuid.uuid4(), workspace_id=uuid.uuid4(), user_id=1, kind="clerk")
    assert unset.to_dict()["push_categories"] == []


# --------------------------------------------------------------------------- the steps


def test_the_upgrade_touches_only_two_tables_and_each_step_is_if_exists():
    statements = _statements("upgrade")
    assert len(statements) == 2
    widen, add_column = statements
    assert widen.startswith(f"ALTER TABLE mobile_push_tickets DROP CONSTRAINT IF EXISTS {TICKET_CHECK}, ")
    assert f", ADD CONSTRAINT {TICKET_CHECK} CHECK (" in widen
    assert add_column.startswith("ALTER TABLE mobile_devices ADD COLUMN IF NOT EXISTS ")
    for statement in statements:
        assert not re.search(r"\b(CREATE|UPDATE \w+ SET|DELETE FROM|INSERT INTO|DROP TABLE|DROP COLUMN)\b",
                             statement), statement
    altered = [re.match(r"ALTER TABLE (\w+) ", statement).group(1) for statement in statements]
    assert tuple(altered) == TABLES


def test_the_downgrade_deletes_only_suggestion_tickets_and_drops_the_column_if_present():
    delete, restore, drop = _statements("downgrade")
    assert delete == "DELETE FROM mobile_push_tickets WHERE push_type = 'suggestion';"
    assert restore.startswith(f"ALTER TABLE mobile_push_tickets DROP CONSTRAINT IF EXISTS {TICKET_CHECK}, ")
    assert drop == "ALTER TABLE mobile_devices DROP COLUMN IF EXISTS push_categories;"


# --------------------------------------------------------------------------- Postgres (CI)


def _seed_user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd268w2-{tag}", "e": f"prd268w2-{tag}@t.test"},
    ).scalar()


def _seed_device(db, workspace_id: str, user_id: int) -> str:
    device_id = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO mobile_devices (id, workspace_id, user_id, kind, status) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :u, 'clerk', 'active')"),
        {"id": device_id, "ws": workspace_id, "u": user_id})
    return device_id


def _ticket(db, device_id: str, workspace_id: str, push_type: str) -> str:
    ticket_id = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO mobile_push_tickets (ticket_id, device_id, workspace_id, push_type) "
        "VALUES (:t, CAST(:d AS uuid), CAST(:ws AS uuid), :p)"),
        {"t": ticket_id, "d": device_id, "ws": workspace_id, "p": push_type})
    return ticket_id


def _schema(db) -> tuple:
    """The two tables' columns (name, type, nullable, default) and the push-type CHECK's definition."""
    columns = db.execute(text(
        "SELECT table_name, column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns WHERE table_name IN ('mobile_push_tickets', 'mobile_devices') "
        "ORDER BY table_name, ordinal_position")).all()
    check = db.execute(text(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = :name"),
        {"name": TICKET_CHECK}).scalar()
    return tuple(tuple(row) for row in columns), check


def test_the_widened_check_accepts_suggestion_and_refuses_another_value(db_session, seed_workspace):
    from sqlalchemy.exc import IntegrityError

    ws, user = seed_workspace(), _seed_user(db_session)
    device = _seed_device(db_session, ws, user)
    kept = _ticket(db_session, device, ws, "suggestion")
    with pytest.raises(IntegrityError, match=TICKET_CHECK):
        with db_session.begin_nested():
            _ticket(db_session, device, ws, "credit_low")
    stored = db_session.execute(text(
        "SELECT push_type FROM mobile_push_tickets WHERE ticket_id = :t"), {"t": kept}).scalar()
    assert stored == "suggestion"


def test_push_categories_defaults_to_an_empty_list(db_session, seed_workspace):
    ws, user = seed_workspace(), _seed_user(db_session)
    device = _seed_device(db_session, ws, user)
    categories = db_session.execute(text(
        "SELECT push_categories FROM mobile_devices WHERE id = CAST(:id AS uuid)"), {"id": device}).scalar()
    assert categories == []


def test_upgrading_a_create_all_database_twice_changes_nothing(db_session, seed_workspace):
    """create_all built both tables from the models; the upgrade runs over them twice and the
    columns and the CHECK stay exactly as they were. Rolled back with db_session."""
    ws, user = seed_workspace(), _seed_user(db_session)
    device = _seed_device(db_session, ws, user)
    before = _schema(db_session)
    assert "'suggestion'" in before[1]
    for _ in range(2):
        for statement in _statements("upgrade"):
            db_session.execute(text(statement))
    assert _schema(db_session) == before
    categories = db_session.execute(text(
        "SELECT push_categories FROM mobile_devices WHERE id = CAST(:id AS uuid)"), {"id": device}).scalar()
    assert categories == []


def test_the_downgrade_drops_suggestion_tickets_the_fourth_type_and_the_column(db_session, seed_workspace):
    """Run inside db_session's transaction (Postgres DDL is transactional) and rolled back."""
    from sqlalchemy.exc import IntegrityError

    ws, user = seed_workspace(), _seed_user(db_session)
    device = _seed_device(db_session, ws, user)
    gone, kept = _ticket(db_session, device, ws, "suggestion"), _ticket(db_session, device, ws, "review")
    for statement in _statements("downgrade"):
        db_session.execute(text(statement))
    left = db_session.execute(text(
        "SELECT ticket_id FROM mobile_push_tickets WHERE device_id = CAST(:d AS uuid)"), {"d": device}).scalars().all()
    assert (gone not in left, left) == (True, [kept])
    with pytest.raises(IntegrityError, match=TICKET_CHECK):
        with db_session.begin_nested():
            _ticket(db_session, device, ws, "suggestion")
    columns = db_session.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'mobile_devices'")).scalars().all()
    assert "push_categories" not in columns


def test_the_upgrade_from_the_old_schema_gives_back_the_create_all_schema(db_session, seed_workspace):
    """The real path: the downgrade puts the database where prd268_w1_proactive left it, and the
    upgrade from there yields the very columns and CHECK the models build. Rolled back."""
    ws, user = seed_workspace(), _seed_user(db_session)
    device = _seed_device(db_session, ws, user)
    built = _schema(db_session)
    for statement in _statements("downgrade"):
        db_session.execute(text(statement))
    assert _schema(db_session) != built
    for statement in _statements("upgrade"):
        db_session.execute(text(statement))
    assert _schema(db_session) == built
    _ticket(db_session, device, ws, "suggestion")
    categories = db_session.execute(text(
        "SELECT push_categories FROM mobile_devices WHERE id = CAST(:id AS uuid)"), {"id": device}).scalar()
    assert categories == []
