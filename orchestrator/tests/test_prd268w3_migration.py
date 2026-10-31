"""PRD-268 US-C11a (§6.3, §6.11.5, K6) — the wave's one migration: follow-up rows, the posted outcome, proactive_calibrations.

``prd268_w3_follow_ups`` widens ``ck_scheduled_task_deliver_as`` to ``follow_up``, adds
``agent_scheduled_tasks.for_user_id`` with its partial index, widens
``ck_proactive_runs_outcome`` with ``posted``, and creates ``proactive_calibrations``.
Proven here:

* The revision chains onto the base head (``prd268_w2_delivery``) and only
  ``prd270_w1_cloud_sync`` (PRD-270 W1) chains onto it; the single head is pinned by
  test_prd209_alembic_single_head.py.
* It does exactly K6's four things (five statements), each ``IF EXISTS`` / ``IF NOT
  EXISTS``, and touches no other table; the downgrade deletes only the follow-up rows and
  the posted runs, restores both CHECKs and drops the index, the column and the table.
* The widened ``deliver_as`` CHECK is ``DELIVERY_MODES`` plus ``DELIVER_FOLLOW_UP``, and
  ``DELIVERY_MODES`` itself is unchanged (create_task can never make a follow-up, K7).
* The model's outcome CHECK (built from ``RUN_OUTCOMES``) equals the revision's.
* ``proactive_calibrations`` is declared the same way in the model and the revision:
  columns in one order, types, nullability, defaults, the primary key, the workspace
  CASCADE, the status CHECK and the per-person unique; ``to_dict`` gives every column.
* On Postgres (inside a rolled-back transaction): ``follow_up`` is accepted and another
  value refused; ``for_user_id`` is nullable; ``posted`` is accepted; a second calibration
  for one person and workspace is refused; deleting the workspace deletes its calibrations
  and nobody else's; upgrading a ``create_all``-built database changes nothing; and the
  upgrade from the pre-revision state gives back exactly the schema ``create_all`` builds.

``agent_scheduled_tasks`` is made by migrations only (no model), so the CI database, which
``create_all`` builds, lacks it. Each test makes it as a temporary table with the columns
and the CHECK the earlier revisions leave (prd77, calendar_scheduled_board_tasks,
prd205_auto_speaks); a temporary table shadows any permanent one of the same name, so the
tests never touch a real one, and it is rolled back with the rest.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
MIGRATION = VERSIONS / "prd268_w3_follow_ups.py"
REVISION = "prd268_w3_follow_ups"
BASE_HEAD = "prd268_w2_delivery"

DELIVER_AS_CHECK = "ck_scheduled_task_deliver_as"
OUTCOME_CHECK = "ck_proactive_runs_outcome"
STATUS_CHECK = "ck_proactive_calibrations_status"
PERSON_UNIQUE = "uq_proactive_calibrations_person"
FOLLOW_UPS_INDEX = "ix_scheduled_tasks_follow_ups"
DELIVERIES = ("chat", "board_task", "follow_up")
W1_OUTCOMES = (
    "silent", "suggested", "error", "skipped_quiet", "skipped_cap",
    "skipped_credits", "skipped_paused", "skipped_off",
)
CALIBRATION_STATUSES = ("declined", "open", "answered")
CALIBRATION_COLUMNS = (
    "id", "workspace_id", "user_id", "status", "calibration", "rejected_lines", "accepted_lines",
    "chat_id", "message_id", "consented_at", "decided_at", "created_at",
)
TOUCHED = ("agent_scheduled_tasks", "proactive_runs", "proactive_calibrations")
CREATE_ALL_TABLES = ("proactive_runs", "proactive_calibrations")
NOW = datetime(2026, 10, 29, 15, 10, tzinfo=timezone.utc)

# The shape prd77 + calendar_scheduled_board_tasks + prd205_auto_speaks leave, before this
# revision. A temporary table may not reference a permanent one, so no foreign keys.
SCHEDULED_TASKS_BEFORE = """
    CREATE TEMP TABLE agent_scheduled_tasks (
        id SERIAL PRIMARY KEY, workspace_id UUID NOT NULL, created_by_agent_id INTEGER,
        target_agent_id INTEGER, task_type VARCHAR(20) NOT NULL DEFAULT 'one_shot',
        description TEXT NOT NULL, schedule VARCHAR(100) NOT NULL, max_runs INTEGER,
        run_count INTEGER NOT NULL DEFAULT 0, status VARCHAR(20) NOT NULL DEFAULT 'active',
        last_run_at TIMESTAMPTZ, next_run_at TIMESTAMPTZ, last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_by_user_id VARCHAR(255), deliver_as VARCHAR(20) NOT NULL DEFAULT 'chat', payload JSONB,
        origin_chat_id UUID,
        CONSTRAINT ck_scheduled_task_deliver_as CHECK (deliver_as IN ('chat', 'board_task')))
"""


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


def _readded(statement: str, table: str, check: str) -> str:
    """The CHECK a one-statement DROP IF EXISTS + ADD re-adds, or a failed assertion."""
    prefix = f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {check}, ADD CONSTRAINT {check} CHECK ("
    assert statement.startswith(prefix) and statement.endswith(");"), statement
    return statement[len(prefix):-2]


def _values(check_sql: str) -> tuple:
    return tuple(re.findall(r"'([a-z_]+)'", check_sql))


def _model_checks(model) -> dict:
    from sqlalchemy import CheckConstraint

    return {c.name: str(c.sqltext) for c in model.__table__.constraints if isinstance(c, CheckConstraint)}


def _calibration_ddl() -> list[str]:
    create_sql = _statements("upgrade")[4]
    lines = [line.strip().rstrip(",") for line in create_sql.strip().splitlines()]
    assert lines[0] == "CREATE TABLE IF NOT EXISTS proactive_calibrations (" and lines[-1] == ");"
    return [line for line in lines[1:-1] if line]


def _pg(clause) -> str:
    from sqlalchemy.dialects import postgresql

    return str(clause.compile(dialect=postgresql.dialect()))


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
    assert revises(REVISION) == ["prd270_w1_cloud_sync.py"]  # 2026-10-31: W1's revision (PRD-270 US-270-00)


# --------------------------------------------------------------------------- the steps


def test_the_upgrade_does_exactly_k6s_four_things():
    widen, add_column, index, outcome, create = _statements("upgrade")
    assert _values(_readded(widen, "agent_scheduled_tasks", DELIVER_AS_CHECK)) == DELIVERIES
    assert add_column == "ALTER TABLE agent_scheduled_tasks ADD COLUMN IF NOT EXISTS for_user_id INTEGER;"
    assert index == (
        f"CREATE INDEX IF NOT EXISTS {FOLLOW_UPS_INDEX} ON agent_scheduled_tasks "
        "(workspace_id, for_user_id, created_at) WHERE deliver_as = 'follow_up';"
    )
    assert _values(_readded(outcome, "proactive_runs", OUTCOME_CHECK)) == W1_OUTCOMES + ("posted",)
    assert create.strip().startswith("CREATE TABLE IF NOT EXISTS proactive_calibrations (")


def test_the_upgrade_touches_only_its_three_tables_and_changes_no_row():
    statements = _statements("upgrade")
    named = set()
    for statement in statements:
        sql = " ".join(statement.split())
        assert not re.search(r"\b(UPDATE \w+ SET|DELETE FROM|INSERT INTO|DROP TABLE|DROP COLUMN|DROP INDEX)\b",
                             sql), sql
        assert re.match(r"(ALTER TABLE \w+ (DROP CONSTRAINT IF EXISTS|ADD COLUMN IF NOT EXISTS) "
                        r"|CREATE (TABLE|INDEX) IF NOT EXISTS )", sql), sql
        named |= set(re.findall(r"(?:ALTER TABLE|CREATE TABLE IF NOT EXISTS|CREATE INDEX IF NOT EXISTS \w+ ON) (\w+)",
                                sql))
    assert named == set(TOUCHED)


def test_the_downgrade_deletes_the_transient_rows_restores_both_checks_and_drops_the_rest():
    follow_ups, posted, deliver_as, outcome, index, column, table = _statements("downgrade")
    assert follow_ups == "DELETE FROM agent_scheduled_tasks WHERE deliver_as = 'follow_up';"
    assert posted == "DELETE FROM proactive_runs WHERE outcome = 'posted';"
    assert _readded(deliver_as, "agent_scheduled_tasks", DELIVER_AS_CHECK) == "deliver_as IN ('chat', 'board_task')"
    assert _values(_readded(outcome, "proactive_runs", OUTCOME_CHECK)) == W1_OUTCOMES
    assert index == f"DROP INDEX IF EXISTS {FOLLOW_UPS_INDEX};"
    assert column == "ALTER TABLE agent_scheduled_tasks DROP COLUMN IF EXISTS for_user_id;"
    assert table == "DROP TABLE IF EXISTS proactive_calibrations;"


def test_the_deliver_as_check_is_the_delivery_modes_and_the_follow_up_and_the_modes_are_unchanged():
    from services import scheduled_task_steps as steps

    assert steps.DELIVER_FOLLOW_UP == "follow_up"
    # create_task and platform_schedule_task accept only DELIVERY_MODES: neither can make a follow-up.
    assert steps.DELIVERY_MODES == ("chat", "board_task")
    assert steps.DELIVER_FOLLOW_UP not in steps.DELIVERY_MODES
    assert steps.request_error("one_shot", steps.DELIVER_FOLLOW_UP, 1, None, 1, {}) == (
        "deliver_as must be one of: chat, board_task"
    )
    widen = _statements("upgrade")[0]
    assert _values(_readded(widen, "agent_scheduled_tasks", DELIVER_AS_CHECK)) == (
        steps.DELIVERY_MODES + (steps.DELIVER_FOLLOW_UP,)
    )


# --------------------------------------------------------------------------- one shape


def test_the_models_outcome_check_equals_the_revisions():
    from core.models import proactive

    assert proactive.RUN_POSTED == "posted"
    assert proactive.RUN_OUTCOMES == W1_OUTCOMES + (proactive.RUN_POSTED,)
    model_check = _model_checks(proactive.ProactiveRun)[OUTCOME_CHECK]
    assert _readded(_statements("upgrade")[3], "proactive_runs", OUTCOME_CHECK) == model_check


def test_the_calibration_model_and_the_revision_declare_every_column_in_one_order():
    from core.models.proactive import ProactiveCalibration

    table = ProactiveCalibration.__table__
    columns = [line for line in _calibration_ddl() if not line.startswith("CONSTRAINT ")]
    assert tuple(column.name for column in table.columns) == CALIBRATION_COLUMNS
    assert tuple(line.split(" ", 1)[0] for line in columns) == CALIBRATION_COLUMNS
    for column, line in zip(table.columns, columns):
        head = f"{column.name} {_pg(column.type)}"
        assert line == head or line.startswith(head + " "), line
        assert (not column.nullable) is ("NOT NULL" in line or "PRIMARY KEY" in line), line
        default = column.server_default
        if default is None:
            assert " DEFAULT " not in line, line
        else:
            sql = f"'{default.arg}'" if isinstance(default.arg, str) else _pg(default.arg)
            assert line.endswith(f" DEFAULT {sql}"), line


def test_the_calibration_keys_check_and_unique_match_the_revision():
    from sqlalchemy import UniqueConstraint

    from core.models import proactive

    table = proactive.ProactiveCalibration.__table__
    lines = _calibration_ddl()
    assert tuple(c.name for c in table.primary_key.columns) == ("id",)
    assert "id UUID PRIMARY KEY" in lines
    fks = {fk.parent.name: (fk.target_fullname, fk.ondelete) for fk in table.foreign_keys}
    assert fks == {"workspace_id": ("workspaces.id", "CASCADE")}  # user, chat and message are plain values
    assert "workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE" in lines
    assert proactive.CALIBRATION_STATUSES == CALIBRATION_STATUSES
    assert (proactive.CALIBRATION_DECLINED, proactive.CALIBRATION_OPEN,
            proactive.CALIBRATION_ANSWERED) == CALIBRATION_STATUSES
    check = _model_checks(proactive.ProactiveCalibration)[STATUS_CHECK]
    assert _values(check) == CALIBRATION_STATUSES
    uniques = {c.name: tuple(col.name for col in c.columns) for c in table.constraints
               if isinstance(c, UniqueConstraint)}
    assert uniques == {PERSON_UNIQUE: ("workspace_id", "user_id")}
    assert [line for line in lines if line.startswith("CONSTRAINT ")] == [
        f"CONSTRAINT {STATUS_CHECK} CHECK ({check})",
        f"CONSTRAINT {PERSON_UNIQUE} UNIQUE (workspace_id, user_id)",
    ]
    assert not table.indexes  # the unique leads with workspace_id: the CASCADE never scans


def test_the_calibration_model_is_registered_for_create_all():
    import core.models as models
    from core.database.base import Base

    assert Base.metadata.tables["proactive_calibrations"] is models.ProactiveCalibration.__table__


def test_calibration_to_dict_gives_every_column_json_ready():
    import core.models  # noqa: F401  (registers every mapper before one is built)
    from core.models.proactive import ProactiveCalibration

    cid, ws, chat, message = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    said = {"lines": [{"id": "l1", "text": "You run the Lisbon launch", "sources": {"deliverables": 6}}]}
    row = ProactiveCalibration(
        id=cid, workspace_id=ws, user_id=7, status="answered", calibration=said,
        rejected_lines=["l2"], accepted_lines=["l1"], chat_id=chat, message_id=message,
        consented_at=NOW, decided_at=NOW, created_at=NOW,
    )
    out = row.to_dict()
    assert tuple(out) == CALIBRATION_COLUMNS
    assert (out["id"], out["workspace_id"], out["chat_id"], out["message_id"]) == (
        str(cid), str(ws), str(chat), str(message))
    assert out["consented_at"] == out["decided_at"] == out["created_at"] == NOW.isoformat()
    assert (out["calibration"], out["rejected_lines"], out["accepted_lines"]) == (said, ["l2"], ["l1"])
    out["accepted_lines"].append("tampered")
    out["calibration"]["lines"][0]["text"] = "tampered"
    assert row.accepted_lines == ["l1"]
    assert row.calibration["lines"][0]["text"] == "You run the Lisbon launch"
    json.dumps(out)  # JSON-ready
    empty = ProactiveCalibration(id=cid, workspace_id=ws, user_id=7).to_dict()
    assert (empty["calibration"], empty["rejected_lines"], empty["chat_id"]) == ({}, [], None)


# --------------------------------------------------------------------------- Postgres (CI)


@pytest.fixture
def db(db_session):
    """The rolled-back session, with agent_scheduled_tasks as the earlier revisions leave it
    (a temporary table: it shadows any permanent one)."""
    db_session.execute(text(SCHEDULED_TASKS_BEFORE))
    return db_session


def _upgrade(db) -> None:
    for statement in _statements("upgrade"):
        db.execute(text(statement))


def _schedule(db, ws: str, deliver_as: str = "follow_up", for_user_id: int | None = 7) -> int:
    return db.execute(text(
        "INSERT INTO agent_scheduled_tasks (workspace_id, task_type, description, schedule, next_run_at, "
        "deliver_as, for_user_id) VALUES (CAST(:ws AS uuid), 'one_shot', 'Check the Lisbon card', :at, :at, "
        ":d, :u) RETURNING id"),
        {"ws": ws, "at": NOW, "d": deliver_as, "u": for_user_id}).scalar()


def _run(db, ws: str, outcome: str) -> str:
    rid = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO proactive_runs (id, workspace_id, agent_id, kind, outcome) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), 578, 'follow_up', :o)"),
        {"id": rid, "ws": ws, "o": outcome})
    return rid


def _calibrate(db, ws: str, user_id: int = 7) -> str:
    cid = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO proactive_calibrations (id, workspace_id, user_id) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :u)"), {"id": cid, "ws": ws, "u": user_id})
    return cid


def _calibrations(db, ws: str) -> int:
    return db.execute(text(
        "SELECT count(*) FROM proactive_calibrations WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws}).scalar()


def _schema(db, tables: tuple) -> dict:
    """Each table's columns (name, type, nullable, default), constraints and indexes, by definition."""
    found = {}
    for table in tables:
        named = {"t": table}
        columns = db.execute(text(
            "SELECT attname, format_type(atttypid, atttypmod), attnotnull, pg_get_expr(adbin, adrelid) "
            "FROM pg_attribute LEFT JOIN pg_attrdef ON adrelid = attrelid AND adnum = attnum "
            "WHERE attrelid = CAST(:t AS regclass) AND attnum > 0 AND NOT attisdropped ORDER BY attnum"),
            named).all()
        constraints = db.execute(text(
            "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid = CAST(:t AS regclass) "
            "ORDER BY conname"), named).all()
        indexes = db.execute(text(
            "SELECT pg_get_indexdef(indexrelid) FROM pg_index WHERE indrelid = CAST(:t AS regclass) ORDER BY 1"),
            named).scalars().all()
        found[table] = (tuple(map(tuple, columns)), tuple(map(tuple, constraints)), tuple(indexes))
    return found


def test_a_follow_up_row_is_accepted_and_another_delivery_refused(db, seed_workspace):
    from sqlalchemy.exc import IntegrityError

    ws = seed_workspace()
    _upgrade(db)
    kept = _schedule(db, ws)
    with pytest.raises(IntegrityError, match=DELIVER_AS_CHECK):
        with db.begin_nested():
            _schedule(db, ws, deliver_as="email")
    row = db.execute(text(
        "SELECT deliver_as, for_user_id FROM agent_scheduled_tasks WHERE id = :id"), {"id": kept}).one()
    assert tuple(row) == ("follow_up", 7)


def test_for_user_id_is_nullable_and_an_existing_kind_of_row_still_fits(db, seed_workspace):
    ws = seed_workspace()
    _upgrade(db)
    chat = _schedule(db, ws, deliver_as="chat", for_user_id=None)
    board = _schedule(db, ws, deliver_as="board_task", for_user_id=None)
    found = db.execute(text(
        "SELECT deliver_as, for_user_id FROM agent_scheduled_tasks WHERE id IN (:a, :b) ORDER BY id"),
        {"a": chat, "b": board}).all()
    assert [tuple(row) for row in found] == [("chat", None), ("board_task", None)]


def test_posted_is_accepted_and_an_unknown_outcome_refused(db, seed_workspace):
    from sqlalchemy.exc import IntegrityError

    ws = seed_workspace()
    _upgrade(db)
    rid = _run(db, ws, "posted")
    with pytest.raises(IntegrityError, match=OUTCOME_CHECK):
        with db.begin_nested():
            _run(db, ws, "delivered")
    outcome = db.execute(text("SELECT outcome FROM proactive_runs WHERE id = CAST(:id AS uuid)"), {"id": rid}).scalar()
    assert outcome == "posted"


def test_one_calibration_per_person_and_workspace(db, seed_workspace):
    from sqlalchemy.exc import IntegrityError

    ws, other_ws = seed_workspace(), seed_workspace()
    _upgrade(db)
    first = _calibrate(db, ws)
    _calibrate(db, ws, user_id=8)  # another person in the workspace
    _calibrate(db, other_ws)  # the same person in another workspace
    with pytest.raises(IntegrityError, match=PERSON_UNIQUE):
        with db.begin_nested():
            _calibrate(db, ws)
    row = db.execute(text(
        "SELECT status, calibration, rejected_lines, accepted_lines, created_at FROM proactive_calibrations "
        "WHERE id = CAST(:id AS uuid)"), {"id": first}).one()
    assert (row.status, row.calibration, row.rejected_lines, row.accepted_lines) == ("open", {}, [], [])
    assert row.created_at is not None
    with pytest.raises(IntegrityError, match=STATUS_CHECK):
        with db.begin_nested():
            db.execute(text("UPDATE proactive_calibrations SET status = 'pending' WHERE id = CAST(:id AS uuid)"),
                       {"id": first})


def test_deleting_the_workspace_deletes_its_calibrations_and_nobody_elses(db, seed_workspace):
    ws, other_ws = seed_workspace(), seed_workspace()
    _upgrade(db)
    _calibrate(db, ws)
    _calibrate(db, ws, user_id=8)
    _calibrate(db, other_ws)
    db.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    assert (_calibrations(db, ws), _calibrations(db, other_ws)) == (0, 1)


def test_upgrading_a_create_all_database_changes_nothing(db, seed_workspace):
    """create_all built proactive_runs and proactive_calibrations from the models: the upgrade
    leaves them exactly as they were, and a second upgrade changes nothing anywhere."""
    ws = seed_workspace()
    built = _schema(db, CREATE_ALL_TABLES)
    assert "'posted'" in dict(built["proactive_runs"][1])[OUTCOME_CHECK]
    _upgrade(db)
    assert _schema(db, CREATE_ALL_TABLES) == built
    once = _schema(db, TOUCHED)
    _upgrade(db)
    assert _schema(db, TOUCHED) == once
    _calibrate(db, ws)
    _schedule(db, ws)


def test_the_upgrade_from_the_old_schema_gives_back_the_create_all_schema(db, seed_workspace):
    """The real path: the downgrade puts the database where prd268_w2_delivery left it (no
    follow-up, no posted run, no calibrations, agent_scheduled_tasks exactly as it was), and
    the upgrade from there yields the very columns, constraints and indexes the models build
    (the first upgrade leaves the create_all tables as they were). Rolled back."""
    ws = seed_workspace()
    before = _schema(db, ("agent_scheduled_tasks",))
    _upgrade(db)
    built = _schema(db, TOUCHED)
    _schedule(db, ws)
    kept = _schedule(db, ws, deliver_as="chat", for_user_id=None)
    _run(db, ws, "posted")
    silent = _run(db, ws, "silent")
    for statement in _statements("downgrade"):
        db.execute(text(statement))
    assert db.execute(text("SELECT to_regclass('proactive_calibrations')")).scalar() is None
    assert _schema(db, ("agent_scheduled_tasks",)) == before
    left = db.execute(text("SELECT id FROM agent_scheduled_tasks WHERE workspace_id = CAST(:ws AS uuid)"),
                      {"ws": ws}).scalars().all()
    runs = db.execute(text("SELECT CAST(id AS text) FROM proactive_runs WHERE workspace_id = CAST(:ws AS uuid)"),
                      {"ws": ws}).scalars().all()
    assert (left, runs) == ([kept], [silent])
    _upgrade(db)
    assert _schema(db, TOUCHED) == built
