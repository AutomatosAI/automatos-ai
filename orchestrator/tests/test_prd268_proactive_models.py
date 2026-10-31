"""PRD-268 US-C01 (§6.2.4, §6.9) — suggestions, runs and person settings: the models and the revision describe three tables.

``proactive_suggestions`` holds what a check-in found for one person, ``proactive_runs`` the
audit of every run and refusal, and ``proactive_person_settings`` each person's pause and
quiet hours. Proven here:

* ``core.models.proactive`` and the Alembic revision ``prd268_w1_proactive`` declare the same
  columns, in the same order, with the same types, nullability, primary keys, defaults and
  foreign keys (the workspace CASCADEs; persons and agents are plain integers); the same
  CHECKs, built from the module's status and outcome constants; and the same indexes,
  the partial unique on open rows included. A CHECK a later revision re-adds is read from
  the newest revision that sets it: the outcome CHECK gained ``posted`` in
  ``prd268_w3_follow_ups`` (PRD-268 W3, K20).
* The revision chains onto the old head, creates only what is absent (every CREATE is
  ``IF NOT EXISTS``), alters no other table, and its downgrade drops only these three.
* ``ProactiveSuggestion.to_dict`` gives every column as a JSON-ready value.
* On Postgres (inside a rolled-back transaction): the upgrade survives a ``create_all``
  database; a second open row with a person's subject_key is refused, while a hidden one
  and an open one may share it (and another person or workspace may hold it open); an
  unknown status or outcome is refused; deleting the workspace deletes its rows in all
  three tables and nobody else's.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import uuid
from datetime import datetime, time, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
MIGRATION = ORCH / "alembic" / "versions" / "prd268_w1_proactive.py"
# K20 (PRD-268 W3): a CHECK a later revision re-adds, and the newest revision that sets it.
NEWEST_CHECK = {"ck_proactive_runs_outcome": ORCH / "alembic" / "versions" / "prd268_w3_follow_ups.py"}
OLD_HEAD = "prd266_w2_push"
TABLES = ("proactive_suggestions", "proactive_runs", "proactive_person_settings")

SUGGESTION_COLUMNS = (
    "id", "workspace_id", "user_id", "source_type", "source_agent_id", "run_id", "title", "detail",
    "evidence", "proposed_action", "subject_key", "severity", "status", "occurrences", "last_seen_at",
    "started_ref", "decided_via", "hide_reason", "decided_at", "est_credits", "created_at",
    "delivered_at", "expires_at",
)
RUN_COLUMNS = (
    "id", "workspace_id", "agent_id", "kind", "window_since", "window_until", "outcome", "signals",
    "suggestions", "model", "tokens", "cost_usd", "error_class", "created_at",
)
PERSON_COLUMNS = ("workspace_id", "user_id", "paused", "quiet_start", "quiet_end", "timezone", "updated_at")
COLUMNS = dict(zip(TABLES, (SUGGESTION_COLUMNS, RUN_COLUMNS, PERSON_COLUMNS)))
STATUSES = ("open", "started", "hidden", "expired")
OUTCOMES = (
    "silent", "suggested", "error", "skipped_quiet", "skipped_cap",
    "skipped_credits", "skipped_paused", "skipped_off", "posted",
)
OPEN_UNIQUE = "uq_proactive_suggestions_open_subject"
INDEXES = {
    OPEN_UNIQUE: ("proactive_suggestions", ("workspace_id", "user_id", "subject_key"), True, "status = 'open'"),
    "ix_proactive_suggestions_workspace_status_created": (
        "proactive_suggestions", ("workspace_id", "status", "created_at"), False, None,
    ),
    "ix_proactive_runs_workspace_agent_created": (
        "proactive_runs", ("workspace_id", "agent_id", "created_at"), False, None,
    ),
}
CHECKS = {
    "proactive_suggestions": {"ck_proactive_suggestions_status": ("status", STATUSES)},
    "proactive_runs": {"ck_proactive_runs_outcome": ("outcome", OUTCOMES)},
    "proactive_person_settings": {},
}
KEY = "ticket:0846"
NOW = datetime(2026, 10, 28, 9, 30, tzinfo=timezone.utc)


def _tables() -> dict:
    from core.models.proactive import ProactivePersonSettings, ProactiveRun, ProactiveSuggestion

    models = (ProactiveSuggestion, ProactiveRun, ProactivePersonSettings)
    return {model.__tablename__: model.__table__ for model in models}


def _constants(migration: Path = MIGRATION) -> dict:
    """The revision's module-level literal constants, read with ast (never imported)."""
    found = {}
    for node in ast.parse(migration.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return found


def _ddl_lines(table: str) -> list[str]:
    create_sql = dict(_constants()["_TABLES"])[table]
    lines = [line.strip().rstrip(",") for line in create_sql.strip().splitlines()]
    assert lines[0] == f"CREATE TABLE IF NOT EXISTS {table} (" and lines[-1] == ");"
    return [line for line in lines[1:-1] if line]


def _ddl_columns(table: str) -> dict[str, str]:
    columns = [line for line in _ddl_lines(table) if not line.startswith(("CONSTRAINT ", "PRIMARY KEY ("))]
    by_name = {line.split(" ", 1)[0]: line for line in columns}
    assert len(by_name) == len(columns), "a column is declared twice in the revision"
    return by_name


def _ddl_primary_key(table: str) -> tuple:
    composite = [line for line in _ddl_lines(table) if line.startswith("PRIMARY KEY (")]
    if composite:
        return tuple(name.strip() for name in composite[0][len("PRIMARY KEY ("):-1].split(","))
    return tuple(name for name, line in _ddl_columns(table).items() if "PRIMARY KEY" in line)


def _pg_type(column) -> str:
    from sqlalchemy.dialects import postgresql

    return column.type.compile(dialect=postgresql.dialect())


def _default_sql(column) -> str | None:
    from sqlalchemy.dialects import postgresql

    default = column.server_default
    if default is None:
        return None
    if isinstance(default.arg, str):
        return f"'{default.arg}'"
    return str(default.arg.compile(dialect=postgresql.dialect()))


def _assert_column_matches(column, line: str) -> None:
    head = f"{column.name} {_pg_type(column)}"
    assert line == head or line.startswith(head + " "), line
    assert (not column.nullable) is ("NOT NULL" in line or "PRIMARY KEY" in line), line
    default = _default_sql(column)
    assert (default is not None) is (" DEFAULT " in line), line
    if default is not None:
        assert line.endswith(f" DEFAULT {default}") or f" DEFAULT {default} " in line, line
    fks = list(column.foreign_keys)
    assert len(fks) == line.count("REFERENCES"), line
    for fk in fks:
        table, target = fk.target_fullname.split(".")
        assert f"REFERENCES {table}({target}) ON DELETE {fk.ondelete}" in line, line


# --------------------------------------------------------------------------- one shape


@pytest.mark.parametrize("table_name", TABLES)
def test_the_models_and_the_revision_declare_every_column_in_one_order(table_name):
    table = _tables()[table_name]
    assert tuple(column.name for column in table.columns) == COLUMNS[table_name]
    assert tuple(_ddl_columns(table_name)) == COLUMNS[table_name]


@pytest.mark.parametrize("table_name", TABLES)
def test_model_columns_equal_ddl_columns(table_name):
    table, lines = _tables()[table_name], _ddl_columns(table_name)
    for column in table.columns:
        _assert_column_matches(column, lines[column.name])
    assert tuple(c.name for c in table.primary_key.columns) == _ddl_primary_key(table_name)


def test_the_keys_only_the_workspace_cascades_and_persons_are_plain_integers():
    tables = _tables()
    assert _ddl_primary_key("proactive_suggestions") == ("id",)
    assert _ddl_primary_key("proactive_runs") == ("id",)
    assert _ddl_primary_key("proactive_person_settings") == ("workspace_id", "user_id")
    for name, table in tables.items():
        fks = {fk.parent.name: (fk.target_fullname, fk.ondelete) for fk in table.foreign_keys}
        assert fks == {"workspace_id": ("workspaces.id", "CASCADE")}, name
    assert _pg_type(tables["proactive_suggestions"].c.user_id) == "INTEGER"
    assert _pg_type(tables["proactive_suggestions"].c.title) == "VARCHAR(120)"
    assert _pg_type(tables["proactive_suggestions"].c.detail) == "VARCHAR(600)"
    assert tables["proactive_suggestions"].c.occurrences.server_default is not None
    assert "occurrences INTEGER NOT NULL DEFAULT 1" in _ddl_lines("proactive_suggestions")
    assert "paused BOOLEAN NOT NULL DEFAULT false" in _ddl_lines("proactive_person_settings")


@pytest.mark.parametrize("table_name", TABLES)
def test_the_checks_are_built_from_the_module_constants_and_match_the_revision(table_name):
    from sqlalchemy import CheckConstraint

    checks = {c.name: str(c.sqltext) for c in _tables()[table_name].constraints if isinstance(c, CheckConstraint)}
    assert set(checks) == set(CHECKS[table_name])
    lines = _ddl_lines(table_name)
    assert len([line for line in lines if line.startswith("CONSTRAINT ")]) == len(checks)
    for name, (column, values) in CHECKS[table_name].items():
        assert checks[name].startswith(f"{column} IN (")
        assert tuple(re.findall(r"'([a-z_]+)'", checks[name])) == values
        if name in NEWEST_CHECK:  # K20: re-added whole by the newest revision that sets it
            readd = f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {name}, ADD CONSTRAINT {name} CHECK ({checks[name]});"
            assert readd in _constants(NEWEST_CHECK[name])["_UPGRADE"]
        else:
            assert f"CONSTRAINT {name} CHECK ({checks[name]})" in lines


def test_the_status_and_outcome_constants():
    from core.models import proactive

    assert proactive.SUGGESTION_STATUSES == STATUSES
    assert (proactive.SUGGESTION_OPEN, proactive.SUGGESTION_STARTED,
            proactive.SUGGESTION_HIDDEN, proactive.SUGGESTION_EXPIRED) == STATUSES
    assert proactive.RUN_OUTCOMES == OUTCOMES
    assert (proactive.RUN_SILENT, proactive.RUN_SUGGESTED, proactive.RUN_ERROR, proactive.RUN_SKIPPED_QUIET,
            proactive.RUN_SKIPPED_CAP, proactive.RUN_SKIPPED_CREDITS, proactive.RUN_SKIPPED_PAUSED,
            proactive.RUN_SKIPPED_OFF, proactive.RUN_POSTED) == OUTCOMES


def test_model_indexes_equal_ddl_indexes():
    expected, found = [], {}
    for table_name, table in _tables().items():
        for index in table.indexes:
            where = index.dialect_options["postgresql"]["where"]
            where = None if where is None else str(where)
            found[index.name] = (table_name, tuple(c.name for c in index.columns), index.unique, where)
            kind = "UNIQUE INDEX" if index.unique else "INDEX"
            tail = f" WHERE {where};" if where else ";"
            columns = ", ".join(c.name for c in index.columns)
            expected.append(f"CREATE {kind} IF NOT EXISTS {index.name} ON {table_name} ({columns}){tail}")
    assert found == INDEXES
    assert sorted(expected) == sorted(_constants()["_INDEXES"])
    unique = next(i for i in _tables()["proactive_suggestions"].indexes if i.name == OPEN_UNIQUE)
    # The same predicate under SQLite, so create_all in the unit tests enforces what Postgres does.
    assert str(unique.dialect_options["sqlite"]["where"]) == "status = 'open'"


def test_the_models_are_registered_for_create_all():
    import core.models as models
    from core.database.base import Base

    assert Base.metadata.tables["proactive_suggestions"] is models.ProactiveSuggestion.__table__
    assert Base.metadata.tables["proactive_runs"] is models.ProactiveRun.__table__
    assert Base.metadata.tables["proactive_person_settings"] is models.ProactivePersonSettings.__table__


def test_to_dict_gives_every_column_json_ready():
    import core.models  # noqa: F401  (registers every mapper before one is built)
    from core.models.proactive import ProactiveSuggestion

    sid, ws, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = ProactiveSuggestion(
        id=sid, workspace_id=ws, user_id=7, source_type="check_in", source_agent_id=578, run_id=run,
        title="Ticket #0846 is blocked", detail="It has waited two days.",
        evidence=[{"type": "ticket", "id": "0846"}],
        proposed_action={"kind": "chat", "payload": {}}, subject_key=KEY, severity="today",
        status="started", occurrences=2, last_seen_at=NOW, started_ref={"board_task_id": 12},
        decided_via="web", decided_at=NOW, est_credits=3, created_at=NOW, expires_at=NOW,
    )
    out = row.to_dict()
    assert tuple(out) == SUGGESTION_COLUMNS
    assert (out["id"], out["workspace_id"], out["run_id"]) == (str(sid), str(ws), str(run))
    assert out["last_seen_at"] == out["decided_at"] == out["created_at"] == NOW.isoformat()
    assert out["delivered_at"] is None and out["hide_reason"] is None
    assert out["evidence"] == [{"type": "ticket", "id": "0846"}]
    assert out["started_ref"] == {"board_task_id": 12}
    json.dumps(out)  # JSON-ready


# --------------------------------------------------------------------------- the revision


def test_the_revision_chains_onto_the_old_head():
    constants = _constants()
    assert constants["revision"] == "prd268_w1_proactive"
    assert constants["down_revision"] == OLD_HEAD
    assert "Revision ID: prd268_w1_proactive\nRevises: prd266_w2_push\n" in MIGRATION.read_text()


def _statements(step: str) -> list[str]:
    spec = importlib.util.spec_from_file_location(f"prd268_w1_proactive_{step}", MIGRATION)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements: list[str] = []
    migration.op = SimpleNamespace(execute=statements.append)
    getattr(migration, step)()
    return statements


def test_every_create_is_if_not_exists_and_no_other_table_is_touched():
    statements = _statements("upgrade")
    assert len(statements) == len(TABLES) + len(INDEXES)
    for statement in statements:
        sql = " ".join(statement.split())
        assert re.match(r"CREATE (UNIQUE )?(TABLE|INDEX) IF NOT EXISTS ", sql), sql
        assert not re.search(r"\b(ALTER TABLE|DROP|UPDATE \w+ SET|DELETE FROM|INSERT INTO)\b", sql), sql
    created = [re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", s) for s in statements]
    assert [m.group(1) for m in created if m] == list(TABLES)
    indexed = {re.search(r" ON (\w+) \(", s).group(1) for s in statements if " INDEX " in s}
    assert indexed <= set(TABLES)
    assert "ALTER TABLE" not in MIGRATION.read_text()
    assert "op.create_table" not in MIGRATION.read_text()


def test_the_downgrade_drops_only_these_three_tables():
    assert _statements("downgrade") == [f"DROP TABLE IF EXISTS {name};" for name in reversed(TABLES)]


# --------------------------------------------------------------------------- Postgres (CI)


def _suggest(db, ws: str, user_id: int = 7, key: str = KEY, status: str = "open") -> str:
    sid = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO proactive_suggestions (id, workspace_id, user_id, source_type, title, "
        "proposed_action, subject_key, status) VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :u, "
        "'check_in', 'Ticket #0846 is blocked', CAST(:a AS jsonb), :k, :s)"),
        {"id": sid, "ws": ws, "u": user_id, "a": json.dumps({"kind": "chat", "payload": {}}),
         "k": key, "s": status})
    return sid


def _run(db, ws: str, outcome: str = "silent") -> str:
    rid = str(uuid.uuid4())
    db.execute(text(
        "INSERT INTO proactive_runs (id, workspace_id, agent_id, kind, outcome) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), 578, 'agent', :o)"),
        {"id": rid, "ws": ws, "o": outcome})
    return rid


def _person(db, ws: str, user_id: int = 7) -> None:
    db.execute(text(
        "INSERT INTO proactive_person_settings (workspace_id, user_id, quiet_start, quiet_end, timezone) "
        "VALUES (CAST(:ws AS uuid), :u, :qs, :qe, 'Europe/Dublin')"),
        {"ws": ws, "u": user_id, "qs": time(22, 0), "qe": time(7, 0)})


def _count(db, table: str, ws: str) -> int:
    assert table in TABLES
    return db.execute(
        text(f"SELECT count(*) FROM {table} WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws},
    ).scalar()


def _index_names(db) -> set[str]:
    return set(db.execute(text(
        "SELECT indexname FROM pg_indexes WHERE tablename IN "
        "('proactive_suggestions', 'proactive_runs', 'proactive_person_settings')")).scalars())


def test_the_upgrade_keeps_create_all_tables_and_adds_only_the_missing_index(db_session):
    """create_all built the three tables; the upgrade runs over them and puts back the dropped
    partial unique. Everything runs inside db_session's transaction and is rolled back."""
    assert set(INDEXES) <= _index_names(db_session)
    db_session.execute(text(f"DROP INDEX {OPEN_UNIQUE}"))
    assert OPEN_UNIQUE not in _index_names(db_session)

    for statement in _statements("upgrade"):
        db_session.execute(text(statement))

    assert set(INDEXES) <= _index_names(db_session)
    for table, columns in COLUMNS.items():
        found = db_session.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = :t "
            "ORDER BY ordinal_position"), {"t": table}).scalars().all()
        assert tuple(found) == columns


def test_a_second_open_row_with_the_same_subject_is_refused(db_session, seed_workspace):
    from sqlalchemy.exc import IntegrityError

    ws = seed_workspace()
    _suggest(db_session, ws)
    with pytest.raises(IntegrityError, match=OPEN_UNIQUE):
        with db_session.begin_nested():
            _suggest(db_session, ws)
    assert _count(db_session, "proactive_suggestions", ws) == 1


def test_one_hidden_and_one_open_row_may_share_a_subject(db_session, seed_workspace):
    ws, other_ws = seed_workspace(), seed_workspace()
    _suggest(db_session, ws, status="hidden")
    _suggest(db_session, ws, status="hidden")
    _suggest(db_session, ws)
    _suggest(db_session, ws, user_id=8)  # another person in the workspace
    _suggest(db_session, other_ws)  # the same person and key in another workspace
    open_rows = db_session.execute(text(
        "SELECT count(*) FROM proactive_suggestions WHERE workspace_id = CAST(:ws AS uuid) "
        "AND user_id = 7 AND subject_key = :k AND status = 'open'"), {"ws": ws, "k": KEY}).scalar()
    assert (open_rows, _count(db_session, "proactive_suggestions", ws)) == (1, 4)


def test_a_new_suggestion_takes_its_defaults(db_session, seed_workspace):
    ws = seed_workspace()
    sid = _suggest(db_session, ws)
    row = db_session.execute(text(
        "SELECT detail, evidence, severity, status, occurrences, last_seen_at, created_at "
        "FROM proactive_suggestions WHERE id = CAST(:id AS uuid)"), {"id": sid}).one()
    assert (row.detail, row.evidence, row.severity, row.status, row.occurrences) == ("", [], "can_wait", "open", 1)
    assert row.last_seen_at is not None and row.created_at is not None


@pytest.mark.parametrize(("insert", "check"), [
    (lambda db, ws: _suggest(db, ws, status="dismissed"), "ck_proactive_suggestions_status"),
    (lambda db, ws: _run(db, ws, outcome="delivered"), "ck_proactive_runs_outcome"),
])
def test_an_unknown_status_or_outcome_is_refused(db_session, seed_workspace, insert, check):
    from sqlalchemy.exc import IntegrityError

    ws = seed_workspace()
    with pytest.raises(IntegrityError, match=check):
        with db_session.begin_nested():
            insert(db_session, ws)


def test_one_settings_row_per_workspace_and_person(db_session, seed_workspace):
    from sqlalchemy.exc import IntegrityError

    ws = seed_workspace()
    _person(db_session, ws)
    _person(db_session, ws, user_id=8)
    with pytest.raises(IntegrityError):
        with db_session.begin_nested():
            _person(db_session, ws)
    paused = db_session.execute(text(
        "SELECT paused FROM proactive_person_settings WHERE workspace_id = CAST(:ws AS uuid) AND user_id = 7"),
        {"ws": ws}).scalar()
    assert paused is False


def test_deleting_the_workspace_deletes_its_rows_in_all_three_tables(db_session, seed_workspace):
    ws, other_ws = seed_workspace(), seed_workspace()
    for workspace in (ws, other_ws):
        _suggest(db_session, workspace)
        _run(db_session, workspace)
        _person(db_session, workspace)

    db_session.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})

    assert [_count(db_session, table, ws) for table in TABLES] == [0, 0, 0]
    assert [_count(db_session, table, other_ws) for table in TABLES] == [1, 1, 1]
