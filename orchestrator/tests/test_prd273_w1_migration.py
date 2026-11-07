"""PRD-273 W1 — prd273_w1_workspace_move_jobs: the workspace_move_jobs table.

* The revision chains onto ``prd270_w2_cloud_bundles`` and sits under the one head.
* Its upgrade is one CREATE TABLE IF NOT EXISTS and two CREATE INDEX IF NOT EXISTS, writing no
  row; its downgrade drops the table and nothing else.
* The model (``core.models.workspace_move.WorkspaceMoveJob``) and the DDL declare the same
  columns, nullability, CHECKs and indexes.
* On Postgres, each case in a new empty schema inside a rolled-back transaction: a
  ``create_all``-built table and a migration-built one are the same; the upgrade over a
  ``create_all``-built table changes nothing; a second queued or running export of a
  workspace is refused while a finished one is not; an unknown status is refused.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy.exc import IntegrityError

import core.models  # noqa: F401  (registers the tables the foreign keys name)
from core.database.base import Base
from core.models.workspace_move import (
    ACTIVE_WHERE, JOB_KINDS, JOB_STATUSES, KIND_CHECK, STATUS_CHECK, WorkspaceMoveJob,
)

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
MIGRATION = VERSIONS / "prd273_w1_workspace_move_jobs.py"
REVISION, BASE_HEAD = "prd273_w1_workspace_move_jobs", "prd270_w2_cloud_bundles"
TABLE = "workspace_move_jobs"
COLUMNS = (
    "id", "workspace_id", "kind", "status", "categories", "encrypted", "requested_by", "storage_key", "file_name",
    "size_bytes", "summary", "error", "created_at", "started_at", "finished_at", "expires_at", "downloaded_at",
)
ROW_WRITES = re.compile(r"\b(ALTER TABLE|DELETE FROM|TRUNCATE|UPDATE \w+ SET|INSERT INTO)\b")


def _constants() -> dict:
    found = {}
    for node in ast.parse(MIGRATION.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return found


def _load():
    spec = importlib.util.spec_from_file_location(f"{MIGRATION.stem}_{uuid.uuid4().hex[:8]}", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _statements(step: str) -> list:
    module = _load()
    statements: list = []
    module.op = SimpleNamespace(execute=statements.append)
    getattr(module, step)()
    return statements


def _revisions() -> dict:
    revisions = {}
    for path in VERSIONS.glob("*.py"):
        values = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("revision", "down_revision"):
                    values[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(values.get("revision"), str):
            revisions[values["revision"]] = values.get("down_revision")
    return revisions


def _parents(down) -> list:
    return list(down) if isinstance(down, (list, tuple)) else [down] if down else []


def test_the_revision_chains_onto_the_base_head_and_sits_under_the_one_head():
    constants = _constants()
    assert (constants["revision"], constants["down_revision"]) == (REVISION, BASE_HEAD)
    revisions = _revisions()
    assert [rev for rev, down in revisions.items() if BASE_HEAD in _parents(down)] == [REVISION]
    heads = set(revisions) - {parent for down in revisions.values() for parent in _parents(down)}
    assert len(heads) == 1
    seen, todo = set(), list(heads)
    while todo:
        current = todo.pop()
        if current not in seen:
            seen.add(current)
            todo.extend(_parents(revisions.get(current)))
    assert REVISION in seen


def test_the_upgrade_creates_one_table_and_two_indexes_and_the_downgrade_drops_it():
    constants = _constants()
    assert [name for name, _sql in constants["_TABLES"]] == [TABLE]
    upgrade = _statements("upgrade")
    assert upgrade == [constants["_TABLES"][0][1], *constants["_INDEXES"]]
    assert upgrade[0].strip().startswith(f"CREATE TABLE IF NOT EXISTS {TABLE} (")
    assert all(statement.startswith(("CREATE UNIQUE INDEX IF NOT EXISTS", "CREATE INDEX IF NOT EXISTS"))
               for statement in constants["_INDEXES"])
    assert not any(ROW_WRITES.search(statement) for statement in upgrade)
    assert _statements("downgrade") == [f"DROP TABLE IF EXISTS {TABLE};"]


def test_the_model_and_the_ddl_declare_the_same_table():
    table = WorkspaceMoveJob.__table__
    assert tuple(table.columns.keys()) == COLUMNS
    ddl = _constants()["_TABLES"][0][1]
    lines = [line.strip().rstrip(",") for line in ddl.strip().splitlines()[1:-1]]
    columns = [line for line in lines if not line.startswith("CONSTRAINT ")]
    assert tuple(line.split(" ", 1)[0] for line in columns) == COLUMNS
    for line in columns:
        name = line.split(" ", 1)[0]
        assert ("NOT NULL" in line or "PRIMARY KEY" in line) == (not table.c[name].nullable), name
    checks = {c.name: str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint)}
    assert checks == {"ck_workspace_move_jobs_kind": KIND_CHECK, "ck_workspace_move_jobs_status": STATUS_CHECK}
    assert f"CONSTRAINT ck_workspace_move_jobs_kind CHECK ({KIND_CHECK})" in lines
    assert f"CONSTRAINT ck_workspace_move_jobs_status CHECK ({STATUS_CHECK})" in lines
    assert KIND_CHECK == "kind IN (" + ", ".join(f"'{kind}'" for kind in JOB_KINDS) + ")"
    assert STATUS_CHECK == "status IN (" + ", ".join(f"'{status}'" for status in JOB_STATUSES) + ")"
    active = next(index for index in table.indexes if index.unique)
    assert active.name == "uq_workspace_move_jobs_active" and tuple(active.columns.keys()) == ("workspace_id", "kind")
    assert str(active.dialect_options["postgresql"]["where"]) == ACTIVE_WHERE
    assert f"WHERE {ACTIVE_WHERE};" in _constants()["_INDEXES"][0]
    assert Base.metadata.tables[TABLE] is table


# --------------------------------------------------------------------------- Postgres (CI)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def pg(engine):
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            yield conn
        finally:
            trans.rollback()


def _empty_schema(conn) -> None:
    name = f"prd273_w1_{uuid.uuid4().hex[:12]}"
    conn.execute(sa.text(f"CREATE SCHEMA {name}"))
    conn.execute(sa.text(f"SET LOCAL search_path TO {name}, public"))


def _run(conn) -> None:
    with Operations.context(MigrationContext.configure(conn)):
        _load().upgrade()


_SNAPSHOT = (
    "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns "
    "WHERE table_schema = current_schema() AND table_name = :t",
    "SELECT indexname, regexp_replace(indexdef, ' ON [a-z0-9_]+\\.', ' ON ') FROM pg_indexes "
    "WHERE schemaname = current_schema() AND tablename = :t",
    "SELECT con.conname, pg_get_constraintdef(con.oid) FROM pg_constraint con JOIN pg_class cls ON cls.oid = con.conrelid "
    "JOIN pg_namespace ns ON ns.oid = cls.relnamespace WHERE ns.nspname = current_schema() AND cls.relname = :t",
)


def _schema(conn) -> list:
    return [sorted(tuple(str(value) for value in row) for row in conn.execute(sa.text(sql), {"t": TABLE}))
            for sql in _SNAPSHOT]


def test_create_all_and_the_migration_build_the_same_table(pg):
    _empty_schema(pg)
    Base.metadata.create_all(pg, tables=[WorkspaceMoveJob.__table__], checkfirst=False)
    by_model = _schema(pg)
    _run(pg)  # over a create_all-built table: nothing changes
    assert _schema(pg) == by_model
    pg.execute(sa.text("DROP TABLE workspace_move_jobs"))
    _run(pg)
    assert _schema(pg) == by_model


def _workspace(conn) -> str:
    ws = str(uuid.uuid4())
    conn.execute(sa.text("INSERT INTO public.workspaces (id, name) VALUES (:id, 'prd273-migration')"), {"id": ws})
    return ws


def _insert(conn, ws: str, status: str) -> None:
    conn.execute(sa.text("INSERT INTO workspace_move_jobs (id, workspace_id, kind, status) VALUES (:id, :ws, 'export', :s)"),
                 {"id": str(uuid.uuid4()), "ws": ws, "s": status})


def test_one_active_export_per_workspace_and_known_statuses_only(pg):
    _empty_schema(pg)
    _run(pg)
    ws = _workspace(pg)
    _insert(pg, ws, "downloaded")
    _insert(pg, ws, "queued")
    with pytest.raises(IntegrityError):
        with pg.begin_nested():
            _insert(pg, ws, "running")
    with pytest.raises(IntegrityError):
        with pg.begin_nested():
            _insert(pg, ws, "paused")
    _insert(pg, _workspace(pg), "running")
