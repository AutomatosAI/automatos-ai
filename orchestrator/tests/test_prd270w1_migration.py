"""PRD-270 W1 US-270-00 (K6) — prd270_w1_cloud_sync: theme_packages.source and marketplace_cloud_sync.

The wave's one revision does exactly two things: a guarded DO block (``_ADD_SOURCE``) that adds
``theme_packages.source`` (nullable, no default: every existing row reads NULL, this install's
own) and its CHECK (``'cloud'`` only on a Marketplace row), and ``marketplace_cloud_sync``, the
Sync's one status row. Proven here:

* the revision chains onto the base's head (``prd268_w2_delivery``) and sits under the one head;
* its upgrade is those two statements and nothing else, and its downgrade drops the table, the
  CHECK and the column and deletes no row;
* the model (``ThemePackage.source`` and ``ck_theme_packages_source``;
  ``core.models.marketplace_cloud.MarketplaceCloudSync``) and the DDL declare the same columns,
  types, nullability, defaults and CHECKs;
* on Postgres (inside a rolled-back transaction, each case in a new empty schema; only
  ``workspaces`` and ``users`` in public are shared: the foreign keys lock them while the
  transaction lasts, and a test workspace row is written there and rolled back): a row from
  before the revision reads ``source`` NULL; ``'cloud'``
  is accepted on a Marketplace row and refused on a workspace row, and any other value is
  refused, both where ``create_all`` built the table and where the migrations did; a second sync
  row is refused; on a ``create_all``-built schema the upgrade changes nothing; the migrations
  build the model's schema; the downgrade deletes no ``theme_packages`` row.
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
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

import core.models  # noqa: F401  (registers every table the foreign keys name)
from core.database.base import Base
from core.models.marketplace_cloud import MARKETPLACE_CLOUD_SYNC_ROW_ID, MarketplaceCloudSync
from core.models.themes import THEME_SOURCE_CLOUD, ThemePackage

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
MIGRATION = VERSIONS / "prd270_w1_cloud_sync.py"
REVISION = "prd270_w1_cloud_sync"
BASE_HEAD = "prd268_w2_delivery"
# The revisions that built theme_packages before this one, in chain order.
EARLIER = (VERSIONS / "prd261_themes.py", VERSIONS / "prd261_revocation_feed.py")
SOURCE_CHECK = "ck_theme_packages_source"
SOURCE_BODY = "source IS NULL OR (source = 'cloud' AND owner_type = 'marketplace')"
SYNC_CHECK = "ck_marketplace_cloud_sync_single_row"
SYNC_COLUMNS = (
    "id", "last_attempt_at", "last_success_at", "feed_generated_at", "last_error",
    "new_count", "updated_count", "unchanged_count", "refused_count", "updated_at",
)
COUNTS = ("new_count", "updated_count", "unchanged_count", "refused_count")
TABLES = ("theme_packages", "marketplace_cloud_sync")
ALTER = re.compile(r"ALTER TABLE (\w+) ADD (COLUMN|CONSTRAINT) (\w+)")


def _constants() -> dict:
    """The revision's module-level literal constants, read with ast (never imported)."""
    found = {}
    for node in ast.parse(MIGRATION.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return found


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.stem}_{uuid.uuid4().hex[:8]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _statements(step: str) -> list[str]:
    """What the revision's upgrade or downgrade executes, in order, without a database."""
    migration = _load(MIGRATION)
    statements: list[str] = []
    migration.op = SimpleNamespace(execute=statements.append)
    getattr(migration, step)()
    return statements


def _pg_type(column: sa.Column) -> str:
    return column.type.compile(dialect=postgresql.dialect())


def _default_sql(column: sa.Column) -> str | None:
    default = column.server_default
    if default is None:
        return None
    if isinstance(default.arg, str):
        return f"'{default.arg}'"
    return str(default.arg.compile(dialect=postgresql.dialect()))


def _sync_ddl_lines() -> list[str]:
    create_sql = dict(_constants()["_TABLES"])["marketplace_cloud_sync"]
    lines = [line.strip().rstrip(",") for line in create_sql.strip().splitlines()]
    assert lines[0] == "CREATE TABLE IF NOT EXISTS marketplace_cloud_sync (" and lines[-1] == ");"
    return [line for line in lines[1:-1] if line]


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


# --------------------------------------------------------------------------- the revision


def test_the_revision_chains_onto_the_base_head_and_sits_under_the_one_head():
    constants = _constants()
    assert (constants["revision"], constants["down_revision"]) == (REVISION, BASE_HEAD)
    source = MIGRATION.read_text(encoding="utf-8")
    assert re.search(rf'Revision ID: {REVISION}\nRevises: {BASE_HEAD}\nCreate Date: [0-9-]+\n"""', source)
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


def test_the_upgrade_is_exactly_k6_two_steps():
    constants = _constants()
    statements = _statements("upgrade")
    assert statements == [constants["_ADD_SOURCE"], dict(constants["_TABLES"])["marketplace_cloud_sync"]]
    assert [name for name, _sql in constants["_TABLES"]] == ["marketplace_cloud_sync"]
    alters = constants["_ADD_SOURCE"]
    assert ALTER.findall(alters) == [("theme_packages", "COLUMN", "source"), ("theme_packages", "CONSTRAINT", SOURCE_CHECK)]
    assert "ALTER TABLE theme_packages ADD COLUMN source VARCHAR(16);" in alters
    assert f"ALTER TABLE theme_packages ADD CONSTRAINT {SOURCE_CHECK} CHECK ({SOURCE_BODY});" in alters
    # Each ALTER sits behind its own IF NOT EXISTS, asked of the table the name resolves to.
    assert alters.count("ALTER TABLE ") == alters.count("IF NOT EXISTS (") == 2
    assert "attrelid = 'theme_packages'::regclass AND attname = 'source' AND NOT attisdropped" in alters
    assert f"conrelid = 'theme_packages'::regclass AND conname = '{SOURCE_CHECK}'" in alters
    assert re.match(r"\s*CREATE TABLE IF NOT EXISTS marketplace_cloud_sync \(", statements[1])
    for sql in statements:
        assert not re.search(r"\b(DROP|DELETE FROM|TRUNCATE|UPDATE \w+ SET|INSERT INTO)\b", sql), sql
    assert "op.create_table" not in MIGRATION.read_text(encoding="utf-8")


def test_the_downgrade_drops_the_table_the_check_and_the_column_and_deletes_no_row():
    assert _statements("downgrade") == [
        "DROP TABLE IF EXISTS marketplace_cloud_sync;",
        f"ALTER TABLE theme_packages DROP CONSTRAINT IF EXISTS {SOURCE_CHECK};",
        "ALTER TABLE theme_packages DROP COLUMN IF EXISTS source;",
    ]
    assert not re.search(r"\b(DELETE FROM|TRUNCATE|UPDATE)\b", MIGRATION.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- model == DDL


def test_theme_packages_source_is_declared_the_same_in_the_model_and_the_ddl():
    column = ThemePackage.__table__.c.source
    assert THEME_SOURCE_CLOUD == "cloud"
    assert (_pg_type(column), column.nullable) == ("VARCHAR(16)", True)
    assert column.default is None and column.server_default is None  # every existing row reads NULL
    added = re.findall(r"ALTER TABLE theme_packages ADD COLUMN (.+);", _constants()["_ADD_SOURCE"])
    assert added == [f"source {_pg_type(column)}"]
    checks = {c.name: str(c.sqltext) for c in ThemePackage.__table__.constraints if isinstance(c, sa.CheckConstraint)}
    assert checks[SOURCE_CHECK] == SOURCE_BODY
    ddl = re.findall(rf"ADD CONSTRAINT {SOURCE_CHECK} CHECK \((.+)\);", _constants()["_ADD_SOURCE"])
    assert ddl == [checks[SOURCE_CHECK]]


def test_marketplace_cloud_sync_model_equals_its_ddl():
    table = MarketplaceCloudSync.__table__
    lines = _sync_ddl_lines()
    columns = {line.split(" ", 1)[0]: line for line in lines if not line.startswith("CONSTRAINT ")}
    assert tuple(table.columns.keys()) == tuple(columns) == SYNC_COLUMNS
    for column in table.columns:
        line, head = columns[column.name], f"{column.name} {_pg_type(column)}"
        assert line == head or line.startswith(head + " "), line
        assert (not column.nullable) is ("NOT NULL" in line or "PRIMARY KEY" in line), line
        assert column.primary_key is ("PRIMARY KEY" in line), line
        default = _default_sql(column)
        assert (default is not None) is (" DEFAULT " in line), line
        assert default is None or line.endswith(f" DEFAULT {default}"), line
        assert not column.foreign_keys and "REFERENCES" not in line
    checks = {c.name: str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint)}
    assert checks == {SYNC_CHECK: f"id = {MARKETPLACE_CLOUD_SYNC_ROW_ID}"}
    assert [line for line in lines if line.startswith("CONSTRAINT ")] == [f"CONSTRAINT {SYNC_CHECK} CHECK (id = 1)"]


def test_the_sync_row_shape_and_its_registration():
    table = MarketplaceCloudSync.__table__
    assert MARKETPLACE_CLOUD_SYNC_ROW_ID == 1 and table.c.id.autoincrement is False
    assert table.c.id.default.arg == MARKETPLACE_CLOUD_SYNC_ROW_ID
    timestamps = ("last_attempt_at", "last_success_at", "feed_generated_at", "updated_at")
    assert all(table.c[name].type.timezone for name in timestamps)
    assert all(_default_sql(table.c[name]) == "0" and not table.c[name].nullable for name in COUNTS)
    assert not table.indexes
    assert Base.metadata.tables["marketplace_cloud_sync"] is core.models.MarketplaceCloudSync.__table__


# --------------------------------------------------------------------------- Postgres (CI)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM workspaces LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM users LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Postgres checks need the test database: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def pg(engine):
    """A connection inside one transaction that is always rolled back."""
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            yield conn
        finally:
            trans.rollback()


def _empty_schema(conn) -> str:
    """A new, empty schema first on the search path (users and workspaces stay reachable in public)."""
    name = f"prd270_w1_{uuid.uuid4().hex[:12]}"
    conn.execute(sa.text(f"CREATE SCHEMA {name}"))
    conn.execute(sa.text(f"SET LOCAL search_path TO {name}, public"))
    return name


def _run(conn, path: Path, step: str = "upgrade") -> None:
    module = _load(path)
    with Operations.context(MigrationContext.configure(conn)):
        getattr(module, step)()


def _by_create_all(conn) -> None:
    """The two tables as the models declare them, in the current (empty) schema."""
    Base.metadata.create_all(conn, tables=[ThemePackage.__table__, MarketplaceCloudSync.__table__], checkfirst=False)


def _by_migrations(conn) -> None:
    for path in (*EARLIER, MIGRATION):
        _run(conn, path)


_SNAPSHOT = {
    "columns": (
        "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
        "FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = ANY(:tables)"
    ),
    "indexes": (
        "SELECT tablename, indexname, indexdef FROM pg_indexes "
        "WHERE schemaname = current_schema() AND tablename = ANY(:tables)"
    ),
    "constraints": (
        "SELECT cls.relname, con.conname, pg_get_constraintdef(con.oid) FROM pg_constraint con "
        "JOIN pg_class cls ON cls.oid = con.conrelid JOIN pg_namespace ns ON ns.oid = cls.relnamespace "
        "WHERE ns.nspname = current_schema() AND cls.relname = ANY(:tables)"
    ),
}


def _schema(conn) -> dict:
    """The two tables' columns, index definitions and constraints, the schema name removed."""
    schema = conn.execute(sa.text("SELECT current_schema()")).scalar()
    found = {}
    for kind, sql in _SNAPSHOT.items():
        rows = conn.execute(sa.text(sql), {"tables": list(TABLES)})
        found[kind] = sorted(tuple(str(value).replace(f"{schema}.", "") for value in row) for row in rows)
    return found


def _workspace(conn) -> str:
    ws = str(uuid.uuid4())
    conn.execute(sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270-w1')"), {"id": ws})
    return ws


_INSERT = (
    "INSERT INTO theme_packages (id, theme_id, version, owner_type, workspace_id, digest, manifest, "
    "file_manifest, storage_prefix{extra}) VALUES (CAST(:id AS uuid), :theme_id, '1.0.0', :owner, "
    "CAST(:ws AS uuid), :digest, CAST('{{}}' AS jsonb), CAST('{{}}' AS jsonb), 'themes/test/'{value})"
)
INSERT_OLD = sa.text(_INSERT.format(extra="", value=""))
INSERT_WITH_SOURCE = sa.text(_INSERT.format(extra=", source", value=", :source"))


def _package(conn, ws: str | None = None, source: str | None = None, old: bool = False) -> str:
    """A theme_packages row: a workspace's when ``ws`` is given, else a Marketplace one.

    ``old`` inserts it as a row from before the revision (the statement names no source column).
    """
    package_id = str(uuid.uuid4())
    params = {
        "id": package_id, "theme_id": f"prd270/{uuid.uuid4().hex[:8]}",
        "owner": "workspace" if ws else "marketplace", "ws": ws, "digest": uuid.uuid4().hex * 2,
    }
    if old:
        conn.execute(INSERT_OLD, params)
    else:
        conn.execute(INSERT_WITH_SOURCE, {**params, "source": source})
    return package_id


def _source(conn, package_id: str):
    return conn.execute(
        sa.text("SELECT source FROM theme_packages WHERE id = CAST(:id AS uuid)"), {"id": package_id},
    ).scalar()


def _refused(conn, check: str, statement, params: dict) -> None:
    with pytest.raises(IntegrityError, match=check):
        with conn.begin_nested():
            conn.execute(statement, params)


@pytest.mark.integration
def test_a_row_from_before_the_revision_reads_source_null(pg):
    _empty_schema(pg)
    for path in EARLIER:
        _run(pg, path)
    ws = _workspace(pg)
    old = [_package(pg, old=True), _package(pg, ws, old=True)]
    _run(pg, MIGRATION)
    assert [_source(pg, package_id) for package_id in old] == [None, None]
    assert pg.execute(sa.text("SELECT count(*) FROM theme_packages")).scalar() == 2


@pytest.mark.integration
@pytest.mark.parametrize("build", [_by_create_all, _by_migrations], ids=["create_all", "migrations"])
def test_cloud_only_on_a_marketplace_row_and_no_other_value(pg, build):
    _empty_schema(pg)
    build(pg)
    ws = _workspace(pg)
    cloud = _package(pg, source=THEME_SOURCE_CLOUD)
    local = _package(pg, ws)
    assert (_source(pg, cloud), _source(pg, local)) == ("cloud", None)
    params = {"id": str(uuid.uuid4()), "theme_id": "prd270/refused", "digest": "c" * 64}
    _refused(pg, SOURCE_CHECK, INSERT_WITH_SOURCE, {**params, "owner": "workspace", "ws": ws, "source": "cloud"})
    for other in ("local", "feed", "CLOUD", ""):
        _refused(pg, SOURCE_CHECK, INSERT_WITH_SOURCE, {**params, "owner": "marketplace", "ws": None, "source": other})
    to_cloud = sa.text("UPDATE theme_packages SET source = 'cloud' WHERE id = CAST(:id AS uuid)")
    _refused(pg, SOURCE_CHECK, to_cloud, {"id": local})


@pytest.mark.integration
@pytest.mark.parametrize("build", [_by_create_all, _by_migrations], ids=["create_all", "migrations"])
def test_the_sync_table_holds_one_row(pg, build):
    _empty_schema(pg)
    build(pg)
    insert = sa.text("INSERT INTO marketplace_cloud_sync (id) VALUES (:id)")
    pg.execute(insert, {"id": MARKETPLACE_CLOUD_SYNC_ROW_ID})
    row = pg.execute(sa.text(f"SELECT {', '.join(SYNC_COLUMNS)} FROM marketplace_cloud_sync")).one()
    assert tuple(getattr(row, name) for name in COUNTS) == (0, 0, 0, 0)
    assert row.updated_at is not None and row.last_attempt_at is None and row.last_error is None
    with pytest.raises(IntegrityError):
        with pg.begin_nested():
            pg.execute(insert, {"id": MARKETPLACE_CLOUD_SYNC_ROW_ID})
    _refused(pg, SYNC_CHECK, insert, {"id": MARKETPLACE_CLOUD_SYNC_ROW_ID + 1})


@pytest.mark.integration
def test_on_a_create_all_schema_the_upgrade_changes_nothing_and_the_migrations_build_it(pg):
    _empty_schema(pg)
    _by_create_all(pg)
    model = _schema(pg)
    assert {row[1] for row in model["columns"] if row[0] == "theme_packages"} >= {"source"}
    _run(pg, MIGRATION)
    assert _schema(pg) == model
    _run(pg, MIGRATION)  # a second run changes nothing either
    assert _schema(pg) == model
    _empty_schema(pg)
    _by_migrations(pg)
    assert _schema(pg) == model


@pytest.mark.integration
def test_the_downgrade_deletes_no_theme_packages_row(pg):
    _empty_schema(pg)
    _by_migrations(pg)
    ws = _workspace(pg)
    cloud, local = _package(pg, source=THEME_SOURCE_CLOUD), _package(pg, ws)
    _run(pg, MIGRATION, "downgrade")
    assert pg.execute(sa.text("SELECT count(*) FROM theme_packages")).scalar() == 2
    after = _schema(pg)
    assert not [row for row in after["columns"] if row[0] == "marketplace_cloud_sync" or row[1] == "source"]
    assert not [row for row in after["constraints"] if row[0] == "marketplace_cloud_sync" or row[1] == SOURCE_CHECK]
    assert not [row for row in after["indexes"] if row[0] == "marketplace_cloud_sync"]
    _run(pg, MIGRATION)
    assert (_source(pg, cloud), _source(pg, local)) == (None, None)
