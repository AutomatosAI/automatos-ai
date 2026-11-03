"""PRD-270 W2 US-270-11a (K6) — prd270_w2_cloud_bundles: the marketplace_cloud_items table.

The wave's one revision creates one table, ``marketplace_cloud_items`` (which local marketplace
rows a Sync with Automatos Cloud wrote), with its two CHECKs, its unique (type, external id) and
one partial unique index per local key. It alters no table and changes no row. Proven here:

* the revision chains onto the base's head (``prd270_w1_cloud_sync``) and sits under the one head;
* its upgrade is K6's CREATE TABLE and the four indexes, nothing else, and its downgrade drops
  the table and nothing else;
* the model (``core.models.marketplace_cloud.MarketplaceCloudItem``) and the DDL declare the same
  columns, types, nullability, defaults, foreign keys, CHECKs, unique constraint and indexes;
* on Postgres (inside a rolled-back transaction, each case in a new empty schema; the four tables
  the foreign keys name are public's, where a test row is written and rolled back): a mapping
  with two local keys, a key of another type or an unknown type is refused, both where
  ``create_all`` built the table and where the migration did; a second mapping for one (type,
  external id) or one local row is refused; deleting the local row deletes its mapping; on a
  ``create_all``-built schema the upgrade changes nothing and the migration builds the model's
  schema; the downgrade drops the table and deletes no ``agents``, ``workflow_recipes``,
  ``marketplace_packages`` or ``marketplace_plugins`` row.
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
from core.models.marketplace_cloud import (
    CLOUD_ITEM_KEYS, CLOUD_ITEM_TYPES, ITEM_TYPE_CHECK, ONE_ROW_CHECK, MarketplaceCloudItem,
)

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
MIGRATION = VERSIONS / "prd270_w2_cloud_bundles.py"
REVISION = "prd270_w2_cloud_bundles"
BASE_HEAD = "prd270_w1_cloud_sync"
TABLE = "marketplace_cloud_items"
TYPE_CHECK, ONE_ROW = "ck_marketplace_cloud_items_type", "ck_marketplace_cloud_items_one_row"
EXTERNAL = "uq_marketplace_cloud_items_external"
COLUMNS = (
    "id", "item_type", "external_id", "version", "revision", "agent_id", "recipe_id", "package_id",
    "plugin_id", "unresolved", "synced_at",
)
# K6's DDL, as the kit states it (compared with whitespace collapsed).
K6_DDL = (
    "CREATE TABLE IF NOT EXISTS marketplace_cloud_items (id SERIAL PRIMARY KEY, item_type VARCHAR(16) NOT NULL, "
    "external_id VARCHAR(32) NOT NULL, version VARCHAR(32) NOT NULL, revision VARCHAR(64) NOT NULL, "
    "agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE, "
    "recipe_id INTEGER REFERENCES workflow_recipes(id) ON DELETE CASCADE, "
    "package_id UUID REFERENCES marketplace_packages(id) ON DELETE CASCADE, "
    "plugin_id UUID REFERENCES marketplace_plugins(id) ON DELETE CASCADE, "
    "unresolved JSONB NOT NULL DEFAULT '[]'::jsonb, synced_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(), "
    "CONSTRAINT ck_marketplace_cloud_items_type CHECK (item_type IN ('agent', 'playbook', 'package', 'plugin')), "
    "CONSTRAINT ck_marketplace_cloud_items_one_row CHECK (num_nonnulls(agent_id, recipe_id, package_id, plugin_id) = 1 "
    "AND (agent_id IS NULL OR item_type = 'agent') AND (recipe_id IS NULL OR item_type = 'playbook') "
    "AND (package_id IS NULL OR item_type = 'package') AND (plugin_id IS NULL OR item_type = 'plugin')), "
    "CONSTRAINT uq_marketplace_cloud_items_external UNIQUE (item_type, external_id));"
)
INDEX = re.compile(rf"CREATE UNIQUE INDEX IF NOT EXISTS (\w+) ON {TABLE} \((\w+)\) WHERE (\w+) IS NOT NULL;")
ROW_WRITES = re.compile(r"\b(ALTER TABLE|DROP|DELETE FROM|TRUNCATE|UPDATE \w+ SET|INSERT INTO)\b")


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


def _collapsed(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).replace("( ", "(").replace(" )", ")").strip()


def _create_sql() -> str:
    return dict(_constants()["_TABLES"])[TABLE]


def _ddl_lines() -> list[str]:
    lines = [line.strip().rstrip(",") for line in _create_sql().strip().splitlines()]
    assert lines[0] == f"CREATE TABLE IF NOT EXISTS {TABLE} (" and lines[-1] == ");"
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


def _pg_type(column: sa.Column) -> str:
    return column.type.compile(dialect=postgresql.dialect())


def _default_sql(column: sa.Column) -> str | None:
    default = column.server_default
    if default is None:
        return None
    if isinstance(default.arg, str):
        return f"'{default.arg}'"
    return str(default.arg.compile(dialect=postgresql.dialect()))


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


def test_the_upgrade_is_exactly_k6_one_table_and_four_indexes():
    constants = _constants()
    assert [name for name, _sql in constants["_TABLES"]] == [TABLE]
    assert _collapsed(_create_sql()) == _collapsed(K6_DDL)
    assert _statements("upgrade") == [_create_sql(), *constants["_INDEXES"]]
    indexes = [INDEX.fullmatch(statement).groups() for statement in constants["_INDEXES"]]
    assert indexes == [
        (f"uq_marketplace_cloud_items_{key.removesuffix('_id')}", key, key) for key in CLOUD_ITEM_KEYS.values()
    ]
    for sql in _statements("upgrade"):
        assert not ROW_WRITES.search(sql), sql
    assert "op.create_table" not in MIGRATION.read_text(encoding="utf-8")


def test_the_downgrade_drops_the_table_and_nothing_else():
    assert _statements("downgrade") == [f"DROP TABLE IF EXISTS {TABLE};"]
    assert not re.search(r"\b(ALTER TABLE|DELETE FROM|TRUNCATE|UPDATE \w+ SET)\b", MIGRATION.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- model == DDL


def _expected_column_line(column: sa.Column) -> str:
    """The DDL line the model's column stands for, built from the model alone."""
    if column.primary_key:
        assert isinstance(column.type, sa.Integer) and column.autoincrement in (True, "auto")
        return f"{column.name} SERIAL PRIMARY KEY"
    parts = [column.name, _pg_type(column)]
    for fk in column.foreign_keys:
        table, key = fk.target_fullname.split(".")
        parts.append(f"REFERENCES {table}({key}) ON DELETE {fk.ondelete}")
    if not column.nullable:
        parts.append("NOT NULL")
    if _default_sql(column) is not None:
        parts.append(f"DEFAULT {_default_sql(column)}")
    return " ".join(parts)


def test_the_model_columns_equal_the_ddl():
    table = MarketplaceCloudItem.__table__
    columns = [line for line in _ddl_lines() if not line.startswith("CONSTRAINT ")]
    assert tuple(table.columns.keys()) == tuple(line.split(" ", 1)[0] for line in columns) == COLUMNS
    assert [_expected_column_line(column) for column in table.columns] == columns
    assert table.c.synced_at.type.timezone is True
    assert [len(table.c[key].foreign_keys) for key in CLOUD_ITEM_KEYS.values()] == [1, 1, 1, 1]
    assert all(table.c[key].nullable for key in CLOUD_ITEM_KEYS.values())


def test_the_model_checks_unique_and_indexes_equal_the_ddl():
    table = MarketplaceCloudItem.__table__
    checks = {c.name: str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint)}
    assert checks == {TYPE_CHECK: ITEM_TYPE_CHECK, ONE_ROW: ONE_ROW_CHECK}
    constraints = [line for line in _ddl_lines() if line.startswith("CONSTRAINT ")]
    uniques = [c for c in table.constraints if isinstance(c, sa.UniqueConstraint)]
    assert [(c.name, tuple(c.columns.keys())) for c in uniques] == [(EXTERNAL, ("item_type", "external_id"))]
    assert constraints == [
        f"CONSTRAINT {TYPE_CHECK} CHECK ({ITEM_TYPE_CHECK})",
        f"CONSTRAINT {ONE_ROW} CHECK ({ONE_ROW_CHECK})",
        f"CONSTRAINT {EXTERNAL} UNIQUE (item_type, external_id)",
    ]
    model = sorted(
        (index.name, *index.columns.keys(), str(index.dialect_options["postgresql"]["where"]))
        for index in table.indexes
        if index.unique
    )
    ddl = sorted((name, key, f"{where} IS NOT NULL") for name, key, where in (
        INDEX.fullmatch(statement).groups() for statement in _constants()["_INDEXES"]
    ))
    assert model == ddl and len(model) == len(table.indexes) == 4


def test_the_types_and_keys_are_the_checks_and_the_model_is_registered():
    assert CLOUD_ITEM_TYPES == ("agent", "playbook", "package", "plugin")
    assert tuple(CLOUD_ITEM_KEYS) == CLOUD_ITEM_TYPES
    assert ITEM_TYPE_CHECK == "item_type IN (" + ", ".join(f"'{kind}'" for kind in CLOUD_ITEM_TYPES) + ")"
    keys = ", ".join(CLOUD_ITEM_KEYS.values())
    per_type = "".join(f" AND ({key} IS NULL OR item_type = '{kind}')" for kind, key in CLOUD_ITEM_KEYS.items())
    assert ONE_ROW_CHECK == f"num_nonnulls({keys}) = 1{per_type}"
    with pytest.raises(TypeError):
        CLOUD_ITEM_KEYS["theme"] = "theme_id"  # read-only
    assert Base.metadata.tables[TABLE] is core.models.MarketplaceCloudItem.__table__
    assert core.models.CLOUD_ITEM_TYPES is CLOUD_ITEM_TYPES


# --------------------------------------------------------------------------- Postgres (CI)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for table in ("agents", "workflow_recipes", "marketplace_packages", "marketplace_plugins"):
                conn.execute(sa.text(f"SELECT 1 FROM {table} LIMIT 1"))
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
    """A new, empty schema first on the search path (the foreign keys' tables stay reachable in public)."""
    name = f"prd270_w2_{uuid.uuid4().hex[:12]}"
    conn.execute(sa.text(f"CREATE SCHEMA {name}"))
    conn.execute(sa.text(f"SET LOCAL search_path TO {name}, public"))
    return name


def _run(conn, step: str = "upgrade") -> None:
    module = _load(MIGRATION)
    with Operations.context(MigrationContext.configure(conn)):
        getattr(module, step)()


def _by_create_all(conn) -> None:
    """The table as the model declares it, in the current (empty) schema."""
    Base.metadata.create_all(conn, tables=[MarketplaceCloudItem.__table__], checkfirst=False)


def _by_migration(conn) -> None:
    _run(conn)


_SNAPSHOT = {
    "columns": (
        "SELECT column_name, data_type, character_maximum_length, is_nullable, column_default "
        "FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = :table"
    ),
    "indexes": "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = current_schema() AND tablename = :table",
    "constraints": (
        "SELECT con.conname, pg_get_constraintdef(con.oid) FROM pg_constraint con "
        "JOIN pg_class cls ON cls.oid = con.conrelid JOIN pg_namespace ns ON ns.oid = cls.relnamespace "
        "WHERE ns.nspname = current_schema() AND cls.relname = :table"
    ),
}


def _schema(conn) -> dict:
    """The table's columns, index definitions and constraints in the current schema, its name removed."""
    schema = conn.execute(sa.text("SELECT current_schema()")).scalar()
    found = {}
    for kind, sql in _SNAPSHOT.items():
        rows = conn.execute(sa.text(sql), {"table": TABLE})
        found[kind] = sorted(tuple(str(value).replace(f"{schema}.", "") for value in row) for row in rows)
    return found


def _tag() -> str:
    return f"prd270w2-{uuid.uuid4().hex[:10]}"


def _agent(conn) -> int:
    return conn.execute(sa.text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
        "VALUES (:name, 'custom', NULL, 'active', CAST('{}' AS json), 'marketplace') RETURNING id"
    ), {"name": _tag()}).scalar()


def _recipe(conn) -> int:
    return conn.execute(sa.text(
        "INSERT INTO workflow_recipes (template_id, name, description, owner_type, template_definition, "
        "created_at, updated_at, created_by) VALUES (:tag, :tag, 'A Cloud Playbook', 'marketplace', "
        "CAST('{}' AS jsonb), now(), now(), 'prd270w2') RETURNING id"
    ), {"tag": _tag()}).scalar()


def _package(conn) -> str:
    package_id = str(uuid.uuid4())
    conn.execute(sa.text(
        "INSERT INTO marketplace_packages (id, slug, name, vertical_tags, matching, members, setup_manifest, "
        "showcase) VALUES (CAST(:id AS uuid), :tag, :tag, CAST('[]' AS jsonb), CAST('{}' AS jsonb), "
        "CAST('[]' AS jsonb), CAST('{}' AS jsonb), false)"
    ), {"id": package_id, "tag": _tag()})
    return package_id


def _plugin(conn) -> str:
    plugin_id = str(uuid.uuid4())
    conn.execute(sa.text(
        "INSERT INTO marketplace_plugins (id, slug, name, version) VALUES (CAST(:id AS uuid), :tag, :tag, '1.0.0')"
    ), {"id": plugin_id, "tag": _tag()})
    return plugin_id


LOCAL_ROW = {"agent": _agent, "playbook": _recipe, "package": _package, "plugin": _plugin}
LOCAL_TABLE = {"agent": "agents", "playbook": "workflow_recipes", "package": "marketplace_packages",
               "plugin": "marketplace_plugins"}


def _local_rows(conn) -> dict:
    """One fresh local row of each type, in public: {type: id}."""
    return {kind: make(conn) for kind, make in LOCAL_ROW.items()}


UUID_KEYS = ("package_id", "plugin_id")


def _bind(key: str) -> str:
    return f"CAST(:{key} AS uuid)" if key in UUID_KEYS else f":{key}"


def _where_id(kind: str) -> str:
    """``id = :id`` on the local table of ``kind``, cast to uuid where its key is a UUID."""
    return "id = CAST(:id AS uuid)" if CLOUD_ITEM_KEYS[kind] in UUID_KEYS else "id = :id"


def _insert_mapping(item_type: str, keys: dict, external_id: str | None = None) -> tuple[sa.TextClause, dict]:
    """A mapping INSERT naming the given local keys ({column: id}, possibly none), with fresh Cloud facts."""
    names = ", ".join(("item_type", "external_id", "version", "revision", *keys))
    values = ", ".join((":item_type", ":external_id", "'1.0.0'", ":revision", *(_bind(key) for key in keys)))
    statement = sa.text(f"INSERT INTO {TABLE} ({names}) VALUES ({values}) RETURNING id")
    params = {
        "item_type": item_type, "external_id": external_id or uuid.uuid4().hex,
        "revision": uuid.uuid4().hex * 2, **keys,
    }
    return statement, params


def _map(conn, item_type: str, local_id, external_id: str | None = None) -> int:
    statement, params = _insert_mapping(item_type, {CLOUD_ITEM_KEYS[item_type]: local_id}, external_id)
    return conn.execute(statement, params).scalar()


def _refused(conn, constraint: str, item_type: str, keys: dict, external_id: str | None = None) -> None:
    statement, params = _insert_mapping(item_type, keys, external_id)
    with pytest.raises(IntegrityError, match=constraint):
        with conn.begin_nested():
            conn.execute(statement, params)


def _mappings(conn) -> int:
    return conn.execute(sa.text(f"SELECT count(*) FROM {TABLE}")).scalar()


@pytest.mark.integration
@pytest.mark.parametrize("build", [_by_create_all, _by_migration], ids=["create_all", "migration"])
def test_a_mapping_holds_exactly_one_local_key_of_its_type(pg, build):
    _empty_schema(pg)
    build(pg)
    local = _local_rows(pg)
    ids = [_map(pg, kind, local[kind]) for kind in CLOUD_ITEM_TYPES]  # one of each type coexists
    row = pg.execute(sa.text(f"SELECT unresolved, synced_at FROM {TABLE} WHERE id = :id"), {"id": ids[0]}).one()
    assert row.unresolved == [] and row.synced_at is not None
    two_keys = {"agent_id": _agent(pg), "recipe_id": _recipe(pg)}
    _refused(pg, ONE_ROW, "agent", two_keys)
    _refused(pg, ONE_ROW, "playbook", two_keys)
    _refused(pg, ONE_ROW, "playbook", {"agent_id": _agent(pg)})  # a key of another type
    _refused(pg, ONE_ROW, "agent", {"plugin_id": _plugin(pg)})
    _refused(pg, ONE_ROW, "agent", {})  # no local key at all
    for unknown in ("theme", "AGENT", ""):
        _refused(pg, rf"({ONE_ROW}|{TYPE_CHECK})", unknown, {"agent_id": _agent(pg)})
    assert _mappings(pg) == 4


@pytest.mark.integration
@pytest.mark.parametrize("build", [_by_create_all, _by_migration], ids=["create_all", "migration"])
def test_one_mapping_per_cloud_item_and_per_local_row(pg, build):
    _empty_schema(pg)
    build(pg)
    local = _local_rows(pg)
    external = uuid.uuid4().hex
    _map(pg, "agent", local["agent"], external)
    _refused(pg, EXTERNAL, "agent", {"agent_id": _agent(pg)}, external)  # one (type, external id)
    _map(pg, "playbook", local["playbook"], external)  # the same external id of another type is another item
    _map(pg, "package", local["package"])
    _map(pg, "plugin", local["plugin"])
    for kind in CLOUD_ITEM_TYPES:  # a second mapping of one local row, under a new external id
        key = CLOUD_ITEM_KEYS[kind]
        _refused(pg, f"uq_marketplace_cloud_items_{key.removesuffix('_id')}", kind, {key: local[kind]})
    assert _mappings(pg) == 4


@pytest.mark.integration
@pytest.mark.parametrize("kind", CLOUD_ITEM_TYPES)
def test_deleting_the_local_row_deletes_its_mapping(pg, kind):
    _empty_schema(pg)
    _by_migration(pg)
    local = _local_rows(pg)
    for each in CLOUD_ITEM_TYPES:
        _map(pg, each, local[each])
    pg.execute(sa.text(f"DELETE FROM {LOCAL_TABLE[kind]} WHERE {_where_id(kind)}"), {"id": local[kind]})
    left = pg.execute(sa.text(f"SELECT item_type FROM {TABLE} ORDER BY item_type")).scalars().all()
    assert left == sorted(each for each in CLOUD_ITEM_TYPES if each != kind)


@pytest.mark.integration
def test_on_a_create_all_schema_the_upgrade_changes_nothing_and_the_migration_builds_it(pg):
    _empty_schema(pg)
    _by_create_all(pg)
    model = _schema(pg)
    assert sorted(row[0] for row in model["columns"]) == sorted(COLUMNS)
    assert len(model["indexes"]) == 6  # the primary key, the external unique and the four per local key
    _run(pg)
    assert _schema(pg) == model
    _run(pg)  # a second run changes nothing either
    assert _schema(pg) == model
    _empty_schema(pg)
    _by_migration(pg)
    assert _schema(pg) == model


@pytest.mark.integration
def test_the_downgrade_drops_the_table_and_deletes_no_local_row(pg):
    _empty_schema(pg)
    _by_migration(pg)
    local = _local_rows(pg)
    for kind in CLOUD_ITEM_TYPES:
        _map(pg, kind, local[kind])
    _run(pg, "downgrade")
    assert _schema(pg) == {"columns": [], "indexes": [], "constraints": []}
    for kind, table in LOCAL_TABLE.items():
        found = pg.execute(sa.text(f"SELECT count(*) FROM {table} WHERE {_where_id(kind)}"), {"id": local[kind]})
        assert found.scalar() == 1, table
    _run(pg)  # and up again: an empty table
    assert _mappings(pg) == 0
