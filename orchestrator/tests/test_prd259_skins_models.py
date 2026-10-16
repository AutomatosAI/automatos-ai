"""PRD-259 S-002 — marketplace_skins and skin_installations: the models and the migration, held together.

Two writers build these tables: the model layer (``create_all``: the fresh path and
the unit tests) and the wave's ONE migration, ``prd259_skins`` (existing databases).
A table that differs between them is how a column goes missing on one path, so this
file pins:

* the columns (Postgres type, nullability), the foreign keys and their ON DELETE, the
  three CHECKs and the indexes (three of them partial and unique), from the models'
  metadata;
* the migration, parsed (never run here: its SQL is Postgres's): revision
  ``prd259_skins`` on the head pinned before this wave, IF NOT EXISTS on every CREATE,
  every model column, CHECK and index in its SQL, no existing table altered, and a
  downgrade that drops both tables IF EXISTS;
* both tables build under SQLite through ``create_all`` and the partial unique
  indexes hold there: a slug is unique within one workspace's private skins, and a
  skin has one active installation per workspace.
"""
from __future__ import annotations

import ast
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.dialects import postgresql  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the FKs name)
from core.models.marketplace_skins import MarketplaceSkin, SkinInstallation  # noqa: E402

MIGRATION = _ORCH / "alembic" / "versions" / "prd259_skins.py"
# The single head on the base this wave was cut from (origin/main f2be2abb0).
HEAD_BEFORE_WAVE = "prd258_voice_timings"

SKIN_COLUMNS = {
    "id": ("UUID", False),
    "slug": ("VARCHAR(64)", False),
    "name": ("VARCHAR(40)", False),
    "version": ("VARCHAR(32)", False),
    "author_user_id": ("INTEGER", True),
    "skin": ("JSONB", False),
    "asset_prefix": ("VARCHAR", False),
    "preview_path": ("VARCHAR", True),
    "owner_type": ("VARCHAR(16)", False),
    "workspace_id": ("UUID", True),
    "status": ("VARCHAR(16)", False),
    "rejection_reason": ("TEXT", True),
    "featured": ("BOOLEAN", False),
    "install_count": ("INTEGER", False),
    "created_at": ("TIMESTAMP WITH TIME ZONE", False),
    "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
}
INSTALLATION_COLUMNS = {
    "id": ("UUID", False),
    "workspace_id": ("UUID", False),
    "skin_id": ("UUID", False),
    "version": ("VARCHAR(32)", False),
    "installed_by": ("INTEGER", True),
    "installed_at": ("TIMESTAMP WITH TIME ZONE", False),
    "uninstalled_at": ("TIMESTAMP WITH TIME ZONE", True),
}
CHECKS = {
    "ck_marketplace_skins_owner_type": "owner_type IN ('marketplace', 'workspace')",
    "ck_marketplace_skins_workspace_owner": "owner_type <> 'workspace' OR workspace_id IS NOT NULL",
    "ck_marketplace_skins_status": "status IN ('pending', 'approved', 'rejected')",
}
# name -> (table, columns, unique, partial predicate or None)
INDEXES = {
    "ix_marketplace_skins_workspace_id": ("marketplace_skins", ("workspace_id",), False, None),
    "ix_marketplace_skins_status": ("marketplace_skins", ("status",), False, None),
    "uq_marketplace_skins_workspace_slug": (
        "marketplace_skins", ("workspace_id", "slug"), True, "owner_type = 'workspace'",
    ),
    "uq_marketplace_skins_author_slug_version": (
        "marketplace_skins", ("author_user_id", "slug", "version"), True, "owner_type = 'marketplace'",
    ),
    "ix_skin_installations_workspace_id": ("skin_installations", ("workspace_id",), False, None),
    "uq_skin_installations_active": (
        "skin_installations", ("workspace_id", "skin_id"), True, "uninstalled_at IS NULL",
    ),
}
FOREIGN_KEYS = {
    ("marketplace_skins", "author_user_id"): ("users.id", "SET NULL"),
    ("skin_installations", "installed_by"): ("users.id", "SET NULL"),
    ("skin_installations", "skin_id"): ("marketplace_skins.id", "CASCADE"),
}
TABLES = {"marketplace_skins": MarketplaceSkin.__table__, "skin_installations": SkinInstallation.__table__}


def _pg_type(column: sa.Column) -> str:
    return column.type.compile(dialect=postgresql.dialect())


def _migration_constants() -> dict:
    """The migration's module-level string and tuple constants, read with ast (never imported)."""
    tree = ast.parse(MIGRATION.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return found


def _migration_sql() -> list:
    constants = _migration_constants()
    return [constants["_CREATE_SKINS"], constants["_CREATE_INSTALLATIONS"], *constants["_INDEXES"]]


def _function_source(name: str) -> str:
    source = MIGRATION.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def _column_line(create_sql: str, column: str) -> str:
    lines = [line.strip() for line in create_sql.splitlines() if line.strip().startswith(f"{column} ")]
    assert len(lines) == 1, f"{column} is declared {len(lines)} times in the migration"
    return lines[0]


# ---------------------------------------------------------------------------
# The models' metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_name,expected", [
    ("marketplace_skins", SKIN_COLUMNS),
    ("skin_installations", INSTALLATION_COLUMNS),
])
def test_the_columns_types_and_nullability(table_name, expected):
    table = TABLES[table_name]
    assert set(table.columns.keys()) == set(expected)
    for name, (pg_type, nullable) in expected.items():
        column = table.columns[name]
        assert _pg_type(column) == pg_type, name
        assert column.nullable is nullable, name
    assert [c.name for c in table.primary_key.columns] == ["id"]


def test_the_server_defaults():
    table = MarketplaceSkin.__table__
    assert str(table.columns["status"].server_default.arg) == "pending"
    assert str(table.columns["install_count"].server_default.arg) == "0"
    assert table.columns["featured"].server_default is not None
    assert table.columns["created_at"].server_default is not None
    assert SkinInstallation.__table__.columns["installed_at"].server_default is not None


def test_the_foreign_keys_and_their_on_delete():
    found = {}
    for table_name, table in TABLES.items():
        for fk in table.foreign_keys:
            found[(table_name, fk.parent.name)] = (fk.target_fullname, fk.ondelete)
    assert found == FOREIGN_KEYS


def test_the_three_checks():
    checks = {
        c.name: str(c.sqltext)
        for c in MarketplaceSkin.__table__.constraints
        if isinstance(c, sa.CheckConstraint)
    }
    assert checks == CHECKS
    assert not [c for c in SkinInstallation.__table__.constraints if isinstance(c, sa.CheckConstraint)]


def test_the_indexes_and_the_three_partial_unique_ones():
    found = {}
    for table_name, table in TABLES.items():
        for index in table.indexes:
            where = index.dialect_options["postgresql"]["where"]
            found[index.name] = (
                table_name, tuple(c.name for c in index.columns), bool(index.unique),
                None if where is None else str(where),
            )
    assert found == INDEXES
    partial_unique = [name for name, (_t, _c, unique, where) in found.items() if unique and where]
    assert len(partial_unique) == 3


# ---------------------------------------------------------------------------
# The migration, parsed
# ---------------------------------------------------------------------------


def test_one_revision_on_the_head_pinned_before_the_wave():
    constants = _migration_constants()
    assert constants["revision"] == "prd259_skins"
    assert constants["down_revision"] == HEAD_BEFORE_WAVE
    chained = [
        path.name for path in MIGRATION.parent.glob("*.py")
        if f'down_revision = "{HEAD_BEFORE_WAVE}"' in path.read_text(encoding="utf-8")
    ]
    assert chained == ["prd259_skins.py"]


def test_every_create_is_if_not_exists_and_nothing_is_altered():
    statements = _migration_sql()
    creates = [m.group(0) for sql in statements for m in re.finditer(r"CREATE\s+(UNIQUE\s+)?(TABLE|INDEX)\s+\S+", sql)]
    assert len(creates) == 2 + len(INDEXES)
    for create in creates:
        assert re.fullmatch(r"CREATE\s+(UNIQUE\s+)?(TABLE|INDEX)\s+IF", create), create
    for sql in statements:
        assert "IF NOT EXISTS" in sql
        assert "ALTER" not in sql.upper()
        assert "DROP" not in sql.upper()
    assert "op.create_table" not in MIGRATION.read_text(encoding="utf-8")
    assert "ALTER TABLE" not in _function_source("upgrade")


@pytest.mark.parametrize("table_name,constant,expected", [
    ("marketplace_skins", "_CREATE_SKINS", SKIN_COLUMNS),
    ("skin_installations", "_CREATE_INSTALLATIONS", INSTALLATION_COLUMNS),
])
def test_the_migration_creates_every_model_column(table_name, constant, expected):
    create_sql = _migration_constants()[constant]
    assert f"CREATE TABLE IF NOT EXISTS {table_name} (" in create_sql
    for name, (pg_type, nullable) in expected.items():
        line = _column_line(create_sql, name)
        assert line.startswith(f"{name} {pg_type}"), line
        assert line[len(f"{name} {pg_type}"):][:1] in ("", " ", ","), line
        not_null = "NOT NULL" in line or "PRIMARY KEY" in line
        assert not_null is (not nullable), line
    for (fk_table, column), (target, ondelete) in FOREIGN_KEYS.items():
        if fk_table == table_name:
            target_table, target_column = target.split(".")
            assert f"REFERENCES {target_table}({target_column}) ON DELETE {ondelete}" in _column_line(create_sql, column)


def test_the_migration_carries_the_checks_and_the_indexes():
    constants = _migration_constants()
    for name, sqltext in CHECKS.items():
        assert f"CONSTRAINT {name} CHECK ({sqltext})" in constants["_CREATE_SKINS"]
    index_sql = list(constants["_INDEXES"])
    assert len(index_sql) == len(INDEXES)
    for name, (table, columns, unique, where) in INDEXES.items():
        kind = "UNIQUE INDEX" if unique else "INDEX"
        expected = f"CREATE {kind} IF NOT EXISTS {name} ON {table} ({', '.join(columns)})"
        expected += f" WHERE {where};" if where else ";"
        assert expected in index_sql, expected


def test_the_downgrade_drops_both_tables_if_exists():
    downgrade = _function_source("downgrade")
    installations = downgrade.index("DROP TABLE IF EXISTS skin_installations;")
    skins = downgrade.index("DROP TABLE IF EXISTS marketplace_skins;")
    assert installations < skins
    assert downgrade.count("DROP TABLE") == 2


# ---------------------------------------------------------------------------
# SQLite: create_all builds both tables; the partial unique indexes hold
# ---------------------------------------------------------------------------


def _session():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    MarketplaceSkin.metadata.create_all(engine, tables=list(TABLES.values()))
    return engine, sessionmaker(bind=engine)()


def _private_skin(workspace_id: uuid.UUID, slug: str = "ink", version: str = "0.1.0") -> MarketplaceSkin:
    return MarketplaceSkin(
        slug=slug, name="Ink", version=version, skin={"slug": slug}, owner_type="workspace",
        workspace_id=workspace_id, status="approved",
        asset_prefix=f"skins/_workspace/{workspace_id}/{slug}/{version}/",
    )


def test_a_private_slug_is_unique_within_one_workspace_only():
    engine, session = _session()
    try:
        workspace_a, workspace_b = uuid.uuid4(), uuid.uuid4()
        session.add_all([_private_skin(workspace_a), _private_skin(workspace_b)])
        session.flush()
        first = session.query(MarketplaceSkin).filter_by(workspace_id=workspace_a).one()
        assert first.status == "approved" and first.featured is False and first.install_count == 0
        session.add(_private_skin(workspace_a, version="0.2.0"))
        with pytest.raises(IntegrityError):
            session.flush()
    finally:
        session.close()
        engine.dispose()


def test_a_workspace_has_one_active_installation_per_skin():
    engine, session = _session()
    try:
        workspace = uuid.uuid4()
        skin = _private_skin(workspace)
        session.add(skin)
        session.flush()
        old = SkinInstallation(workspace_id=workspace, skin_id=skin.id, version="0.1.0")
        session.add(old)
        session.flush()
        old.uninstalled_at = datetime.now(timezone.utc)
        session.flush()
        session.add(SkinInstallation(workspace_id=workspace, skin_id=skin.id, version="0.1.0"))
        session.flush()
        session.add(SkinInstallation(workspace_id=workspace, skin_id=skin.id, version="0.1.0"))
        with pytest.raises(IntegrityError):
            session.flush()
    finally:
        session.close()
        engine.dispose()
