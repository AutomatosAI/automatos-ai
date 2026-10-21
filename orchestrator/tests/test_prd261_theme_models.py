"""PRD-261 A-05 and C-05f — the ten theme tables: the models and the two migrations, held together.

Two writers build these tables: the model layer (``create_all``: the fresh path and
the unit tests) and the migrations (existing databases): ``prd261_themes`` (A-05, the
nine tables) and ``prd261_revocation_feed`` (C-05f: ``theme_revocations.origin`` and
``theme_revocation_sync``). A table that differs between them is how a column goes
missing on one path, so this file pins:

* each migration chains onto the head pinned before its wave and sits under the one
  Alembic head (which revision IS the head stays with the two pins AGENTS.md names);
* model columns equal DDL columns, table by table: type, nullability, primary key,
  server default, foreign key and ON DELETE; the CHECKs and the indexes too, read
  from the migrations' constants with ast (no migration is imported here);
* the plan's shape (§2.6 A-05): ``digest`` has a plain index, never a global unique;
  the four partial uniques; ``previous_package_id``, ``pending_package_id`` and
  ``pinned_version`` on the installation; and (§2.6 C-05f) ``origin VARCHAR(16) NOT
  NULL DEFAULT 'local'`` with its CHECK, the single-row ``theme_revocation_sync``, and
  the live-local unique (one live ``local`` revocation per target and level, each NULL
  target column read as '', so a digest row's NULL theme id doesn't make it distinct);
* under SQLite (``create_all``): the partial uniques refuse duplicates, two workspaces
  may hold the same digest, and a workspace row and a Marketplace row may too; a
  revocation reads ``local`` unless the feed wrote it, the sync table holds one row, and
  a second live ``local`` row with a live row's target and level is refused (a ``feed``
  twin, a lifted twin and another level are not);
* on real Postgres (``@integration``, inside a rolled-back transaction; skipped when no
  Postgres is reachable, run by test.yml): on a ``create_all`` database the migrations
  change nothing; in an empty schema they build exactly the model's schema (columns,
  defaults, constraints, index definitions) and a second run changes nothing; a row
  from before C-05f reads ``local`` and the feed's downgrade restores the schema before
  it; and the partial uniques (the live-local one included) hold there. ``prd261_themes``'
  downgrade is pinned by reading it, not run, so the test never takes an exclusive lock on
  ``workspaces`` or ``users``; the feed's downgrade runs in the empty schema and locks only
  its tables.
"""
from __future__ import annotations

import ast
import importlib.util
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
from alembic.operations import Operations  # noqa: E402
from alembic.runtime.migration import MigrationContext  # noqa: E402
from sqlalchemy.dialects import postgresql  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the FKs name)
from core.models.themes import (  # noqa: E402
    THEME_REVOCATION_ORIGINS,
    THEME_REVOCATION_SYNC_ROW_ID,
    ThemeChoice,
    ThemeInstallation,
    ThemeLook,
    ThemeOptIn,
    ThemePackage,
    ThemePublisher,
    ThemeReviewEvent,
    ThemeRevocation,
    ThemeRevocationSync,
    ThemeWorkspaceSettings,
)

VERSIONS = _ORCH / "alembic" / "versions"
MIGRATION = VERSIONS / "prd261_themes.py"
FEED_MIGRATION = VERSIONS / "prd261_revocation_feed.py"
MIGRATIONS = (MIGRATION, FEED_MIGRATION)  # in chain order
# The single head on the base this wave was cut from (origin/main 0d9f0ac59).
HEAD_BEFORE_WAVE = "prd258_voice_timings"
# The single head on the base W-C3 was cut from (origin/main 7ae677b52).
HEAD_BEFORE_C3 = "clerk_user_relinks_20261019"
MODELS = (
    ThemePackage, ThemePublisher, ThemeInstallation, ThemeChoice, ThemeLook,
    ThemeOptIn, ThemeWorkspaceSettings, ThemeRevocation, ThemeReviewEvent, ThemeRevocationSync,
)
TABLES = {model.__tablename__: model.__table__ for model in MODELS}
# Plan §2.6 A-05 / PA-4: the nine tables of Phase A and C, all in prd261_themes.
PHASE_A_TABLES = {
    "theme_packages", "theme_publishers", "theme_installations", "theme_choices", "theme_looks",
    "theme_opt_ins", "theme_workspace_settings", "theme_revocations", "theme_review_events",
}
# Plan §2.6 C-05f: the table prd261_revocation_feed adds.
FEED_TABLES = {"theme_revocation_sync"}
PLANNED_TABLES = PHASE_A_TABLES | FEED_TABLES
SYNC_COLUMNS = (
    "id", "last_attempt_at", "last_success_at", "feed_generated_at", "last_error", "revoked_key_ids", "updated_at",
)
PARTIAL_UNIQUES = {
    "uq_theme_packages_workspace_version": (("workspace_id", "theme_id", "version"), "owner_type = 'workspace'"),
    "uq_theme_packages_workspace_digest": (("workspace_id", "digest"), "owner_type = 'workspace'"),
    "uq_theme_packages_marketplace_version": (("theme_id", "version"), "owner_type = 'marketplace'"),
    "uq_theme_packages_marketplace_digest": (("digest",), "owner_type = 'marketplace'"),
}
DIGEST = "a" * 64
KEY_ID = "ed25519:0123456789abcdef"
# The feed migration's guarded ALTERs (inside its DO block).
ADDED_COLUMN = re.compile(r"ALTER TABLE (\w+) ADD COLUMN (.+);")
ADDED_CHECK = re.compile(r"ALTER TABLE (\w+) ADD CONSTRAINT (\w+) CHECK \((.+)\);")
CREATED_INDEX = re.compile(r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS (\w+) ON (\w+) ")
LIVE_LOCAL_UNIQUE = "uq_theme_revocations_live_local"
LIVE_LOCAL_TERMS = ("COALESCE(digest, '')", "COALESCE(theme_id, '')", "COALESCE(version_range, '')", "level")
LIVE_LOCAL_PREDICATE = "lifted_at IS NULL AND origin = 'local'"


def _pg_type(column: sa.Column) -> str:
    return column.type.compile(dialect=postgresql.dialect())


def _where(index: sa.Index, dialect: str = "postgresql"):
    where = index.dialect_options[dialect]["where"]
    return None if where is None else str(where)


def _terms(index: sa.Index) -> tuple:
    """What an index is over, as DDL names it: a column's name, or an expression's SQL text."""
    return tuple(term.name if isinstance(term, sa.Column) else str(term) for term in index.expressions)


def _migration_constants(path: Path = MIGRATION) -> dict:
    """A migration's module-level literal constants, read with ast (never imported)."""
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return found


def _create_sql() -> dict:
    """CREATE TABLE of every theme table, from the migration that creates it."""
    found = {}
    for path in MIGRATIONS:
        found.update(dict(_migration_constants(path)["_TABLES"]))
    return found


def _feed_alters() -> str:
    return _migration_constants(FEED_MIGRATION)["_ADD_ORIGIN"]


def _added_columns(table_name: str) -> dict:
    """Column DDL the feed migration adds to an existing table, by column name."""
    found = ADDED_COLUMN.findall(_feed_alters())
    return {line.split(" ", 1)[0]: line for table, line in found if table == table_name}


def _added_checks(table_name: str) -> dict:
    """CHECKs the feed migration adds to an existing table: name -> body."""
    return {name: body for table, name, body in ADDED_CHECK.findall(_feed_alters()) if table == table_name}


def _column_lines(create_sql: str) -> dict:
    lines = [line.strip() for line in create_sql.strip().splitlines()[1:-1]]
    columns = [line for line in lines if line and not line.startswith("CONSTRAINT ")]
    by_name = {line.split(" ", 1)[0]: line for line in columns}
    assert len(by_name) == len(columns), "a column is declared twice in the migration"
    return by_name


def _function_source(name: str, path: Path = MIGRATION) -> str:
    source = path.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


# ---------------------------------------------------------------------------
# One head; the ten tables
# ---------------------------------------------------------------------------


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


def _down_revisions(down) -> list:
    return list(down) if isinstance(down, (list, tuple)) else [down] if down else []


def _ancestors_and_self(revisions: dict, rev: str) -> set:
    seen, todo = set(), [rev]
    while todo:
        current = todo.pop()
        if current not in seen:
            seen.add(current)
            todo.extend(_down_revisions(revisions.get(current)))
    return seen


def _single_head(revisions: dict) -> str:
    parents = {down for downs in revisions.values() for down in _down_revisions(downs)}
    heads = [rev for rev in revisions if rev not in parents]
    assert len(heads) == 1, heads
    return heads[0]


def test_one_head_and_prd261_themes_chains_onto_the_head_before_the_wave():
    # Which revision is the head is pinned only by the two pins AGENTS.md names
    # (test_prd209 EXPECTED_HEAD, test_prd236), so a later migration moves those two
    # and this test keeps passing while prd261_themes stays under the one head.
    revisions = _revisions()
    assert "prd261_themes" in _ancestors_and_self(revisions, _single_head(revisions))
    assert revisions["prd261_themes"] == HEAD_BEFORE_WAVE
    assert [rev for rev, down in revisions.items() if down == HEAD_BEFORE_WAVE] == ["prd261_themes"]


def test_one_head_and_the_feed_migration_chains_onto_the_head_before_w_c3():
    revisions = _revisions()
    assert FEED_MIGRATION.stem in _ancestors_and_self(revisions, _single_head(revisions))
    assert revisions[FEED_MIGRATION.stem] == HEAD_BEFORE_C3
    assert [rev for rev, down in revisions.items() if down == HEAD_BEFORE_C3] == [FEED_MIGRATION.stem]
    assert "prd261_themes" in _ancestors_and_self(revisions, FEED_MIGRATION.stem)


def test_the_ten_tables_are_the_models_and_the_migrations():
    assert set(TABLES) == PLANNED_TABLES
    order = [name for name, _sql in _migration_constants()["_TABLES"]]
    assert set(order) == PHASE_A_TABLES and len(order) == len(PHASE_A_TABLES)
    assert order[0] == "theme_packages", "theme_packages is created before the tables that refer to it"
    feed = [name for name, _sql in _migration_constants(FEED_MIGRATION)["_TABLES"]]
    assert feed == sorted(FEED_TABLES)


# ---------------------------------------------------------------------------
# Model columns equal DDL columns
# ---------------------------------------------------------------------------


def _assert_column_matches(column: sa.Column, line: str) -> None:
    head = f"{column.name} {_pg_type(column)}"
    assert line.startswith(head) and line[len(head):][:1] in ("", " ", ","), line
    assert (not column.nullable) is ("NOT NULL" in line or "PRIMARY KEY" in line), line
    assert column.primary_key is ("PRIMARY KEY" in line), line
    default = column.server_default
    assert (default is not None) is (" DEFAULT " in line), line
    if default is not None and isinstance(default.arg, str):
        assert f"DEFAULT '{default.arg}'" in line, line
    fks = list(column.foreign_keys)
    assert len(fks) == line.count("REFERENCES"), line
    for fk in fks:
        table, target = fk.target_fullname.split(".")
        assert f"REFERENCES {table}({target}) ON DELETE {fk.ondelete}" in line, line


@pytest.mark.parametrize("table_name", sorted(PLANNED_TABLES))
def test_model_columns_equal_ddl_columns(table_name):
    table = TABLES[table_name]
    create_sql = _create_sql()[table_name]
    assert create_sql.strip().startswith(f"CREATE TABLE IF NOT EXISTS {table_name} (")
    created, added = _column_lines(create_sql), _added_columns(table_name)
    assert not set(created) & set(added), "a column is both created and added"
    lines = {**created, **added}
    assert set(lines) == set(table.columns.keys())
    for column in table.columns:
        _assert_column_matches(column, lines[column.name])


@pytest.mark.parametrize("table_name", sorted(PLANNED_TABLES))
def test_model_checks_equal_ddl_checks(table_name):
    checks = {c.name: str(c.sqltext) for c in TABLES[table_name].constraints if isinstance(c, sa.CheckConstraint)}
    create_sql, added = _create_sql()[table_name], _added_checks(table_name)
    assert create_sql.count("CONSTRAINT ") + len(added) == len(checks)
    for name, sqltext in checks.items():
        assert f"CONSTRAINT {name} CHECK ({sqltext})" in create_sql or added.get(name) == sqltext, name


def test_model_indexes_equal_ddl_indexes():
    expected = []
    for table_name, table in TABLES.items():
        for index in table.indexes:
            kind = "UNIQUE INDEX" if index.unique else "INDEX"
            columns = ", ".join(_terms(index))
            where = _where(index)
            tail = f" WHERE {where};" if where else ";"
            expected.append(f"CREATE {kind} IF NOT EXISTS {index.name} ON {table_name} ({columns}){tail}")
    declared = [sql for path in MIGRATIONS for sql in _migration_constants(path).get("_INDEXES", ())]
    assert sorted(expected) == sorted(declared)


def test_the_plan_shape_digest_partial_uniques_and_install_columns():
    indexes = {index.name: index for index in ThemePackage.__table__.indexes}
    assert not indexes["ix_theme_packages_digest"].unique
    digest_uniques = [i for i in indexes.values() if i.unique and "digest" in [c.name for c in i.columns]]
    assert all(_where(i) for i in digest_uniques), "no global unique on digest"
    for name, (columns, predicate) in PARTIAL_UNIQUES.items():
        index = indexes[name]
        assert index.unique and tuple(c.name for c in index.columns) == columns and _where(index) == predicate
    install_columns = set(ThemeInstallation.__table__.columns.keys())
    assert {"previous_package_id", "pending_package_id", "pinned_version"} <= install_columns


def test_every_create_is_if_not_exists_and_nothing_is_altered():
    constants = _migration_constants()
    statements = [sql for _name, sql in constants["_TABLES"]] + list(constants["_INDEXES"])
    for sql in statements:
        assert re.match(r"\s*CREATE (UNIQUE )?(TABLE|INDEX) IF NOT EXISTS ", sql), sql
        assert "ALTER" not in sql.upper() and "DROP" not in sql.upper()
    assert "op.create_table" not in MIGRATION.read_text(encoding="utf-8")


def test_the_downgrade_drops_the_tables_in_reverse_order():
    downgrade = _function_source("downgrade")
    assert "reversed(_TABLES)" in downgrade and "DROP TABLE IF EXISTS" in downgrade


def test_the_c05f_shape_origin_and_the_one_sync_row():
    origin = ThemeRevocation.__table__.c.origin
    assert (_pg_type(origin), origin.nullable, origin.server_default.arg) == ("VARCHAR(16)", False, "local")
    assert THEME_REVOCATION_ORIGINS == ("local", "feed")
    sync = ThemeRevocationSync.__table__
    assert tuple(sync.columns.keys()) == SYNC_COLUMNS
    assert [c.name for c in sync.primary_key.columns] == ["id"] and sync.c.id.autoincrement is False
    timestamps = ("last_attempt_at", "last_success_at", "feed_generated_at", "updated_at")
    assert all(sync.c[name].type.timezone for name in timestamps)
    assert not sync.c.revoked_key_ids.nullable and sync.c.revoked_key_ids.server_default.arg == "[]"
    assert THEME_REVOCATION_SYNC_ROW_ID == 1 and not sync.indexes


def test_the_feed_migration_adds_only_what_the_table_lacks():
    constants = _migration_constants(FEED_MIGRATION)
    for _name, sql in constants["_TABLES"]:
        assert re.match(r"\s*CREATE TABLE IF NOT EXISTS ", sql), sql
        assert "ALTER" not in sql.upper() and "DROP" not in sql.upper()
    alters = constants["_ADD_ORIGIN"]
    columns, checks = ADDED_COLUMN.findall(alters), ADDED_CHECK.findall(alters)
    assert [(table, line.split(" ", 1)[0]) for table, line in columns] == [("theme_revocations", "origin")]
    assert [(table, name) for table, name, _body in checks] == [("theme_revocations", "ck_theme_revocations_origin")]
    # Every ALTER sits behind its own IF NOT EXISTS, asked of the table the name resolves to.
    assert alters.count("ALTER TABLE ") == len(columns) + len(checks) == alters.count("IF NOT EXISTS (")
    assert "attrelid = 'theme_revocations'::regclass AND attname = 'origin' AND NOT attisdropped" in alters
    assert "conrelid = 'theme_revocations'::regclass AND conname = 'ck_theme_revocations_origin'" in alters
    assert re.search(r"\bDROP\b", alters, re.I) is None
    assert "op.create_table" not in FEED_MIGRATION.read_text(encoding="utf-8")
    indexes = constants["_INDEXES"]
    assert [CREATED_INDEX.match(sql).groups() for sql in indexes] == [(LIVE_LOCAL_UNIQUE, "theme_revocations")]
    for sql in indexes:
        assert sql.startswith("CREATE UNIQUE INDEX IF NOT EXISTS ") and re.search(r"\b(ALTER|DROP)\b", sql) is None
    upgrade = _function_source("upgrade", FEED_MIGRATION)
    assert "op.execute(_ADD_ORIGIN)" in upgrade and "_TABLES" in upgrade
    # The index's predicate names origin, so it is built after the column is added.
    assert upgrade.index("_ADD_ORIGIN") < upgrade.index("_INDEXES")


def test_the_c05f_shape_one_live_local_revocation_per_target_and_level():
    indexes = {index.name: index for index in ThemeRevocation.__table__.indexes}
    index = indexes[LIVE_LOCAL_UNIQUE]
    assert index.unique and _terms(index) == LIVE_LOCAL_TERMS
    # The same predicate on both dialects: the unit tests' SQLite enforces what Postgres does.
    assert _where(index) == _where(index, "sqlite") == LIVE_LOCAL_PREDICATE
    # The only unique on the table: a target may repeat across origins and lifts.
    assert [name for name, other in indexes.items() if other.unique] == [LIVE_LOCAL_UNIQUE]


def test_the_feed_downgrade_undoes_the_upgrade_in_reverse_order():
    constants = _migration_constants(FEED_MIGRATION)
    alters = constants["_ADD_ORIGIN"]
    indexes = [
        f"DROP INDEX IF EXISTS {CREATED_INDEX.match(sql).group(1)};" for sql in reversed(constants["_INDEXES"])
    ]
    tables = [f"DROP TABLE IF EXISTS {name};" for name, _sql in reversed(constants["_TABLES"])]
    checks = [
        f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name};"
        for table, name, _body in reversed(ADDED_CHECK.findall(alters))
    ]
    columns = [
        f"ALTER TABLE {table} DROP COLUMN IF EXISTS {line.split(' ', 1)[0]};"
        for table, line in reversed(ADDED_COLUMN.findall(alters))
    ]
    assert list(constants["_DOWNGRADE"]) == indexes + tables + checks + columns
    assert "_DOWNGRADE" in _function_source("downgrade", FEED_MIGRATION)


# ---------------------------------------------------------------------------
# SQLite: create_all builds the tables; the uniques hold
# ---------------------------------------------------------------------------


@pytest.fixture
def session():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=list(TABLES.values()))
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _package(workspace_id=None, version="1.0.0", digest=DIGEST, theme_id="acme/ink") -> ThemePackage:
    owner = "workspace" if workspace_id else "marketplace"
    return ThemePackage(
        theme_id=theme_id, version=version, owner_type=owner, workspace_id=workspace_id, digest=digest,
        manifest={"id": theme_id}, file_manifest={}, storage_prefix=f"themes/{theme_id}/{version}/",
    )


def _refused(db, row) -> None:
    db.add(row)
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


def test_a_workspace_holds_a_version_and_a_digest_once(session):
    workspace = uuid.uuid4()
    session.add(_package(workspace))
    session.commit()
    stored = session.query(ThemePackage).one()
    assert (stored.status, stored.scan_verdict, stored.reviewed) == ("pending", "not_run", False)
    _refused(session, _package(workspace, digest="b" * 64))
    _refused(session, _package(workspace, version="1.1.0"))


def test_two_workspaces_may_hold_the_same_digest(session):
    session.add_all([_package(uuid.uuid4()), _package(uuid.uuid4())])
    session.commit()
    assert session.query(ThemePackage).filter_by(digest=DIGEST).count() == 2


def test_a_workspace_row_and_a_marketplace_row_may_hold_the_same_digest(session):
    session.add_all([_package(uuid.uuid4()), _package(None)])
    session.commit()
    assert session.query(ThemePackage).filter_by(digest=DIGEST).count() == 2


def test_the_marketplace_holds_a_version_and_a_digest_once(session):
    session.add(_package(None))
    session.commit()
    _refused(session, _package(None, digest="b" * 64))
    _refused(session, _package(None, version="1.1.0"))


def _owner_mismatch(owner_type, workspace_id) -> ThemePackage:
    package = _package(uuid.uuid4())
    package.owner_type, package.workspace_id = owner_type, workspace_id
    return package


def test_the_owner_and_the_workspace_agree(session):
    _refused(session, _owner_mismatch("workspace", None))
    _refused(session, _owner_mismatch("marketplace", uuid.uuid4()))
    _refused(session, _owner_mismatch("publisher", uuid.uuid4()))


def test_one_active_installation_per_theme_and_one_default_per_workspace(session):
    workspace = uuid.uuid4()
    package, other = _package(workspace), _package(workspace, version="2.0.0", digest="b" * 64, theme_id="acme/ok")
    session.add_all([package, other])
    session.flush()
    old = ThemeInstallation(workspace_id=workspace, theme_id="acme/ink", package_id=package.id, is_default=True)
    session.add(old)
    session.flush()
    old.uninstalled_at = datetime.now(timezone.utc)
    session.flush()
    session.add(ThemeInstallation(workspace_id=workspace, theme_id="acme/ink", package_id=package.id, is_default=True))
    session.commit()
    _refused(session, ThemeInstallation(workspace_id=workspace, theme_id="acme/ink", package_id=package.id))
    _refused(session, ThemeInstallation(workspace_id=workspace, theme_id="acme/ok", package_id=other.id, is_default=True))


def test_one_choice_one_look_set_and_one_opt_in_per_member(session):
    workspace = uuid.uuid4()
    package = _package(workspace)
    session.add(package)
    session.flush()
    rows = (
        lambda: ThemeChoice(workspace_id=workspace, user_id=7, choice="studio"),
        lambda: ThemeLook(workspace_id=workspace, user_id=7, theme_id="acme/ink", looks=[]),
        lambda: ThemeOptIn(workspace_id=workspace, user_id=7, package_id=package.id),
        lambda: ThemePublisher(publisher="acme"),
    )
    session.add_all([make() for make in rows])
    session.commit()
    for make in rows:
        _refused(session, make())


def test_a_revocation_names_a_digest_or_a_theme(session):
    session.add_all([
        ThemeRevocation(digest=DIGEST, level="disable", reason="leaks data"),
        ThemeRevocation(theme_id="acme/ink", version_range="<1.2.0", level="warn", reason="slow"),
    ])
    session.commit()
    _refused(session, ThemeRevocation(level="warn", reason="names nothing"))
    _refused(session, ThemeRevocation(digest=DIGEST, theme_id="acme/ink", level="warn", reason="names both"))
    _refused(session, ThemeRevocation(digest=DIGEST, level="block", reason="unknown level"))


def test_a_revocation_reads_local_unless_the_feed_wrote_it(session):
    session.add_all([
        ThemeRevocation(digest=DIGEST, level="disable", reason="written here"),
        ThemeRevocation(id=uuid.uuid4(), theme_id="acme/ink", level="warn", reason="pulled", origin="feed"),
    ])
    session.commit()
    assert {row.reason: row.origin for row in session.query(ThemeRevocation)} == {
        "written here": "local", "pulled": "feed",
    }
    # A row that names no origin (one written before the column existed) reads 'local'.
    session.execute(
        sa.text("INSERT INTO theme_revocations (id, digest, level, reason) VALUES (:id, :digest, 'warn', 'raw')"),
        {"id": uuid.uuid4().hex, "digest": "b" * 64},
    )
    raw = session.execute(sa.text("SELECT origin FROM theme_revocations WHERE reason = 'raw'")).scalar()
    assert raw == "local"
    session.commit()
    _refused(session, ThemeRevocation(digest=DIGEST, level="warn", reason="unknown origin", origin="hosted"))
    _refused(session, ThemeRevocation(digest=DIGEST, level="warn", reason="no origin", origin=None))


def _live_local_twins(**target):
    """Two live ``local`` rows that name the same target at the same level (only their reasons differ)."""
    return [ThemeRevocation(level="disable", reason=reason, **target) for reason in ("first", "second")]


@pytest.mark.parametrize("target", [
    {"digest": DIGEST}, {"theme_id": "acme/ink"}, {"theme_id": "acme/ink", "version_range": "^1.0.0"},
], ids=["digest", "theme id", "theme id and range"])
def test_one_live_local_revocation_per_target_and_level(session, target):
    first, twin = _live_local_twins(**target)
    session.add(first)
    session.commit()
    _refused(session, twin)  # NULL theme id or range included: read as '', so the twin collides
    session.add_all([
        ThemeRevocation(id=uuid.uuid4(), level="disable", reason="pulled", origin="feed", **target),
        ThemeRevocation(level="disable", reason="lifted", lifted_at=datetime.now(timezone.utc), **target),
        ThemeRevocation(level="warn", reason="another level", **target),
    ])
    session.commit()
    first.lifted_at = datetime.now(timezone.utc)
    session.commit()
    session.add(ThemeRevocation(level="disable", reason="renewed", **target))  # the live one was lifted
    session.commit()
    assert session.query(ThemeRevocation).count() == 5


def test_another_range_or_no_range_is_another_target(session):
    session.add_all([
        ThemeRevocation(theme_id="acme/ink", level="disable", reason="every version"),
        ThemeRevocation(theme_id="acme/ink", version_range="^1.0.0", level="disable", reason="1.x"),
        ThemeRevocation(theme_id="acme/ink", version_range="1.0.0", level="disable", reason="1.0.0"),
        ThemeRevocation(theme_id="acme/ok", level="disable", reason="another theme"),
    ])
    session.commit()
    assert session.query(ThemeRevocation).count() == 4


def test_the_sync_table_holds_one_row(session):
    session.add(ThemeRevocationSync())
    session.commit()
    stored = session.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    assert (stored.last_attempt_at, stored.last_success_at, stored.feed_generated_at, stored.last_error) == (
        None, None, None, None,
    )
    assert stored.revoked_key_ids == [] and stored.updated_at is not None
    stored.revoked_key_ids = [KEY_ID]
    stored.last_success_at = datetime.now(timezone.utc)
    session.commit()
    session.expunge_all()  # read back from the database, and the next rows' keys reach it too
    assert session.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID).revoked_key_ids == [KEY_ID]
    session.expunge_all()
    _refused(session, ThemeRevocationSync())
    _refused(session, ThemeRevocationSync(id=THEME_REVOCATION_SYNC_ROW_ID + 1))
    assert session.query(ThemeRevocationSync).count() == 1


def test_one_settings_row_per_workspace_with_the_defaults(session):
    workspace = uuid.uuid4()
    session.add(ThemeWorkspaceSettings(workspace_id=workspace))
    session.commit()
    stored = session.get(ThemeWorkspaceSettings, workspace)
    assert (stored.private_uploads, stored.members_may_choose, stored.install_role) == (None, True, "workspace:manage")
    session.expunge_all()  # the second row's key reaches the database, not the identity map
    _refused(session, ThemeWorkspaceSettings(workspace_id=workspace))
    _refused(session, ThemeWorkspaceSettings(workspace_id=uuid.uuid4(), private_uploads="admins"))


# ---------------------------------------------------------------------------
# @integration — real Postgres, inside a rolled-back transaction
# ---------------------------------------------------------------------------

_PG_SNAPSHOT = {
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


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM workspaces LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM users LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Postgres checks need the test database: {exc}")
    yield engine
    engine.dispose()


@pytest.fixture
def pg_conn(pg_engine):
    """A connection inside one transaction that is always rolled back."""
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            ThemePackage.metadata.create_all(conn, tables=list(TABLES.values()))
            yield conn
        finally:
            trans.rollback()


def _run_migration(conn, step: str, path: Path = MIGRATION) -> None:
    spec = importlib.util.spec_from_file_location(f"{path.stem}_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with Operations.context(MigrationContext.configure(conn)):
        getattr(module, step)()


def _upgrade_chain(conn) -> None:
    for path in MIGRATIONS:
        _run_migration(conn, "upgrade", path)


def _pg_schema(conn) -> dict:
    """Columns, index definitions and constraints of the ten tables in the current schema, schema name removed."""
    schema = conn.execute(sa.text("SELECT current_schema()")).scalar()
    names = sorted(TABLES)
    snapshot = {}
    for kind, sql in _PG_SNAPSHOT.items():
        rows = conn.execute(sa.text(sql), {"tables": names})
        snapshot[kind] = sorted(tuple(str(value).replace(f"{schema}.", "") for value in row) for row in rows)
    return snapshot


def _tables_in(conn, schema: str) -> set:
    rows = conn.execute(
        sa.text("SELECT table_name FROM information_schema.tables WHERE table_schema = :schema"),
        {"schema": schema},
    )
    return {row[0] for row in rows} & PLANNED_TABLES


def _enter_empty_schema(conn) -> str:
    """A new, empty schema first on the search path (users and workspaces stay reachable)."""
    empty = f"prd261_a05_{uuid.uuid4().hex[:12]}"
    conn.execute(sa.text(f"CREATE SCHEMA {empty}"))
    conn.execute(sa.text(f"SET LOCAL search_path TO {empty}, public"))
    assert _tables_in(conn, empty) == set()
    return empty


@pytest.mark.integration
def test_on_a_create_all_database_the_migrations_change_nothing(pg_conn):
    schema = pg_conn.execute(sa.text("SELECT current_schema()")).scalar()
    assert _tables_in(pg_conn, schema) == PLANNED_TABLES
    before = _pg_schema(pg_conn)
    _upgrade_chain(pg_conn)
    assert _pg_schema(pg_conn) == before


@pytest.mark.integration
def test_on_an_empty_database_the_migrations_build_the_model_schema(pg_conn):
    # The model's schema, as create_all built it; then the migrations alone, in an empty schema
    # of the same database, so no existing table is dropped or locked exclusively. The
    # transaction is rolled back.
    model_schema = _pg_schema(pg_conn)
    empty = _enter_empty_schema(pg_conn)
    _upgrade_chain(pg_conn)
    assert _tables_in(pg_conn, empty) == PLANNED_TABLES
    assert _pg_schema(pg_conn) == model_schema
    _upgrade_chain(pg_conn)  # a second run changes nothing
    assert _pg_schema(pg_conn) == model_schema


_INSERT_REVOCATION = sa.text(
    "INSERT INTO theme_revocations (id, digest, level, reason) VALUES (CAST(:id AS uuid), :digest, 'warn', :reason)"
)


def _pg_origin(conn, revocation_id: str):
    return conn.execute(
        sa.text("SELECT origin FROM theme_revocations WHERE id = CAST(:id AS uuid)"), {"id": revocation_id},
    ).scalar()


@pytest.mark.integration
def test_the_feed_migration_reads_old_rows_local_and_its_downgrade_restores_the_schema(pg_conn):
    model_schema = _pg_schema(pg_conn)
    empty = _enter_empty_schema(pg_conn)
    _run_migration(pg_conn, "upgrade")  # prd261_themes alone: the schema before C-05f
    before = _pg_schema(pg_conn)
    old = str(uuid.uuid4())
    pg_conn.execute(_INSERT_REVOCATION, {"id": old, "digest": DIGEST, "reason": "before C-05f"})
    _run_migration(pg_conn, "upgrade", FEED_MIGRATION)
    assert _tables_in(pg_conn, empty) == PLANNED_TABLES
    assert _pg_schema(pg_conn) == model_schema
    assert _pg_origin(pg_conn, old) == "local"
    _run_migration(pg_conn, "downgrade", FEED_MIGRATION)
    assert _tables_in(pg_conn, empty) == PHASE_A_TABLES
    assert _pg_schema(pg_conn) == before
    assert pg_conn.execute(sa.text("SELECT count(*) FROM theme_revocations")).scalar() == 1  # no row deleted
    _run_migration(pg_conn, "upgrade", FEED_MIGRATION)
    assert _pg_schema(pg_conn) == model_schema


_INSERT_TARGETED = sa.text(
    "INSERT INTO theme_revocations (id, digest, theme_id, version_range, level, reason, origin, lifted_at) "
    "VALUES (CAST(:id AS uuid), :digest, :theme_id, :version_range, 'disable', :reason, :origin, :lifted_at)"
)


def _targeted(target: dict, reason: str, origin: str = "local", lifted_at=None) -> dict:
    values = {"digest": None, "theme_id": None, "version_range": None, **target}
    return {**values, "id": str(uuid.uuid4()), "reason": reason, "origin": origin, "lifted_at": lifted_at}


@pytest.mark.integration
def test_the_live_local_unique_holds_on_postgres(pg_conn):
    _enter_empty_schema(pg_conn)  # the migrations' own index, not create_all's
    _upgrade_chain(pg_conn)
    now = datetime.now(timezone.utc)
    for target in ({"digest": DIGEST}, {"theme_id": "acme/ink"}, {"theme_id": "acme/ink", "version_range": "^1.0.0"}):
        pg_conn.execute(_INSERT_TARGETED, _targeted(target, "first"))
        _pg_refused(pg_conn, _INSERT_TARGETED, _targeted(target, "second"))
        pg_conn.execute(_INSERT_TARGETED, _targeted(target, "pulled", origin="feed"))
        pg_conn.execute(_INSERT_TARGETED, _targeted(target, "lifted", lifted_at=now))
    assert pg_conn.execute(sa.text("SELECT count(*) FROM theme_revocations")).scalar() == 9


@pytest.mark.integration
def test_the_origin_check_and_the_single_row_hold_on_postgres(pg_conn):
    _enter_empty_schema(pg_conn)  # the migrations' own tables: no row another test wrote, no lock on public
    _upgrade_chain(pg_conn)
    set_origin = sa.text("UPDATE theme_revocations SET origin = :origin WHERE id = CAST(:id AS uuid)")
    local, feed = str(uuid.uuid4()), str(uuid.uuid4())
    pg_conn.execute(_INSERT_REVOCATION, {"id": local, "digest": DIGEST, "reason": "written here"})
    pg_conn.execute(_INSERT_REVOCATION, {"id": feed, "digest": "b" * 64, "reason": "pulled"})
    pg_conn.execute(set_origin, {"id": feed, "origin": "feed"})
    assert (_pg_origin(pg_conn, local), _pg_origin(pg_conn, feed)) == ("local", "feed")
    _pg_refused(pg_conn, set_origin, {"id": local, "origin": "hosted"})
    _pg_refused(pg_conn, set_origin, {"id": local, "origin": None})
    insert_sync = sa.text("INSERT INTO theme_revocation_sync (id) VALUES (:id)")
    pg_conn.execute(insert_sync, {"id": THEME_REVOCATION_SYNC_ROW_ID})
    stored = pg_conn.execute(sa.text("SELECT revoked_key_ids, updated_at FROM theme_revocation_sync")).one()
    assert stored.revoked_key_ids == [] and stored.updated_at is not None
    _pg_refused(pg_conn, insert_sync, {"id": THEME_REVOCATION_SYNC_ROW_ID})
    _pg_refused(pg_conn, insert_sync, {"id": THEME_REVOCATION_SYNC_ROW_ID + 1})


_INSERT_PACKAGE = sa.text(
    "INSERT INTO theme_packages (id, theme_id, version, owner_type, workspace_id, digest, manifest, "
    "file_manifest, storage_prefix) VALUES (CAST(:id AS uuid), :theme_id, :version, :owner, "
    "CAST(:ws AS uuid), :digest, CAST('{}' AS jsonb), CAST('{}' AS jsonb), 'themes/test/')"
)
_INSERT_INSTALL = sa.text(
    "INSERT INTO theme_installations (id, workspace_id, theme_id, package_id, is_default) "
    "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :theme_id, CAST(:package AS uuid), :is_default)"
)


def _pg_package(conn, ws=None, version="1.0.0", digest=DIGEST, theme_id="acme/ink") -> str:
    package_id = str(uuid.uuid4())
    conn.execute(_INSERT_PACKAGE, {
        "id": package_id, "theme_id": theme_id, "version": version,
        "owner": "workspace" if ws else "marketplace", "ws": ws, "digest": digest,
    })
    return package_id


def _pg_refused(conn, statement, params) -> None:
    with pytest.raises(IntegrityError):
        with conn.begin_nested():
            conn.execute(statement, params)


def _pg_workspaces(conn, count: int) -> list:
    ids = [str(uuid.uuid4()) for _ in range(count)]
    for ws in ids:
        conn.execute(
            sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :name)"),
            {"id": ws, "name": "prd261-a05"},
        )
    return ids


@pytest.mark.integration
def test_the_package_partial_uniques_hold_on_postgres(pg_conn):
    ws_a, ws_b = _pg_workspaces(pg_conn, 2)
    _pg_package(pg_conn, ws_a)
    _pg_package(pg_conn, ws_b)  # two workspaces, one digest
    _pg_package(pg_conn, None)  # a Marketplace row with that digest too
    base = {"theme_id": "acme/ink", "version": "1.0.0", "digest": DIGEST}
    cases = (
        {"owner": "workspace", "ws": ws_a, "digest": "b" * 64},  # same workspace, theme and version
        {"owner": "workspace", "ws": ws_a, "version": "1.1.0"},  # same workspace and digest
        {"owner": "marketplace", "ws": None, "digest": "b" * 64},  # same Marketplace theme and version
        {"owner": "marketplace", "ws": None, "version": "1.1.0"},  # same Marketplace digest
    )
    for case in cases:
        _pg_refused(pg_conn, _INSERT_PACKAGE, {**base, **case, "id": str(uuid.uuid4())})
    count = pg_conn.execute(sa.text("SELECT count(*) FROM theme_packages WHERE digest = :d"), {"d": DIGEST})
    assert count.scalar() == 3


@pytest.mark.integration
def test_the_installation_partial_uniques_hold_on_postgres(pg_conn):
    (ws,) = _pg_workspaces(pg_conn, 1)
    ink = _pg_package(pg_conn, ws)
    ok = _pg_package(pg_conn, ws, digest="b" * 64, theme_id="acme/ok")
    first = str(uuid.uuid4())
    pg_conn.execute(_INSERT_INSTALL, {"id": first, "ws": ws, "theme_id": "acme/ink", "package": ink, "is_default": True})
    same_theme = {"ws": ws, "theme_id": "acme/ink", "package": ink, "is_default": False}
    second_default = {"ws": ws, "theme_id": "acme/ok", "package": ok, "is_default": True}
    _pg_refused(pg_conn, _INSERT_INSTALL, {**same_theme, "id": str(uuid.uuid4())})
    _pg_refused(pg_conn, _INSERT_INSTALL, {**second_default, "id": str(uuid.uuid4())})
    pg_conn.execute(
        sa.text("UPDATE theme_installations SET uninstalled_at = now() WHERE id = CAST(:id AS uuid)"), {"id": first},
    )
    pg_conn.execute(_INSERT_INSTALL, {**same_theme, "id": str(uuid.uuid4())})
    pg_conn.execute(_INSERT_INSTALL, {**second_default, "id": str(uuid.uuid4())})
