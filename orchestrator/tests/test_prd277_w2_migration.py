"""PRD-277 Wave 2 (US-201, US-203): prd277_w2_identities_20261120, the wave's one migration.

* The revision chains onto ``merge_heads_20261117``, and Wave 3's revision chains onto it.
* Its upgrade creates ``user_identities`` and its two indexes and adds the nullable
  ``users.system_role`` with its CHECK, each only when missing and the CHECK ``NOT VALID`` (no
  scan of ``users``), writing no row; its downgrade drops exactly those.
* The model (``core.models.user_identities``) and the DDL declare the same names, sizes and CHECK,
  and the stored roles are the policy plane's three install-wide roles.
* On Postgres, in a new empty schema inside a rolled-back transaction: the upgrade over tables
  ``create_all`` built changes nothing, twice (its CHECK stays validated, so it was not added
  again); the migration builds what ``create_all`` builds, its CHECK added ``NOT VALID``;
  the downgrade keeps every person; an unknown role and a second link for one pair are refused.
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

import core.models  # noqa: F401  (registers every table on Base)
from core.database.base import Base
from core.models.core import User
from core.models.user_identities import (
    IDENTITY_PAIR_INDEX,
    IDENTITY_USER_INDEX,
    ISSUER_MAX,
    SUBJECT_MAX,
    SYSTEM_ROLE_CHECK,
    SYSTEM_ROLE_CHECK_NAME,
    SYSTEM_ROLE_MAX,
    SYSTEM_ROLES,
    UserIdentity,
)
from modules.policy.roles import ADMIN, SUPER_ADMIN, USER

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
REVISION, PARENT = "prd277_w2_identities_20261120", "merge_heads_20261117"
WAVE_3 = "prd277_w3_sessions_20261120"
HEAD = "prd277_w6b_phone_sign_in_20261121"  # Wave 6 part 2's revision, on Wave 4 part 2's
MIGRATION = VERSIONS / f"{REVISION}.py"
TABLE = "user_identities"
COLUMNS = ("id", "user_id", "issuer", "subject", "created_at", "updated_at")
ROW_WRITES = re.compile(r"\b(DELETE FROM|TRUNCATE|UPDATE \w+ SET|INSERT INTO)\b")


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


def test_the_revision_chains_onto_the_merge_and_wave_3_chains_onto_it():
    constants = _constants()
    assert (constants["revision"], constants["down_revision"]) == (REVISION, PARENT)
    revisions = _revisions()
    assert [rev for rev, down in revisions.items() if PARENT in _parents(down)] == [REVISION]
    # PRD-277 Wave 3 chained prd277_w3_sessions_20261120 onto this revision, and Wave 4 chained
    # prd277_w4_oidc_sign_ins_20261120 onto Wave 3's and prd277_w4b_step_up_20261120 onto that, and
    # Wave 6 part 2 prd277_w6b_phone_sign_in_20261121 onto that, so the one head is Wave 6 part 2's and
    # the head pins name it (tests/test_prd277_w6b_phone_sign_in_migration.py pins it).
    assert [rev for rev, down in revisions.items() if REVISION in _parents(down)] == [WAVE_3]
    heads = set(revisions) - {parent for down in revisions.values() for parent in _parents(down)}
    assert heads == {HEAD}
    chain = (ORCH / "tests" / "test_prd236_w1_routes.py").read_text(encoding="utf-8")
    assert f'(versions / "{REVISION}.py")' in chain and f'down_revision = "{PARENT}"' in chain


def test_the_upgrade_builds_the_table_indexes_and_column_and_writes_no_row():
    constants = _constants()
    upgrade = _statements("upgrade")
    assert upgrade == [constants["_TABLE"], *constants["_INDEXES"], constants["_SYSTEM_ROLE"]]
    assert upgrade[0].strip().startswith(f"CREATE TABLE IF NOT EXISTS {TABLE} (")
    assert constants["_INDEXES"] == (
        f"CREATE UNIQUE INDEX IF NOT EXISTS {IDENTITY_PAIR_INDEX} ON {TABLE} (issuer, subject);",
        f"CREATE INDEX IF NOT EXISTS {IDENTITY_USER_INDEX} ON {TABLE} (user_id);",
    )
    assert f"ALTER TABLE users ADD COLUMN system_role VARCHAR({SYSTEM_ROLE_MAX});" in constants["_SYSTEM_ROLE"]
    assert not any(ROW_WRITES.search(statement) for statement in upgrade)
    assert _statements("downgrade") == [
        f"DROP TABLE IF EXISTS {TABLE};", "ALTER TABLE users DROP COLUMN IF EXISTS system_role;",
    ]


def test_the_model_and_the_ddl_declare_the_same_names():
    table = UserIdentity.__table__
    assert tuple(table.columns.keys()) == COLUMNS
    assert Base.metadata.tables[TABLE] is table
    ddl = _constants()["_TABLE"]
    lines = [line.strip().rstrip(",") for line in ddl.strip().splitlines()[1:-1]]
    assert tuple(line.split(" ", 1)[0] for line in lines) == COLUMNS
    for line in lines:
        name = line.split(" ", 1)[0]
        assert ("NOT NULL" in line or "PRIMARY KEY" in line) == (not table.c[name].nullable), name
    assert f"issuer VARCHAR({ISSUER_MAX}) NOT NULL" in lines and table.c.issuer.type.length == ISSUER_MAX
    assert f"subject VARCHAR({SUBJECT_MAX}) NOT NULL" in lines and table.c.subject.type.length == SUBJECT_MAX
    assert "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE" in lines
    (fk,) = table.c.user_id.foreign_keys
    assert fk.target_fullname == "users.id" and fk.ondelete == "CASCADE"
    indexes = {index.name: (index.unique, tuple(index.columns.keys())) for index in table.indexes}
    assert indexes == {IDENTITY_PAIR_INDEX: (True, ("issuer", "subject")), IDENTITY_USER_INDEX: (False, ("user_id",))}


def test_users_system_role_is_nullable_and_checked_to_the_three_install_wide_roles():
    column = User.__table__.c.system_role
    assert column.nullable and column.type.length == SYSTEM_ROLE_MAX and column.server_default is None
    checks = {c.name: str(c.sqltext) for c in User.__table__.constraints if isinstance(c, sa.CheckConstraint)}
    assert checks == {SYSTEM_ROLE_CHECK_NAME: SYSTEM_ROLE_CHECK}
    assert SYSTEM_ROLES == (SUPER_ADMIN, ADMIN, USER)
    assert SYSTEM_ROLE_CHECK == "system_role IS NULL OR system_role IN ('super_admin', 'admin', 'user')"
    block = _constants()["_SYSTEM_ROLE"]
    assert f"ADD CONSTRAINT {SYSTEM_ROLE_CHECK_NAME} CHECK ({SYSTEM_ROLE_CHECK}) NOT VALID;" in block
    assert "DROP CONSTRAINT" not in block, "an existing CHECK is kept, never dropped and validated again"
    assert "attname = 'system_role' AND NOT attisdropped" in block
    assert f"conrelid = 'users'::regclass AND conname = '{SYSTEM_ROLE_CHECK_NAME}'" in block


# --------------------------------------------------------------------------- Postgres (CI)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def pg(engine):
    """A connection in a new empty schema, inside a transaction that is always rolled back."""
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            name = f"prd277_w2_{uuid.uuid4().hex[:12]}"
            conn.execute(sa.text(f"CREATE SCHEMA {name}"))
            conn.execute(sa.text(f"SET LOCAL search_path TO {name}"))
            yield conn
        finally:
            trans.rollback()


def _run(conn, step: str = "upgrade") -> None:
    with Operations.context(MigrationContext.configure(conn)):
        getattr(_load(), step)()


def _create_all(conn) -> None:
    Base.metadata.create_all(conn, tables=[User.__table__, UserIdentity.__table__], checkfirst=False)


_SNAPSHOT = (
    "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
    "FROM information_schema.columns WHERE table_schema = current_schema() AND table_name IN ('users', :t)",
    "SELECT tablename, indexname, regexp_replace(indexdef, ' ON [a-z0-9_]+\\.', ' ON ') FROM pg_indexes "
    "WHERE schemaname = current_schema() AND tablename IN ('users', :t)",
    "SELECT cls.relname, con.conname, pg_get_constraintdef(con.oid) FROM pg_constraint con "
    "JOIN pg_class cls ON cls.oid = con.conrelid JOIN pg_namespace ns ON ns.oid = cls.relnamespace "
    "WHERE ns.nspname = current_schema() AND cls.relname IN ('users', :t)",
)


def _schema(conn) -> list:
    return [sorted(tuple(str(value) for value in row) for row in conn.execute(sa.text(sql), {"t": TABLE}))
            for sql in _SNAPSHOT]


def _has(conn, table: str, column: str = None) -> bool:
    sql = "SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = :t"
    params = {"t": table}
    if column:
        sql += " AND column_name = :c"
        params["c"] = column
    return conn.execute(sa.text(sql), params).first() is not None


def _check_validated(conn) -> bool:
    return conn.execute(sa.text(
        "SELECT con.convalidated FROM pg_constraint con JOIN pg_class cls ON cls.oid = con.conrelid "
        "JOIN pg_namespace ns ON ns.oid = cls.relnamespace "
        "WHERE ns.nspname = current_schema() AND cls.relname = 'users' AND con.conname = :n"),
        {"n": SYSTEM_ROLE_CHECK_NAME},
    ).scalar_one()


def _as_validated(schema: list) -> list:
    """The snapshot with the CHECK's ``NOT VALID`` marker taken off, to compare everything else."""
    columns, indexes, constraints = schema
    return [columns, indexes, sorted(tuple(v.removesuffix(" NOT VALID") for v in row) for row in constraints)]


def test_over_create_all_the_upgrade_changes_nothing_and_keeps_the_models_check(pg):
    _create_all(pg)
    by_model = _schema(pg)
    assert any(row[1] == "system_role" for row in by_model[0])          # non-vacuous: the model has it
    _run(pg)
    _run(pg)
    assert _schema(pg) == by_model
    assert _check_validated(pg) is True, "the model's CHECK was kept, not dropped and added again"


def test_the_migration_builds_what_create_all_builds_with_its_check_not_valid(pg):
    _create_all(pg)
    by_model = _schema(pg)
    pg.execute(sa.text(f"DROP TABLE {TABLE}"))
    pg.execute(sa.text("ALTER TABLE users DROP COLUMN system_role"))
    _run(pg)
    migrated = _schema(pg)
    assert _check_validated(pg) is False, "added NOT VALID: no scan of the existing rows"
    assert _as_validated(migrated) == by_model
    assert migrated != by_model                                         # only the NOT VALID marker differs
    _run(pg)
    assert _schema(pg) == migrated
    person = _person(pg, "checked")
    with pytest.raises(IntegrityError):
        with pg.begin_nested():
            pg.execute(sa.text("UPDATE users SET system_role = 'owner' WHERE id = :id"), {"id": person})


def _person(conn, label: str) -> int:
    email = f"prd277-w2-{label}-{uuid.uuid4().hex[:8]}@example.test"
    return conn.execute(sa.text("INSERT INTO users (username, email, is_active) VALUES (:e, :e, true) RETURNING id"),
                        {"e": email}).scalar_one()


def test_the_downgrade_drops_the_table_and_the_column_and_keeps_every_person(pg):
    _create_all(pg)
    kept = _person(pg, "kept")
    _run(pg, "downgrade")
    assert not _has(pg, TABLE) and not _has(pg, "users", "system_role")
    assert pg.execute(sa.text("SELECT COUNT(*) FROM users WHERE id = :id"), {"id": kept}).scalar_one() == 1
    _run(pg)
    assert _has(pg, TABLE) and _has(pg, "users", "system_role")


def test_an_unknown_role_and_a_second_link_for_one_pair_are_refused(pg):
    _create_all(pg)
    _run(pg)
    first, second = _person(pg, "first"), _person(pg, "second")
    for role in SYSTEM_ROLES:
        pg.execute(sa.text("UPDATE users SET system_role = :r WHERE id = :id"), {"r": role, "id": first})
    with pytest.raises(IntegrityError):
        with pg.begin_nested():
            pg.execute(sa.text("UPDATE users SET system_role = 'owner' WHERE id = :id"), {"id": first})
    link = "INSERT INTO user_identities (user_id, issuer, subject) VALUES (:u, 'https://idp.example.test', :s)"
    pg.execute(sa.text(link), {"u": first, "s": "Sub-1"})
    pg.execute(sa.text(link), {"u": second, "s": "sub-1"})                  # another case is another subject
    with pytest.raises(IntegrityError):
        with pg.begin_nested():
            pg.execute(sa.text(link), {"u": second, "s": "Sub-1"})
