"""PRD-277 Wave 3 (US-301): prd277_w3_sessions_20261120, the wave's one migration.

* The revision chains onto ``prd277_w2_identities_20261120``; Wave 4's revision chains onto it.
* Its upgrade creates ``auth_sessions``, its unique token-hash index and its person index, each
  only when missing, writing no row; its downgrade drops exactly the table.
* The model (``core.models.auth_sessions``) and the DDL declare the same names, sizes and CHECK.
  No column can hold a raw token: the only token column is a SHA-256 hex digest's length.
* On Postgres, in a new empty schema inside a rolled-back transaction: the upgrade over tables
  ``create_all`` built changes nothing, twice; the migration builds what ``create_all`` builds;
  the downgrade keeps every person; a second row with one hash and a half-revoked row are refused;
  erasing a person erases their sessions.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy.exc import IntegrityError

import core.models  # noqa: F401  (registers every table on Base)
from core.database.base import Base
from core.models.auth_sessions import (
    IDP_SESSION_ID_MAX,
    IP_ADDRESS_MAX,
    REVOKED_CHECK,
    REVOKED_CHECK_NAME,
    REVOKED_REASON_MAX,
    SHA256_HEX_LENGTH,
    TOKEN_HASH_INDEX,
    USER_AGENT_MAX,
    USER_INDEX,
    AuthSession,
)
from core.models.core import User

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
REVISION, PARENT = "prd277_w3_sessions_20261120", "prd277_w2_identities_20261120"
WAVE_4 = "prd277_w4_oidc_sign_ins_20261120"
MIGRATION = VERSIONS / f"{REVISION}.py"
TABLE = "auth_sessions"
COLUMNS = ("id", "user_id", "token_hash", "created_at", "last_seen_at", "idle_expires_at",
           "absolute_expires_at", "auth_time", "idp_session_id", "revoked_at", "revoked_reason",
           "ip_address", "user_agent")
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


def test_the_revision_chains_onto_wave_2_and_wave_4_chains_onto_it():
    constants = _constants()
    assert (constants["revision"], constants["down_revision"]) == (REVISION, PARENT)
    revisions = _revisions()
    assert [rev for rev, down in revisions.items() if PARENT in _parents(down)] == [REVISION]
    # PRD-277 Wave 4 chained prd277_w4_oidc_sign_ins_20261120 onto this revision, so the one head
    # is Wave 4's and the head guard names this revision as its parent
    # (tests/test_prd277_w4_migration.py pins that head).
    assert [rev for rev, down in revisions.items() if REVISION in _parents(down)] == [WAVE_4]
    heads = set(revisions) - {parent for down in revisions.values() for parent in _parents(down)}
    assert heads == {WAVE_4}
    single_head = (ORCH / "tests" / "test_prd209_alembic_single_head.py").read_text(encoding="utf-8")
    assert REVISION in single_head and f'EXPECTED_HEAD = "{WAVE_4}"' in single_head
    chain = (ORCH / "tests" / "test_prd236_w1_routes.py").read_text(encoding="utf-8")
    assert f'(versions / "{REVISION}.py")' in chain and f'down_revision = "{PARENT}"' in chain


def test_the_upgrade_builds_the_table_and_its_indexes_and_writes_no_row():
    constants = _constants()
    upgrade = _statements("upgrade")
    assert upgrade == [constants["_TABLE"], *constants["_INDEXES"]]
    assert upgrade[0].strip().startswith(f"CREATE TABLE IF NOT EXISTS {TABLE} (")
    assert constants["_INDEXES"] == (
        f"CREATE UNIQUE INDEX IF NOT EXISTS {TOKEN_HASH_INDEX} ON {TABLE} (token_hash);",
        f"CREATE INDEX IF NOT EXISTS {USER_INDEX} ON {TABLE} (user_id);",
    )
    assert not any(ROW_WRITES.search(statement) for statement in upgrade)
    assert not any("users" in statement and "ALTER" in statement for statement in upgrade)
    assert _statements("downgrade") == [f"DROP TABLE IF EXISTS {TABLE};"]


def _ddl_lines() -> list:
    ddl = _constants()["_TABLE"]
    return [line.strip().rstrip(",") for line in ddl.strip().splitlines()[1:-1]]


def test_the_model_and_the_ddl_declare_the_same_names():
    table = AuthSession.__table__
    assert tuple(table.columns.keys()) == COLUMNS
    assert Base.metadata.tables[TABLE] is table
    lines = [line for line in _ddl_lines() if not line.startswith("CONSTRAINT")]
    assert tuple(line.split(" ", 1)[0] for line in lines) == COLUMNS
    for line in lines:
        name = line.split(" ", 1)[0]
        assert ("NOT NULL" in line or "PRIMARY KEY" in line) == (not table.c[name].nullable), name
    sizes = {"token_hash": SHA256_HEX_LENGTH, "idp_session_id": IDP_SESSION_ID_MAX,
             "revoked_reason": REVOKED_REASON_MAX, "ip_address": IP_ADDRESS_MAX, "user_agent": USER_AGENT_MAX}
    for name, size in sizes.items():
        assert table.c[name].type.length == size, name
        assert any(line.startswith(f"{name} VARCHAR({size})") for line in lines), name
    assert "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE" in lines
    (fk,) = table.c.user_id.foreign_keys
    assert fk.target_fullname == "users.id" and fk.ondelete == "CASCADE"
    indexes = {index.name: (index.unique, tuple(index.columns.keys())) for index in table.indexes}
    assert indexes == {TOKEN_HASH_INDEX: (True, ("token_hash",)), USER_INDEX: (False, ("user_id",))}


def test_the_revoked_check_is_declared_once_by_the_same_name():
    checks = {c.name: str(c.sqltext) for c in AuthSession.__table__.constraints if isinstance(c, sa.CheckConstraint)}
    assert checks == {REVOKED_CHECK_NAME: REVOKED_CHECK}
    assert f"CONSTRAINT {REVOKED_CHECK_NAME} CHECK ({REVOKED_CHECK})" in _ddl_lines()


def test_no_column_can_hold_a_raw_token():
    """The token is ``ams_`` plus 43 URL-safe characters; the only token column holds 64 hex."""
    assert SHA256_HEX_LENGTH == 64
    text_columns = [c for c in AuthSession.__table__.columns if isinstance(c.type, sa.String)]
    assert [c.name for c in text_columns if "token" in c.name] == ["token_hash"]


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
            name = f"prd277_w3_{uuid.uuid4().hex[:12]}"
            conn.execute(sa.text(f"CREATE SCHEMA {name}"))
            conn.execute(sa.text(f"SET LOCAL search_path TO {name}"))
            yield conn
        finally:
            trans.rollback()


def _run(conn, step: str = "upgrade") -> None:
    with Operations.context(MigrationContext.configure(conn)):
        getattr(_load(), step)()


def _create_all(conn, *, sessions: bool = True) -> None:
    tables = [User.__table__, AuthSession.__table__] if sessions else [User.__table__]
    Base.metadata.create_all(conn, tables=tables, checkfirst=False)


_SNAPSHOT = (
    "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
    "FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = :t",
    "SELECT tablename, indexname, regexp_replace(indexdef, ' ON [a-z0-9_]+\\.', ' ON ') FROM pg_indexes "
    "WHERE schemaname = current_schema() AND tablename = :t",
    "SELECT cls.relname, con.conname, pg_get_constraintdef(con.oid) FROM pg_constraint con "
    "JOIN pg_class cls ON cls.oid = con.conrelid JOIN pg_namespace ns ON ns.oid = cls.relnamespace "
    "WHERE ns.nspname = current_schema() AND cls.relname = :t",
)


def _schema(conn) -> list:
    return [sorted(tuple(str(value) for value in row) for row in conn.execute(sa.text(sql), {"t": TABLE}))
            for sql in _SNAPSHOT]


def _has_table(conn) -> bool:
    sql = "SELECT 1 FROM information_schema.tables WHERE table_schema = current_schema() AND table_name = :t"
    return conn.execute(sa.text(sql), {"t": TABLE}).first() is not None


def _person(conn, label: str) -> int:
    email = f"prd277-w3-{label}-{uuid.uuid4().hex[:8]}@example.test"
    return conn.execute(sa.text("INSERT INTO users (username, email, is_active) VALUES (:e, :e, true) RETURNING id"),
                        {"e": email}).scalar_one()


_INSERT = sa.text(
    "INSERT INTO auth_sessions (id, user_id, token_hash, idle_expires_at, absolute_expires_at, auth_time, "
    "revoked_at, revoked_reason) VALUES (:id, :u, :h, :idle, :abs, :at, :ra, :rr)"
)


def _session(conn, user_id: int, digest: str, revoked_at=None, revoked_reason=None) -> None:
    now = datetime.now(timezone.utc)
    conn.execute(_INSERT, {"id": uuid.uuid4(), "u": user_id, "h": digest, "idle": now + timedelta(hours=2),
                           "abs": now + timedelta(hours=12), "at": now, "ra": revoked_at, "rr": revoked_reason})


def _digest() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex


def test_over_create_all_the_upgrade_changes_nothing(pg):
    _create_all(pg)
    by_model = _schema(pg)
    assert len(by_model[1]) == 3                                         # the key and both indexes
    _run(pg)
    _run(pg)
    assert _schema(pg) == by_model


def test_the_migration_builds_what_create_all_builds(pg):
    _create_all(pg)
    by_model = _schema(pg)
    pg.execute(sa.text(f"DROP TABLE {TABLE}"))
    _run(pg)
    assert _schema(pg) == by_model
    _run(pg)
    assert _schema(pg) == by_model


def test_the_downgrade_drops_the_table_and_keeps_every_person(pg):
    _create_all(pg, sessions=False)
    _run(pg)
    kept = _person(pg, "kept")
    _session(pg, kept, _digest())
    _run(pg, "downgrade")
    assert not _has_table(pg)
    assert pg.execute(sa.text("SELECT COUNT(*) FROM users WHERE id = :id"), {"id": kept}).scalar_one() == 1
    _run(pg)
    assert _has_table(pg)


def test_one_hash_names_one_session_and_revoked_needs_its_reason(pg):
    _create_all(pg, sessions=False)
    _run(pg)
    person = _person(pg, "first")
    digest = _digest()
    _session(pg, person, digest)
    _session(pg, person, _digest(), datetime.now(timezone.utc), "sign_out")
    with pytest.raises(IntegrityError):
        with pg.begin_nested():
            _session(pg, person, digest)
    for half in ({"revoked_at": datetime.now(timezone.utc)}, {"revoked_reason": "admin"}):
        with pytest.raises(IntegrityError):
            with pg.begin_nested():
                _session(pg, person, _digest(), **half)


def test_erasing_a_person_erases_their_sessions(pg):
    _create_all(pg, sessions=False)
    _run(pg)
    person = _person(pg, "erased")
    _session(pg, person, _digest())
    pg.execute(sa.text("DELETE FROM users WHERE id = :id"), {"id": person})
    assert pg.execute(sa.text("SELECT COUNT(*) FROM auth_sessions WHERE user_id = :id"),
                      {"id": person}).scalar_one() == 0
