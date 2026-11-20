"""PRD-277 Wave 4 (US-402): prd277_w4_oidc_sign_ins_20261120, the first part's one migration.

* The revision chains onto ``prd277_w3_sessions_20261120`` and is the one head; both head pins
  name it.
* Its upgrade creates ``oidc_pending_sign_ins``, its unique state-hash index and its expiry index,
  each only when missing, writing no row and touching no other table; its downgrade drops exactly
  the table.
* The model (``core.models.oidc_pending_sign_ins``) and the DDL declare the same names and sizes.
  No column can hold a raw state or nonce: each is a SHA-256 hex digest's length.
* On Postgres, in a new empty schema inside a rolled-back transaction: the upgrade over a table
  ``create_all`` built changes nothing, twice; the migration builds what ``create_all`` builds;
  one state hash names one row; the downgrade drops the table and the upgrade rebuilds it.
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
from core.models.oidc_pending_sign_ins import (
    CODE_VERIFIER_MAX,
    EXPIRES_INDEX,
    RETURN_TO_MAX,
    SHA256_HEX_LENGTH,
    STATE_HASH_INDEX,
    OidcPendingSignIn,
)

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
REVISION, PARENT = "prd277_w4_oidc_sign_ins_20261120", "prd277_w3_sessions_20261120"
MIGRATION = VERSIONS / f"{REVISION}.py"
TABLE = "oidc_pending_sign_ins"
COLUMNS = ("id", "state_hash", "nonce_hash", "binding_hash", "code_verifier", "return_to", "created_at", "expires_at")
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


def _parents(down) -> list:
    return list(down) if isinstance(down, (list, tuple)) else [down] if down else []


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


def test_the_revision_chains_onto_wave_3_and_is_the_one_head():
    constants = _constants()
    assert (constants["revision"], constants["down_revision"]) == (REVISION, PARENT)
    revisions = _revisions()
    assert [rev for rev, down in revisions.items() if PARENT in _parents(down)] == [REVISION]
    heads = set(revisions) - {parent for down in revisions.values() for parent in _parents(down)}
    assert heads == {REVISION}
    single_head = (ORCH / "tests" / "test_prd209_alembic_single_head.py").read_text(encoding="utf-8")
    assert f'EXPECTED_HEAD = "{REVISION}"' in single_head
    chain = (ORCH / "tests" / "test_prd236_w1_routes.py").read_text(encoding="utf-8")
    assert f'(versions / "{REVISION}.py")' in chain and f'down_revision = "{PARENT}"' in chain


def test_the_upgrade_builds_the_table_and_its_indexes_and_writes_no_row():
    constants = _constants()
    upgrade = _statements("upgrade")
    assert upgrade == [constants["_TABLE"], *constants["_INDEXES"]]
    assert upgrade[0].strip().startswith(f"CREATE TABLE IF NOT EXISTS {TABLE} (")
    assert constants["_INDEXES"] == (
        f"CREATE UNIQUE INDEX IF NOT EXISTS {STATE_HASH_INDEX} ON {TABLE} (state_hash);",
        f"CREATE INDEX IF NOT EXISTS {EXPIRES_INDEX} ON {TABLE} (expires_at);",
    )
    assert not any(ROW_WRITES.search(statement) for statement in upgrade)
    assert not any("ALTER" in statement or "REFERENCES" in statement for statement in upgrade)
    assert _statements("downgrade") == [f"DROP TABLE IF EXISTS {TABLE};"]


def _ddl_lines() -> list:
    ddl = _constants()["_TABLE"]
    return [line.strip().rstrip(",") for line in ddl.strip().splitlines()[1:-1]]


def test_the_model_and_the_ddl_declare_the_same_names_and_sizes():
    table = OidcPendingSignIn.__table__
    assert tuple(table.columns.keys()) == COLUMNS
    assert Base.metadata.tables[TABLE] is table
    lines = _ddl_lines()
    assert [line.split()[0] for line in lines] == list(COLUMNS)
    sizes = {"state_hash": SHA256_HEX_LENGTH, "nonce_hash": SHA256_HEX_LENGTH, "binding_hash": SHA256_HEX_LENGTH,
             "code_verifier": CODE_VERIFIER_MAX, "return_to": RETURN_TO_MAX}
    for name, size in sizes.items():
        assert table.c[name].type.length == size, name
        assert any(line.startswith(f"{name} VARCHAR({size}) NOT NULL") for line in lines), name
    assert not table.foreign_keys
    indexes = {index.name: (index.unique, tuple(index.columns.keys())) for index in table.indexes}
    assert indexes == {STATE_HASH_INDEX: (True, ("state_hash",)), EXPIRES_INDEX: (False, ("expires_at",))}


def test_no_column_can_hold_a_raw_state_nonce_or_binding():
    """A state, a nonce and a binding are 43 URL-safe characters; their only columns hold 64 hex."""
    assert SHA256_HEX_LENGTH == 64
    text_columns = [c.name for c in OidcPendingSignIn.__table__.columns if isinstance(c.type, sa.String)]
    secrets_kept = [name for name in text_columns if any(word in name for word in ("state", "nonce", "binding"))]
    assert secrets_kept == ["state_hash", "nonce_hash", "binding_hash"]


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
            name = f"prd277_w4_{uuid.uuid4().hex[:12]}"
            conn.execute(sa.text(f"CREATE SCHEMA {name}"))
            conn.execute(sa.text(f"SET LOCAL search_path TO {name}"))
            yield conn
        finally:
            trans.rollback()


def _run(conn, step: str = "upgrade") -> None:
    with Operations.context(MigrationContext.configure(conn)):
        getattr(_load(), step)()


_SNAPSHOT = (
    "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
    "FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = :t",
    "SELECT tablename, indexname, regexp_replace(indexdef, ' ON [a-z0-9_]+\\.', ' ON ') FROM pg_indexes "
    "WHERE schemaname = current_schema() AND tablename = :t",
)


def _schema(conn) -> list:
    return [sorted(tuple(str(value) for value in row) for row in conn.execute(sa.text(sql), {"t": TABLE}))
            for sql in _SNAPSHOT]


def _has_table(conn) -> bool:
    sql = "SELECT 1 FROM information_schema.tables WHERE table_schema = current_schema() AND table_name = :t"
    return conn.execute(sa.text(sql), {"t": TABLE}).first() is not None


def _pending(conn, state_hash: str) -> None:
    now = datetime.now(timezone.utc)
    conn.execute(sa.text(
        "INSERT INTO oidc_pending_sign_ins (id, state_hash, nonce_hash, binding_hash, code_verifier, return_to, "
        "expires_at) VALUES (:id, :s, :n, :b, :v, '/', :e)"),
        {"id": uuid.uuid4(), "s": state_hash, "n": uuid.uuid4().hex * 2, "b": uuid.uuid4().hex * 2,
         "v": uuid.uuid4().hex * 2,
         "e": now + timedelta(minutes=10)})


def test_over_create_all_the_upgrade_changes_nothing(pg):
    Base.metadata.create_all(pg, tables=[OidcPendingSignIn.__table__], checkfirst=False)
    by_model = _schema(pg)
    assert len(by_model[1]) == 3                                         # the key and both indexes
    _run(pg)
    _run(pg)
    assert _schema(pg) == by_model


def test_the_migration_builds_what_create_all_builds(pg):
    Base.metadata.create_all(pg, tables=[OidcPendingSignIn.__table__], checkfirst=False)
    by_model = _schema(pg)
    pg.execute(sa.text(f"DROP TABLE {TABLE}"))
    _run(pg)
    assert _schema(pg) == by_model


def test_one_state_hash_names_one_row_and_the_downgrade_drops_the_table(pg):
    _run(pg)
    digest = uuid.uuid4().hex * 2
    _pending(pg, digest)
    with pytest.raises(IntegrityError):
        with pg.begin_nested():
            _pending(pg, digest)
    _run(pg, "downgrade")
    assert not _has_table(pg)
    _run(pg)
    assert _has_table(pg)
