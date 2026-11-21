"""PRD-277 Wave 4 part 2 (US-406): prd277_w4b_step_up_20261120, the part's one migration.

* The revision chains onto ``prd277_w4_oidc_sign_ins_20261120``; Wave 6 part 2's
  ``prd277_w6b_phone_sign_in_20261121`` chains onto it and is the one head both head pins name.
* Its upgrade adds ``oidc_pending_sign_ins.step_up_session_id`` (a nullable UUID) only when missing,
  writing no row, adding no key and touching no other table; its downgrade drops exactly that column.
* The model declares the same column: nullable, a UUID, no foreign key.
* On Postgres, in a new empty schema inside a rolled-back transaction: over a table ``create_all``
  built the upgrade changes nothing, twice; over the first part's table it builds what
  ``create_all`` builds; a row written before it reads NULL; the downgrade drops the column and
  the upgrade adds it back.
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
from sqlalchemy.dialects.postgresql import UUID as PGUUID

import core.models  # noqa: F401  (registers every table on Base)
from core.database.base import Base
from core.models.oidc_pending_sign_ins import OidcPendingSignIn

ORCH = Path(__file__).resolve().parents[1]
VERSIONS = ORCH / "alembic" / "versions"
REVISION, PARENT = "prd277_w4b_step_up_20261120", "prd277_w4_oidc_sign_ins_20261120"
MIGRATION = VERSIONS / f"{REVISION}.py"
TABLE, COLUMN = "oidc_pending_sign_ins", "step_up_session_id"
ROW_WRITES = re.compile(r"\b(DELETE FROM|TRUNCATE|UPDATE \w+ SET|INSERT INTO)\b")
# Wave 6 part 2 (US-603) chains onto this revision and adds four more columns to the table.
PHONE_SIGN_IN = "prd277_w6b_phone_sign_in_20261121"


def _constants(path: Path = MIGRATION) -> dict:
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return found


def _load(path: Path = MIGRATION):
    spec = importlib.util.spec_from_file_location(f"{path.stem}_{uuid.uuid4().hex[:8]}", path)
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


def test_the_revision_chains_onto_part_1_and_wave_6_part_2_chains_onto_it():
    constants = _constants()
    assert (constants["revision"], constants["down_revision"]) == (REVISION, PARENT)
    revisions = _revisions()
    assert [rev for rev, down in revisions.items() if PARENT in _parents(down)] == [REVISION]
    # PRD-277 Wave 6 part 2 chained prd277_w6b_phone_sign_in_20261121 onto this revision, so the one
    # head is Wave 6 part 2's (tests/test_prd277_w6b_phone_sign_in_migration.py pins it).
    assert [rev for rev, down in revisions.items() if REVISION in _parents(down)] == [PHONE_SIGN_IN]
    heads = set(revisions) - {parent for down in revisions.values() for parent in _parents(down)}
    assert heads == {PHONE_SIGN_IN}
    single_head = (ORCH / "tests" / "test_prd209_alembic_single_head.py").read_text(encoding="utf-8")
    assert REVISION in single_head and f'EXPECTED_HEAD = "{PHONE_SIGN_IN}"' in single_head
    chain = (ORCH / "tests" / "test_prd236_w1_routes.py").read_text(encoding="utf-8")
    assert f'(versions / "{REVISION}.py")' in chain and f'down_revision = "{PARENT}"' in chain


def test_the_upgrade_adds_one_nullable_column_and_writes_no_row():
    upgrade = _statements("upgrade")
    assert upgrade == [f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS {COLUMN} UUID;"]
    assert not any(ROW_WRITES.search(statement) for statement in upgrade)
    assert not any("REFERENCES" in statement or "NOT NULL" in statement or "DEFAULT" in statement
                   for statement in upgrade)
    assert _statements("downgrade") == [f"ALTER TABLE {TABLE} DROP COLUMN IF EXISTS {COLUMN};"]


def test_the_model_declares_the_same_column():
    column = OidcPendingSignIn.__table__.c[COLUMN]
    assert column.nullable and isinstance(column.type, PGUUID) and not column.foreign_keys
    assert Base.metadata.tables[TABLE].c[COLUMN] is column


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
            name = f"prd277_w4b_{uuid.uuid4().hex[:12]}"
            conn.execute(sa.text(f"CREATE SCHEMA {name}"))
            conn.execute(sa.text(f"SET LOCAL search_path TO {name}"))
            yield conn
        finally:
            trans.rollback()


def _run(conn, step: str = "upgrade", path: Path = MIGRATION) -> None:
    with Operations.context(MigrationContext.configure(conn)):
        getattr(_load(path), step)()


_SNAPSHOT = (
    "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
    "FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = :t",
    "SELECT tablename, indexname, regexp_replace(indexdef, ' ON [a-z0-9_]+\\.', ' ON ') FROM pg_indexes "
    "WHERE schemaname = current_schema() AND tablename = :t",
)


def _schema(conn) -> list:
    return [sorted(tuple(str(value) for value in row) for row in conn.execute(sa.text(sql), {"t": TABLE}))
            for sql in _SNAPSHOT]


def _columns(conn) -> set:
    sql = "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = :t"
    return {row[0] for row in conn.execute(sa.text(sql), {"t": TABLE})}


def test_over_create_all_the_upgrade_changes_nothing(pg):
    Base.metadata.create_all(pg, tables=[OidcPendingSignIn.__table__], checkfirst=False)
    by_model = _schema(pg)
    _run(pg)
    _run(pg)
    assert _schema(pg) == by_model


def test_over_part_1_it_builds_what_create_all_builds_and_old_rows_read_null(pg):
    Base.metadata.create_all(pg, tables=[OidcPendingSignIn.__table__], checkfirst=False)
    by_model = _schema(pg)
    pg.execute(sa.text(f"DROP TABLE {TABLE}"))
    _run(pg, path=VERSIONS / f"{PARENT}.py")
    pg.execute(sa.text(
        "INSERT INTO oidc_pending_sign_ins (id, state_hash, nonce_hash, binding_hash, client_hash, code_verifier, "
        "return_to, expires_at) VALUES (:id, :s, :s, :s, :s, :s, '/', :e)"),
        {"id": uuid.uuid4(), "s": uuid.uuid4().hex * 2, "e": datetime.now(timezone.utc) + timedelta(minutes=10)})
    _run(pg)
    for statement in _load(VERSIONS / f"{PHONE_SIGN_IN}.py")._UPGRADE:  # the model has its four columns
        if f"ALTER TABLE {TABLE} " in statement:
            pg.execute(sa.text(statement))
    assert _schema(pg) == by_model
    assert pg.execute(sa.text(f"SELECT {COLUMN} FROM {TABLE}")).scalar_one() is None


def test_the_downgrade_drops_the_column_and_the_upgrade_adds_it_back(pg):
    Base.metadata.create_all(pg, tables=[OidcPendingSignIn.__table__], checkfirst=False)
    _run(pg, "downgrade")
    assert COLUMN not in _columns(pg)
    _run(pg)
    assert COLUMN in _columns(pg)
