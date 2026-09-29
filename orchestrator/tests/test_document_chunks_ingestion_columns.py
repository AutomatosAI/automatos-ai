"""#825 — every schema path gives document_chunks the columns ingestion writes.

Fresh-built databases had only id / document_id / chunk_index / content /
metadata / created_at (``init_db()``'s vector-free raw DDL), so every document
upload failed. The revision ``document_chunks_ingestion_columns`` adds the rest
wherever they are missing. Proven here against the suite's own Postgres on a
table shaped exactly like the fresh path's, inside a transaction that is rolled
back (Postgres DDL is transactional), so the shared test schema is untouched.
The pgvector ``embedding`` leg needs the ``vector`` type, which the stock CI
Postgres lacks; the alembic-from-zero lane (pgvector) asserts it end to end.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

_REVISION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "document_chunks_ingestion_columns.py"
_FRESH_SHAPE = """
    CREATE TEMPORARY TABLE document_chunks (
        id SERIAL PRIMARY KEY,
        document_id INTEGER,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        metadata JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMP DEFAULT NOW()
    )
"""
_INGESTION_COLUMNS = {"parent_content", "headers", "chunk_type", "workspace_id"}


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — no database here: skip, don't fail
        pytest.skip(f"needs a reachable Postgres: {exc}")
    return eng


def _upgrade_sql(monkeypatch) -> list[str]:
    """The SQL the revision's upgrade() runs, captured instead of executed."""
    from alembic import op

    spec = importlib.util.spec_from_file_location("document_chunks_ingestion_columns", _REVISION)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    captured: list[str] = []
    monkeypatch.setattr(op, "execute", captured.append, raising=False)
    revision.upgrade()
    return captured


def _columns(conn) -> dict[str, str]:
    rows = conn.execute(text(
        "SELECT attname, format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = to_regclass('document_chunks') AND attnum > 0 AND NOT attisdropped"
    )).fetchall()
    return {name: kind for name, kind in rows}


def test_the_revision_adds_what_ingestion_writes_and_is_idempotent(engine, monkeypatch):
    statements = _upgrade_sql(monkeypatch)
    with engine.connect() as conn:
        tx = conn.begin()
        try:
            # A temporary table shadows any real document_chunks for this session only.
            conn.execute(text(_FRESH_SHAPE))
            has_vector = conn.execute(text("SELECT 1 FROM pg_type WHERE typname = 'vector'")).scalar() is not None
            for _ in range(2):  # the second run must be a no-op, not an error
                for sql in statements:
                    conn.execute(text(sql))
            columns = _columns(conn)
            assert _INGESTION_COLUMNS <= set(columns), columns
            assert columns["workspace_id"] == "uuid"
            assert ("embedding" in columns) == has_vector
            if has_vector:
                assert columns["embedding"] == "vector"  # no fixed dimension: any configured size fits
            # ingestion's own row shape now inserts (S3-vectors mode: no embedding)
            conn.execute(text(
                "INSERT INTO document_chunks (document_id, chunk_index, content, metadata, parent_content, headers, workspace_id) "
                "VALUES (NULL, 0, 'x', '{}', 'parent', '{}', '00000000-0000-0000-0000-0000000000c1')"
            ))
            assert conn.execute(text("SELECT chunk_type FROM document_chunks")).scalar() == "child"
        finally:
            tx.rollback()


def test_the_revision_is_a_no_op_without_the_table(engine, monkeypatch):
    statements = _upgrade_sql(monkeypatch)
    with engine.connect() as conn:
        tx = conn.begin()
        try:
            conn.execute(text("SET LOCAL search_path TO pg_temp"))  # no document_chunks visible
            for sql in statements:
                conn.execute(text(sql))
        finally:
            tx.rollback()
