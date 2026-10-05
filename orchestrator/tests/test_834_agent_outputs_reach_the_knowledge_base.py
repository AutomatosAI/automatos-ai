"""#834 — DocumentManager.upload_document() writes the model's ``documents`` columns.

The manager looked an upload up by ``file_hash`` and stored ``metadata``; the
model (``core.models.core.Document``) and every database have ``content_hash``
and ``doc_metadata``. Every upload through the manager — the knowledge
flywheel's agent outputs, mission exports, intake-wizard pages, cloud sync,
Auto's document tool — failed with ``column "file_hash" does not exist``.

The lookup is also scoped: a processed copy is reused only within the caller's
workspace and provenance scope, and a copy that failed is never deleted.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from types import SimpleNamespace as NS

import pytest

_SQL_PARAM_COLUMN = re.compile(r"\b(\w+)\s*(?:=|IS NOT DISTINCT FROM)\s*%s")
_INSERT_COLUMNS = re.compile(r"INSERT INTO documents\s*\(([^)]*)\)", re.S)


def _manager(mgr, workspace_id, processed):
    """A DocumentManager without __init__ (no S3, no embedding provider)."""
    manager = mgr.DocumentManager.__new__(mgr.DocumentManager)
    manager.db_config, manager.workspace_id, manager.s3_bucket = {}, workspace_id, "docs"
    manager.processor = NS(detect_file_type=lambda _path: mgr.DocumentType.MARKDOWN)
    manager._upload_to_s3 = lambda _path, document_id, filename: f"ws/{document_id}_{filename}"

    async def _process(document_id, file_path, file_type, s3_key=None, filename=None):
        processed.append((document_id, file_type, s3_key, filename))

    manager._process_document = _process
    return manager


def _upload(manager, path, source_type=None):
    return asyncio.run(manager.upload_document(
        str(path), filename="report.md", tags=["agent_output"], description="weekly",
        created_by="agent:7", source_type=source_type))


def test_every_documents_column_the_upload_names_is_on_the_model(monkeypatch, tmp_path):
    from core.models.core import Document
    from modules.rag.ingestion import manager as mgr

    statements = []

    class Cursor:
        def execute(self, sql, params=None):
            statements.append(" ".join(sql.split()))

        def fetchone(self):
            return (7,) if statements[-1].startswith("INSERT INTO documents") else None

        def close(self):
            pass

    class Conn:
        def cursor(self, *_a, **_kw):
            return Cursor()

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(mgr.psycopg2, "connect", lambda **_kw: Conn())
    path = tmp_path / "report.md"
    path.write_text("# Weekly report\n\nRevenue rose.\n")
    assert _upload(_manager(mgr, "ws-1", []), path) == 7

    model_columns = set(Document.__table__.columns.keys())
    on_documents = [sql for sql in statements if re.search(r"\b(FROM|INTO|UPDATE) documents\b", sql)]
    named = {col for sql in on_documents for col in _SQL_PARAM_COLUMN.findall(sql)}
    inserted = {c.strip() for c in _INSERT_COLUMNS.search(" ".join(on_documents)).group(1).split(",")}
    assert {"content_hash", "workspace_id"} <= named
    assert {"content_hash", "doc_metadata"} <= inserted
    assert (named | inserted) - model_columns == set()


class _SharedConnection:
    """The test transaction's own DBAPI connection, handed to the manager so its
    writes stay inside the transaction ``db_session`` rolls back."""

    def __init__(self, raw):
        self._raw = raw

    def cursor(self, *args, **kwargs):
        return self._raw.cursor(*args, **kwargs)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.mark.integration
def test_an_agent_output_is_stored_and_a_processed_copy_is_reused(
    db_session, seed_workspace, monkeypatch, tmp_path
):
    from sqlalchemy import text

    from modules.rag.ingestion import manager as mgr

    ws, other_ws = seed_workspace(), seed_workspace()
    shared = _SharedConnection(db_session.connection().connection.dbapi_connection)
    monkeypatch.setattr(mgr.psycopg2, "connect", lambda **_kw: shared)
    path = tmp_path / "report.md"
    path.write_bytes(b"# Weekly report\n\nRevenue rose 4%.\n")
    processed = []
    manager = _manager(mgr, ws, processed)

    first = _upload(manager, path, source_type="agent_output")
    row = db_session.execute(text(
        "SELECT content_hash, doc_metadata, status, source_type, CAST(workspace_id AS text), "
        "file_path, tags FROM documents WHERE id = :id"), {"id": first}).one()
    assert row[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (row[1]["filename"], row[1]["created_by"]) == ("report.md", "agent:7")
    assert tuple(row)[2:] == ("processing", "agent_output", ws, f"s3://docs/ws/{first}_report.md", ["agent_output"])
    assert processed == [(first, mgr.DocumentType.MARKDOWN, f"ws/{first}_report.md", "report.md")]

    # Not processed yet: a second upload gets its own row; the first is kept.
    second = _upload(manager, path, source_type="agent_output")
    assert second != first
    db_session.execute(text("UPDATE documents SET status = 'completed' WHERE id = :id"), {"id": second})
    assert _upload(manager, path, source_type="agent_output") == second    # the processed copy
    assert len(processed) == 2

    owners = _upload(manager, path)                                        # the owner's own scope
    elsewhere = _upload(_manager(mgr, other_ws, processed), path, source_type="agent_output")
    assert len({first, second, owners, elsewhere}) == 4
    kept = db_session.execute(text("SELECT count(*) FROM documents WHERE id = ANY(:ids)"),
                              {"ids": [first, second]}).scalar()
    assert kept == 2
