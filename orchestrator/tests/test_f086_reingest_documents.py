"""F086 — the existing documents are re-ingested in place, from their sources.

480 of 575 documents were stored short by the old chunker. The re-ingest script
measures each against its source (the upload on disk or the copy in object
storage), re-processes the short ones under the same id — without paying for a
second knowledge-graph extraction — and moves uploads onto the persistent
volume so a rebuild cannot take the sources away again (F102).
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reingest_documents.py"
spec = importlib.util.spec_from_file_location("reingest_documents", _SCRIPT)
reingest_documents = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reingest_documents          # dataclasses resolve their module by name
spec.loader.exec_module(reingest_documents)


def test_the_plan_is_the_short_documents_and_the_ones_with_no_source(tmp_path):
    on_disk = tmp_path / "brand.md"
    on_disk.write_text("x")
    rows = [
        (720, "ws", "brand.md", "markdown", str(on_disk), ["a"]),
        (721, "ws", "full.md", "markdown", str(on_disk), ["b"]),
        (900, "ws", "report.md", "md", "s3://bucket/workspaces/ws/documents/900_report.md", ["c"]),
        (1, "ws", "gone.pdf", "pdf", "/tmp/nowhere/gone.pdf", []),
    ]
    kept = {"a": 37, "b": 100, "c": 81}
    found = reingest_documents.plan(rows, below=98, measure=lambda ws, path, chunks: kept[chunks[0]])
    assert [(c.id, c.kept, c.source) for c in found] == [
        (720, 37, "local file"), (900, 81, "object storage"), (1, None, "none")]


def test_an_upload_moves_onto_the_volume_once(tmp_path):
    volume = tmp_path / "data" / "uploads"
    assert reingest_documents.moved_path("/tmp/automotas_uploads/ab12.md", volume) == volume / "ab12.md"
    assert reingest_documents.moved_path(str(volume / "ab12.md"), volume) is None


def test_reingest_clears_then_processes_the_same_id_without_a_second_graph_pass():
    calls = []

    class Manager:
        def clear_chunks(self, document_id):
            calls.append(("clear", document_id))

        async def _process_document(self, document_id, path, file_type, s3_key=None, filename=None,
                                    update_graph=True):
            calls.append(("process", document_id, path, file_type.value, s3_key, filename, update_graph))

    candidate = reingest_documents.Candidate(
        900, "ws", "report.md", "markdown", "s3://bucket/workspaces/ws/documents/900_report.md", 81, "object storage")
    asyncio.run(reingest_documents.reingest(candidate, Manager(), "/tmp/copy.md"))
    assert calls == [("clear", 900),
                     ("process", 900, "/tmp/copy.md", "md", "workspaces/ws/documents/900_report.md", "report.md", False)]


def test_clear_chunks_keeps_the_document_and_drops_what_ingestion_stored(monkeypatch):
    from modules.rag.ingestion import manager as mgr

    log = []

    class Cursor:
        rowcount = 4

        def execute(self, sql, params=None):
            log.append((" ".join(sql.split()), params))

        def fetchone(self):
            return (True, True)

        def close(self):
            pass

    class Conn:
        def cursor(self):
            return Cursor()

        def commit(self):
            log.append(("COMMIT", None))

        def close(self):
            pass

    manager = mgr.DocumentManager.__new__(mgr.DocumentManager)
    manager.db_config, manager.use_s3_vectors, manager._s3_backend = {}, False, None
    monkeypatch.setattr(manager, "_ensure_database_initialized", lambda: None)
    monkeypatch.setattr(mgr.psycopg2, "connect", lambda **_kw: Conn())
    assert manager.clear_chunks(720) == 4
    statements = [sql for sql, _ in log]
    assert "DELETE FROM document_chunks WHERE document_id = %s" in statements
    assert "DELETE FROM kb_tables WHERE knowledge_item_id = %s" in statements
    assert "DELETE FROM kb_formulas WHERE knowledge_item_id = %s" in statements
    assert not any(sql.startswith("DELETE FROM documents") for sql in statements)
    assert statements[-1] == "COMMIT"


def test_the_stack_keeps_uploads_on_the_volume():
    defaults = (Path(__file__).resolve().parents[2] / "envs" / "api.defaults").read_text()
    assert "\nDOCUMENT_UPLOAD_DIR=/app/data/uploads\n" in defaults
