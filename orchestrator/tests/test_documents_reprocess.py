"""#835 — reprocessing one document rebuilds THAT document from its source.

``POST /api/documents/{id}/reprocess`` returned 500: it called
``DocumentManager.upload_document(file_type=…)`` (no such parameter), then read
the int it returns as a dict — and would have created a second document. It now
shares ``_reprocess_from_source`` with ``reprocess-all``: fetch the stored source
(local or ``s3://``), drop the document's chunks, run the pipeline on the same
row. Fakes stand in for Postgres, S3 and the manager: what is pinned is which
document is rebuilt, from where, and what is left behind.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import psycopg2
import pytest

from api import documents as api

SETTINGS = api.config  # patched below; a module-level name keeps secret scanners off the lines


class _Cursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))


class _Conn:
    def __init__(self, log):
        self.log = log

    def cursor(self):
        return _Cursor(self.log)

    def commit(self):
        self.log.append(("COMMIT", None))

    def close(self):
        pass


class _Manager:
    def __init__(self):
        self.processed = []
        self.processor = SimpleNamespace(detect_file_type=lambda path: "markdown")

    async def _process_document(self, doc_id, path, file_type):
        with open(path, encoding="utf-8") as fh:  # the source is readable while it is processed
            self.processed.append((doc_id, os.path.basename(path), file_type, fh.read()))

    async def upload_document(self, *a, **k):
        raise AssertionError("reprocess must rebuild the existing document, not upload a new one")


@pytest.fixture
def sql(monkeypatch):
    log: list = []
    monkeypatch.setattr(psycopg2, "connect", lambda **kw: _Conn(log))
    return log


@pytest.mark.asyncio
async def test_a_local_source_rebuilds_the_same_document(sql, tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# hello", encoding="utf-8")
    manager = _Manager()
    assert await api._reprocess_from_source(manager, 7, str(source), "notes.md") is True
    assert sql[0] == ("DELETE FROM document_chunks WHERE document_id = %s", (7,))
    assert manager.processed == [(7, "notes.md", "markdown", "# hello")]


@pytest.mark.asyncio
async def test_a_missing_source_touches_nothing(sql, tmp_path):
    manager = _Manager()
    assert await api._reprocess_from_source(manager, 7, str(tmp_path / "gone.md"), "gone.md") is False
    assert await api._reprocess_from_source(manager, 7, None, None) is False
    assert sql == [] and manager.processed == []  # its chunks survive for the re-embed fallback


@pytest.mark.asyncio
async def test_an_s3_source_is_fetched_processed_and_cleaned_up(sql, monkeypatch):
    fetched = []

    class _S3:
        def download_file(self, bucket, key, dest):
            fetched.append((bucket, key, dest))
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write("from s3")

    import core.storage
    monkeypatch.setattr(core.storage, "get_s3_client", lambda: _S3(), raising=False)
    manager = _Manager()
    assert await api._reprocess_from_source(manager, 9, "s3://docs/ws/9/report.md", "report.md") is True
    (bucket, key, dest), = fetched
    assert (bucket, key) == ("docs", "ws/9/report.md") and dest.endswith(".md")
    assert manager.processed[0][0] == 9 and manager.processed[0][3] == "from s3"
    assert not os.path.exists(dest)  # the temp copy is removed


@pytest.mark.asyncio
async def test_an_s3_source_that_cannot_be_fetched_is_no_source(sql, monkeypatch):
    class _S3:
        def download_file(self, bucket, key, dest):
            raise RuntimeError("NoSuchKey")

    import core.storage
    monkeypatch.setattr(core.storage, "get_s3_client", lambda: _S3(), raising=False)
    manager = _Manager()
    assert await api._reprocess_from_source(manager, 9, "s3://docs/ws/9/report.md", "report.md") is False
    assert sql == [] and manager.processed == []


class _Db:
    def __init__(self, document):
        self.document = document
        self.commits = 0

    def query(self, *a):
        return self

    def filter(self, *a):
        return self

    def first(self):
        return self.document

    def commit(self):
        self.commits += 1

    def refresh(self, obj):  # what the pipeline wrote to the row
        obj.status, obj.chunk_count = "completed", 12


def _document(**over):
    return SimpleNamespace(**{"id": 5, "file_path": "/data/notes.md", "filename": "notes.md",
                              "status": "completed", "chunk_count": 3, **over})


@pytest.mark.asyncio
async def test_the_endpoint_rebuilds_in_place_and_reports_the_row(monkeypatch):
    calls = []

    async def rebuilt(manager, doc_id, file_path, filename):
        calls.append((doc_id, file_path))
        return True

    monkeypatch.setattr(api, "_reprocess_from_source", rebuilt)
    monkeypatch.setattr(api, "get_document_manager", lambda ws: _Manager())
    db = _Db(_document())
    out = await api.reprocess_document(5, ctx=SimpleNamespace(workspace_id="ws-1"), db=db)
    assert calls == [(5, "/data/notes.md")]
    assert out == {"message": "Document reprocessed successfully", "document_id": 5,
                   "chunk_count": 12, "status": "completed"}


@pytest.mark.asyncio
async def test_without_a_source_the_endpoint_falls_back_to_the_stored_chunks(monkeypatch):
    async def no_source(*a):
        return False

    async def reembedded(db, document, workspace_id):
        return 4

    monkeypatch.setattr(api, "_reprocess_from_source", no_source)
    monkeypatch.setattr(api, "_reembed_document_from_chunks", reembedded)
    monkeypatch.setattr(api, "get_document_manager", lambda ws: _Manager())
    monkeypatch.setattr(SETTINGS, "S3_VECTORS_ENABLED", False, raising=False)
    document = _document()
    out = await api.reprocess_document(5, ctx=SimpleNamespace(workspace_id="ws-1"), db=_Db(document))
    assert out["source"] == "chunks" and out["chunk_count"] == 4 and document.status == "processed"


@pytest.mark.asyncio
async def test_with_nothing_to_rebuild_from_the_status_is_restored_and_it_says_so(monkeypatch):
    async def no_source(*a):
        return False

    monkeypatch.setattr(api, "_reprocess_from_source", no_source)
    monkeypatch.setattr(api, "get_document_manager", lambda ws: _Manager())
    monkeypatch.setattr(SETTINGS, "S3_VECTORS_ENABLED", True, raising=False)
    document = _document(status="completed")
    with pytest.raises(api.HTTPException) as refused:
        await api.reprocess_document(5, ctx=SimpleNamespace(workspace_id="ws-1"), db=_Db(document))
    assert refused.value.status_code == 400 and document.status == "completed"
