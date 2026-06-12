"""PRD-158 S5 — widget docs rebuilt on the real schema.

The old surface queried documents.title/content (non-existent) so search and
get were impossible. These integration tests prove the rebuilt endpoints work on
the real schema: get reassembles content from document_chunks; search groups the
PRD-157 retrieval chunks by document and resolves real filenames.
"""

from __future__ import annotations

import os
import sys
import types
import uuid

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest  # noqa: E402


def _seed_doc(db, ws, filename="brief.pdf", chunks=("alpha content", "beta content")):
    from sqlalchemy import text

    doc_id = db.execute(
        text(
            "INSERT INTO documents (filename, original_filename, workspace_id, team_access, status, upload_date) "
            "VALUES (:fn, :fn, CAST(:ws AS uuid), '{}', 'processed', NOW()) RETURNING id"
        ),
        {"fn": filename, "ws": ws},
    ).scalar()
    for i, c in enumerate(chunks):
        db.execute(
            text(
                "INSERT INTO document_chunks (document_id, chunk_index, content, workspace_id) "
                "VALUES (:doc, :idx, :content, :ws)"
            ),
            {"doc": doc_id, "idx": i, "content": c, "ws": ws},
        )
    db.flush()
    return doc_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_document_reassembles_content_from_chunks(db_session, seed_workspace):
    from api.widgets.documents import get_document

    ws = seed_workspace()  # FK parent for documents/document_chunks.workspace_id
    doc_id = _seed_doc(db_session, ws, filename="brief.pdf", chunks=("alpha content", "beta content"))
    auth = types.SimpleNamespace(workspace_id=uuid.UUID(ws), team=None)

    detail = await get_document(doc_id, auth=auth, _perm=auth, db=db_session)
    assert detail.id == doc_id
    assert detail.title == "brief.pdf"            # filename, not a non-existent title col
    assert "alpha content" in detail.content and "beta content" in detail.content
    assert detail.created_at is not None          # upload_date, not created_at


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_groups_chunks_and_resolves_filenames(db_session, seed_workspace, monkeypatch):
    from api.widgets.documents import search_documents, DocumentSearchRequest
    from modules.rag.service import RAGService

    ws = seed_workspace()  # FK parent for documents/document_chunks.workspace_id
    doc_id = _seed_doc(db_session, ws, filename="report.pdf")

    # Stand in for the real vector retrieval: two chunks of the same doc.
    class _Result:
        chunks = [
            {"document_id": doc_id, "content": "the answer about widgets", "similarity": 0.91, "source_file": "temp123.txt", "metadata": {}},
            {"document_id": doc_id, "content": "secondary chunk", "similarity": 0.40, "source_file": "temp123.txt", "metadata": {}},
        ]

    async def _fake_retrieve(self, **kwargs):
        return _Result()

    # retrieve is imported lazily inside the endpoint; patch the class directly.
    monkeypatch.setattr(RAGService, "retrieve", _fake_retrieve)

    auth = types.SimpleNamespace(workspace_id=uuid.UUID(ws), team=None)
    resp = await search_documents(DocumentSearchRequest(query="widgets", limit=10), auth=auth, _perm=auth, db=db_session)

    assert resp.total == 1                         # two chunks → one document
    assert resp.results[0].id == doc_id
    assert resp.results[0].title == "report.pdf"   # real filename, not the temp source
    assert resp.results[0].score == pytest.approx(0.91)
