"""F311 (night 9) — when agents' reports fill a search, the owner's documents are searched too.

"Who do I talk to about reordering Kirinyaga, and how long will it take to arrive?"
(chat bb91ddec): the 15 hits were agents' reports and christmas-boxes-2026.md; the
importers document, which names Maya Odum at Tidewater and a 3-week lead time, was
not among them, and Auto answered "Ellie" (ledger L99). Now a search whose hits hold
an agent's report also searches the owner's own documents, and their hits join in.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

import modules.rag.owner_leg as owner_leg
import modules.rag.service as rag_service
from config import config
from modules.rag.service import RAGService
from modules.search.vector_store.backends import pgvector_local_backend as pgvector
from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend

WS = str(uuid4())
QUESTION = "Who do I talk to about reordering Kirinyaga, and how long will it take to arrive?"
TIDEWATER = ("# Green coffee — who we buy from\n## Tidewater Importers (London)\n"
             "- Supplies: **Guji Shakiso**, **Kirinyaga AA**, **Yirgacheffe Konga**\n"
             "- Contact: Maya Odum, maya@tidewater-importers.example, 020 7946 0381\n"
             "- Lead time: **3 weeks** from order to our door")
REPORTS = range(1529, 1544)          # fifteen agents' reports
OWNERS = (1520, 1517)                # importers-and-green-buying.md, christmas-boxes-2026.md
NOW = datetime(2026, 10, 4)


def _hit(doc_id, chunk, score, content):
    return {"key": f"doc_{doc_id}_chunk_{chunk}", "score": score, "content": content,
            "file_name": f"doc-{doc_id}.md", "file_path": f"/app/data/uploads/{doc_id}.md",
            "external_file_id": str(doc_id),
            "metadata": {"external_file_id": str(doc_id), "chunk_index": chunk, "workspace_id": WS}}


class _Backend:
    """The workspace's store: the reports win the whole search; the owner's are below them."""

    def __init__(self, hits):
        self.hits = hits
        self.asked_within = []

    def search(self, query_embedding, limit=10, min_score=0.5, filters=None):
        return sorted(self.hits, key=lambda h: h["score"], reverse=True)[:limit]

    def search_in_documents(self, query_embedding, document_ids, limit=10, min_score=0.5):
        self.asked_within.append(sorted(document_ids))
        mine = [h for h in self.hits if int(h["external_file_id"]) in document_ids and h["score"] >= min_score]
        return sorted(mine, key=lambda h: h["score"], reverse=True)[:limit]


class _Embeddings:
    async def generate_embedding(self, text):
        return [0.1, 0.2, 0.3]


def _store():
    reports = [_hit(doc_id, 0, 0.72 - i * 0.005, f"Task report {doc_id}: Ellie checks the Kirinyaga stock.")
               for i, doc_id in enumerate(REPORTS)]
    return reports + [_hit(1520, 1, 0.61, TIDEWATER), _hit(1517, 0, 0.58, "Christmas boxes: Kirinyaga AA, 250 g.")]


def _rag(backend):
    rag = RAGService.__new__(RAGService)
    rag._embedding_manager = _Embeddings()
    rag._doc_backends = {WS: backend}
    rag._workspace_id = WS
    return rag


@pytest.fixture(autouse=True)
def the_workspace(monkeypatch):
    monkeypatch.setattr(rag_service, "record_substrate_search_nowait", lambda **kwargs: None)
    monkeypatch.setattr(config, "RAG_OWNER_LEG_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "RAG_OWNER_LEG_MAX_DOCUMENTS", 200, raising=False)
    facts = {**{str(d): (True, NOW) for d in REPORTS}, **{str(d): (False, NOW) for d in OWNERS}}
    monkeypatch.setattr(RAGService, "_document_ranking_facts",
                        classmethod(lambda cls, ids: {i: facts[i] for i in ids if i in facts}))
    asked = []
    monkeypatch.setattr(owner_leg, "owner_document_ids", lambda ws, cap: asked.append((ws, cap)) or list(OWNERS))
    return asked


def test_the_importers_section_is_found_when_fifteen_reports_fill_the_search(the_workspace):
    backend = _Backend(_store())
    found = asyncio.run(_rag(backend)._get_candidates(QUESTION, limit=15, min_similarity=0.5, workspace_id=WS))

    assert any("Maya Odum" in c["content"] and "**3 weeks**" in c["content"] for c in found)
    tidewater = next(c for c in found if "Maya Odum" in c["content"])
    assert tidewater["document_id"] == "1520" and tidewater["metadata"]["chunk_index"] == 1
    assert backend.asked_within == [[1517, 1520]] and the_workspace == [(WS, 200)]
    assert len(found) == 17 and len({c["id"] for c in found}) == 17


def test_a_search_whose_hits_are_all_the_owners_costs_nothing_more(the_workspace):
    backend = _Backend([_hit(1520, 1, 0.61, TIDEWATER), _hit(1517, 0, 0.58, "Christmas boxes.")])
    found = asyncio.run(_rag(backend)._get_candidates(QUESTION, limit=15, workspace_id=WS))

    assert [c["document_id"] for c in found] == ["1520", "1517"]
    assert backend.asked_within == [] and the_workspace == []


def test_the_owners_leg_off_leaves_the_search_as_it_was(the_workspace, monkeypatch):
    monkeypatch.setattr(config, "RAG_OWNER_LEG_ENABLED", False, raising=False)
    backend = _Backend(_store())
    found = asyncio.run(_rag(backend)._get_candidates(QUESTION, limit=15, workspace_id=WS))

    assert not any("Maya Odum" in c["content"] for c in found) and backend.asked_within == []


# ── the backends search within the documents named, in the workspace only ────────────


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self.rows, self.calls, self.closed = rows, [], False

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params))
        return _Rows(self.rows)

    def close(self):
        self.closed = True


def test_the_local_store_binds_the_documents_and_the_workspace(monkeypatch):
    import core.database.database as database

    row = NS(document_id=1520, chunk_index=1, content=TIDEWATER, file_name="importers-and-green-buying.md",
             file_path="/app/data/uploads/b9.md", similarity=0.61)
    session = _Session([row])
    monkeypatch.setattr(database, "SessionLocal", lambda: session)
    monkeypatch.setattr(pgvector, "_COLUMN_TYPES", ("USER-DEFINED", "uuid"))

    hits = pgvector.PgVectorLocalBackend(WS).search_in_documents([0.1, 0.2], [1520, "1517"], limit=15, min_score=0.5)

    sql, params = session.calls[-1]
    assert "dc.document_id = ANY(:ids)" in sql and "d.workspace_id = CAST(:ws AS uuid)" in sql
    assert params["ids"] == [1517, 1520] and params["ws"] == WS and params["limit"] == 15
    assert hits[0]["content"] == TIDEWATER and hits[0]["external_file_id"] == "1520" and session.closed


class _S3Client:
    def __init__(self, vectors):
        self.vectors, self.queries = vectors, []

    def query_vectors(self, **kwargs):
        self.queries.append(kwargs)
        return {"vectors": self.vectors}


def test_s3_vectors_filters_on_the_workspace_and_the_documents_and_drops_any_other_workspace():
    mine = {"key": "doc_1520_chunk_1", "distance": 0.39,
            "metadata": {"workspace_id": WS, "external_file_id": "1520", "chunk_text": TIDEWATER, "chunk_index": 1}}
    foreign = {"key": "doc_1520_chunk_2", "distance": 0.10,
               "metadata": {"workspace_id": str(uuid4()), "external_file_id": "1520", "chunk_text": "not ours"}}
    backend = S3VectorsBackend.__new__(S3VectorsBackend)
    backend.workspace_id, backend.bucket_name, backend.index_name = WS, "bucket", "index"
    backend.client = _S3Client([mine, foreign])
    backend._ensure_setup = lambda: None

    hits = backend.search_in_documents([0.1, 0.2], [1520, 1517], limit=15, min_score=0.5)

    query = backend.client.queries[-1]
    assert query["filter"] == {"$and": [{"workspace_id": {"$eq": WS}},
                                        {"external_file_id": {"$in": ["1517", "1520"]}}]}
    assert query["topK"] == 15
    assert [h["content"] for h in hits] == [TIDEWATER] and hits[0]["score"] == pytest.approx(0.61)
