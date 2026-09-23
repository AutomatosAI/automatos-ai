"""F088 A+B (night 3) — on a near tie, the owner's document and the newer copy rank first.

Night 3: 408 agent reports shared the index with 166 of the owner's documents,
and six copies of the Christmas sheet sat side by side. Near-equal scores let
an agent's report or an old copy rank first, and Auto answered from it. Small
multiplicative priors now lift the owner's documents and newer documents, so
only near ties change order — a clearly better hit keeps its place.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.rag.service import RAGService

OLD, NEW = datetime(2026, 9, 1), datetime(2026, 9, 22)          # upload_date is naive UTC


def _service(monkeypatch, facts, upload_prior=0.05, recency_prior=0.02):
    svc = RAGService.__new__(RAGService)
    svc.config = NS(upload_prior=upload_prior, recency_prior=recency_prior)
    monkeypatch.setattr(RAGService, "_document_ranking_facts", classmethod(lambda cls, ids: facts))
    return svc


def _hit(doc_id, score, key="similarity"):
    return {"document_id": doc_id, key: score, "content": f"chunk of {doc_id}"}


def test_the_owners_upload_wins_a_near_tie_with_an_agents_report(monkeypatch):
    svc = _service(monkeypatch, {"900": (True, NEW), "717": (False, NEW)})
    ranked = svc._apply_source_priors([_hit(900, 0.61), _hit(717, 0.60)])
    assert [h["document_id"] for h in ranked] == [717, 900]


def test_a_clearly_better_report_keeps_its_place(monkeypatch):
    svc = _service(monkeypatch, {"900": (True, NEW), "717": (False, NEW)})
    ranked = svc._apply_source_priors([_hit(900, 0.80), _hit(717, 0.60)])
    assert [h["document_id"] for h in ranked] == [900, 717]


def test_the_newer_copy_wins_a_near_tie_with_the_older(monkeypatch):
    svc = _service(monkeypatch, {"716": (False, OLD), "731": (False, NEW)})
    ranked = svc._apply_source_priors([_hit(716, 0.605), _hit(731, 0.60)])
    assert [h["document_id"] for h in ranked] == [731, 716]


def test_the_reranked_order_is_the_one_adjusted(monkeypatch):
    svc = _service(monkeypatch, {"900": (True, NEW), "717": (False, NEW)})
    hits = [{**_hit(900, 0.30), "rerank_score": 0.91}, {**_hit(717, 0.70), "rerank_score": 0.89}]
    ranked = svc._apply_source_priors(hits)
    assert [h["document_id"] for h in ranked] == [717, 900]
    assert ranked[0]["rerank_score"] == pytest.approx(0.89 * 1.07)


def test_the_candidates_are_not_changed_in_place_and_zero_priors_change_nothing(monkeypatch):
    hits = [_hit(900, 0.61), _hit(717, 0.60)]
    before = [dict(h) for h in hits]
    _service(monkeypatch, {"900": (True, NEW), "717": (False, NEW)})._apply_source_priors(hits)
    assert hits == before
    off = _service(monkeypatch, {"900": (True, NEW), "717": (False, NEW)}, upload_prior=0, recency_prior=0)
    assert off._apply_source_priors(hits) is hits


def test_a_hit_with_no_document_row_keeps_its_own_score(monkeypatch):
    svc = _service(monkeypatch, {"717": (False, NEW)})
    ranked = svc._apply_source_priors([_hit(999, 0.70), _hit(717, 0.60)])
    assert [(h["document_id"], round(h["similarity"], 3)) for h in ranked] == [(999, 0.70), (717, 0.642)]


# ── the one read, on Postgres ────────────────────────────────────────────────

_DROP = "DROP TABLE IF EXISTS pg_temp.documents"


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text(_DROP))
        session.execute(text("CREATE TEMP TABLE documents (LIKE public.documents INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        session.execute(text(_DROP))
        session.commit()
        session.close()


def test_the_facts_come_from_the_documents_rows(db, monkeypatch):
    import core.database.database as database

    ws = str(uuid4())
    for doc_id, source_type, created in ((717, None, OLD), (900, "agent_output", NEW)):
        db.execute(text("INSERT INTO documents (id, workspace_id, filename, source_type, status, upload_date) "
                        "VALUES (:id, CAST(:ws AS uuid), 'f.md', :st, 'completed', :at)"),
                   {"id": doc_id, "ws": ws, "st": source_type, "at": created})
    monkeypatch.setattr(database, "SessionLocal", lambda: _Borrowed(db))
    facts = RAGService._document_ranking_facts(["717", "900", None, "s3-key"])
    assert facts["717"][0] is False and facts["900"][0] is True
    assert facts["900"][1] > facts["717"][1]


class _Borrowed:
    """The fixture's session, which the read must not close."""

    def __init__(self, session):
        self._session = session

    def execute(self, *a, **k):
        return self._session.execute(*a, **k)

    def close(self):
        pass
