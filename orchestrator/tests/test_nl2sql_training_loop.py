"""PRD-160 S3 — training loop: persisted embeddings + verified-pair few-shot.

Unit-level: the embedding manager and DB session are faked. Pins that (a) the
question embedding is now PERSISTED on add_example (it used to be computed then
discarded), and (b) get-similar ranks verified pairs by cosine similarity of
the persisted vectors, preferring the semantically-closest pair.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _store(**kw):
    from modules.nl2sql.training.example_store import SQLExampleStore

    return SQLExampleStore(**kw)


# --- embedding is persisted, not discarded -----------------------------------

def test_add_example_persists_embedding_vector():
    captured = {}

    class FakeExample:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.id = 1

    em = MagicMock()
    em.embed_text.return_value = [0.1, 0.2, 0.3]
    db = MagicMock()
    store = _store(embedding_manager=em, db_session=db)

    with patch(
        "core.models.database_knowledge.NL2SQLTrainingExample", FakeExample
    ):
        asyncio.run(store.add_example(
            question="how many active users?",
            sql="SELECT count(*) FROM users WHERE status='active'",
            database_source_id="7",
            workspace_id="ws-1",
            is_verified=True,
        ))

    # the embedding was stored on the row (previously thrown away)
    assert captured["embedding"] == [0.1, 0.2, 0.3]
    assert captured["is_verified"] is True
    db.add.assert_called_once()
    db.commit.assert_called()


def test_add_example_survives_embedding_failure():
    class FakeExample:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.id = 2

    em = MagicMock()
    em.embed_text.side_effect = RuntimeError("embedder down")
    db = MagicMock()
    store = _store(embedding_manager=em, db_session=db)

    with patch("core.models.database_knowledge.NL2SQLTrainingExample", FakeExample):
        out = asyncio.run(store.add_example(
            question="q", sql="SELECT 1", database_source_id="7", workspace_id="ws-1",
        ))
    assert out == "2"  # still stored, just without an embedding


# --- cosine ranking of verified pairs (golden) -------------------------------

def _example(id_, question, embedding):
    return SimpleNamespace(
        id=id_, question=question, sql=f"SELECT {id_}", tables_used=[],
        embedding=embedding, is_verified=True, usage_count=0, last_used_at=None,
    )


def test_embedding_similarity_ranks_closest_pair_first():
    store = _store()
    db = MagicMock()
    # query vector points along x; ex1 aligns, ex2 is orthogonal
    q = [1.0, 0.0, 0.0]
    ex_close = _example(1, "active users count", [0.9, 0.1, 0.0])
    ex_far = _example(2, "revenue by region", [0.0, 1.0, 0.0])

    ranked = store._embedding_similarity(q, [ex_far, ex_close], limit=5,
                                         min_similarity=0.3, db=db)

    assert [r["id"] for r in ranked] == [1]  # only the aligned pair clears 0.3
    assert ranked[0]["similarity"] >= 0.3
    # usage telemetry bumped + persisted
    assert ex_close.usage_count == 1
    db.commit.assert_called()


def test_get_similar_examples_prefers_embeddings_over_keyword():
    store = _store()
    db = MagicMock()
    em = MagicMock()
    em.embed_text.return_value = [1.0, 0.0]
    store.embedding_manager = em

    rows = [
        _example(1, "totally different words", [1.0, 0.02]),  # close by vector
        _example(2, "similar question text here", [0.0, 1.0]),  # close by keyword only
    ]
    query_obj = MagicMock()
    query_obj.filter.return_value = query_obj
    query_obj.all.return_value = rows
    db.query.return_value = query_obj

    with patch("core.models.database_knowledge.NL2SQLTrainingExample", MagicMock()):
        out = asyncio.run(store.get_similar_examples(
            question="similar question text here",
            database_source_id="7",
            workspace_id="ws-1",
        ))

    # vector match (id 1) wins despite id 2 having the keyword overlap
    assert out and out[0]["id"] == 1
