"""F172 — a saved NL2SQL example carries its question's embedding.

example_store called ``em.embed_text(question)``, a method EmbeddingManager does
not have (it has ``generate_embedding`` / ``generate_embedding_sync``). Every
example was saved without an embedding ("'EmbeddingManager' object has no
attribute 'embed_text'" on each query), and every lookup fell back to keyword
overlap. The tests passed because they faked the manager with a bare MagicMock,
which accepts any method name. The fakes here are specced from the real class.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

QUESTION = "How many subscribers do I have right now, not counting cancelled ones?"


def _store(**kw):
    from modules.nl2sql.training.example_store import SQLExampleStore

    return SQLExampleStore(**kw)


def _real_shaped_embedder(vector):
    from core.llm.embedding_manager import EmbeddingManager

    em = create_autospec(EmbeddingManager, instance=True)
    em.generate_embedding = AsyncMock(return_value=vector)
    return em


def test_a_saved_example_carries_its_embedding():
    saved = {}

    class Row:
        def __init__(self, **kwargs):
            saved.update(kwargs)
            self.id = 105

    store = _store(embedding_manager=_real_shaped_embedder([0.6, 0.8]), db_session=MagicMock())
    with patch("core.models.database_knowledge.NL2SQLTrainingExample", Row):
        asyncio.run(store.add_example(question=QUESTION, sql="SELECT count(*) FROM subscribers",
                                      database_source_id="36", workspace_id="ws-1", verification_source="auto"))
    assert saved["embedding"] == [0.6, 0.8] and saved["embedding_id"]


def test_a_lookup_ranks_by_the_questions_embedding():
    em = _real_shaped_embedder([1.0, 0.0])
    db = MagicMock()
    rows = [SimpleNamespace(id=1, question="count the live members", sql="SELECT 1", tables_used=[],
                            embedding=[0.99, 0.05], is_verified=True, usage_count=0, last_used_at=None),
            SimpleNamespace(id=2, question=QUESTION.lower(), sql="SELECT 2", tables_used=[],
                            embedding=[0.0, 1.0], is_verified=True, usage_count=0, last_used_at=None)]
    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = rows
    db.query.return_value = query
    store = _store(embedding_manager=em, db_session=db)
    with patch("core.models.database_knowledge.NL2SQLTrainingExample", MagicMock()):
        out = asyncio.run(store.get_similar_examples(question=QUESTION, database_source_id="36", workspace_id="ws-1"))
    em.generate_embedding.assert_awaited_with(QUESTION)
    assert out and out[0]["id"] == 1  # by vector, not by the keyword twin
