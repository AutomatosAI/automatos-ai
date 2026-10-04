"""F311 (night 9) — a passage from a short document of the owner's is handed over whole.

"A café wants 10 kg of coffee next week. What do we charge them for delivery?" (chat
9928b259, 12:33): the search handed over wholesale-terms-2026.md's Prices chunk and
kept one passage per document, so the chunk with the carriage charges (document 1526,
chunk 5, as stored that night) never came, and Auto said the terms name no delivery
charge (ledger L1). The document is 1,201 bytes. Now a passage from an owner's
document that short is replaced by the whole document, read within the workspace.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from modules.rag.service import RAGResult, RAGService
from modules.rag.whole_documents import document_texts, short_documents_whole, whole_where_short

NOW = datetime(2026, 10, 4)
QUESTION = "A café wants 10 kg of coffee next week. What do we charge them for delivery, and is delivery ever free?"
# wholesale-terms-2026.md as the night stored it: 8 chunks, the carriage lines in chunk 5.
STORED_1526 = [
    "# Harbourline wholesale terms (cafés) — 2026\nUpdated after the January review. "
    "Send this to any new café before their first order.",
    "## Prices\n- Standard wholesale price: **£22.00 per kg** (Harbour Blend and the core range). "
    "- Cafés on our single origins pay **£24.00 per kg**.",
    "- Prices are per kg of roasted coffee. Coffee is zero-rated, so there's no VAT on it. "
    "## Ordering\n- Minimum order: **6 kg** per delivery.",
    "- Order by **noon the day before** your delivery day (email or the order form). - We roast Tuesdays "
    "and Thursdays, so anything ordered after Thursday noon goes out on the following week's run.",
    "## Delivery\nOur van does fixed days: - Bristol: Thursday and Friday\n- Bath: Monday and Wednesday\n"
    "- Taunton and Cheltenham: Monday\n- Exeter: Tuesday\n- Plymouth: Tuesday and Wednesday\n"
    "- Bournemouth: Wednesday (every other week)",
    "Carriage is charged per drop: - under 12 kg: **£8.50**\n- 12 kg and over: **£5.00** ## Payment\n"
    "- **30 days** from the invoice date.",
    "A handful of accounts are on 14 days by agreement. - Bank transfer only, please quote the invoice number.",
    "## Returns\nIf a bag arrives damaged, photograph it and tell us within 48 hours. "
    "We'll replace it on the next drop.",
]
REPORT = "# Shopify Operations Manager — Task Report\n**Task:** Minimum wholesale order and order cut-off"

_TABLES = ("DROP TABLE IF EXISTS pg_temp.document_chunks", "DROP TABLE IF EXISTS pg_temp.documents")


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        for drop in _TABLES:
            session.execute(text(drop))
        session.execute(text("CREATE TEMP TABLE documents (id int PRIMARY KEY, workspace_id uuid, filename text, "
                             "source_type text, status text, upload_date timestamp)"))
        session.execute(text("CREATE TEMP TABLE document_chunks (id serial PRIMARY KEY, document_id int, "
                             "chunk_index int, content text, chunk_type varchar(20) DEFAULT 'child')"))
        yield session
        session.rollback()
        for drop in _TABLES:
            session.execute(text(drop))
        session.commit()
        session.close()


class _Borrowed:
    """The fixture's session, which the read must not close."""

    def __init__(self, session):
        self._session = session

    def execute(self, *a, **k):
        return self._session.execute(*a, **k)

    def close(self):
        pass


def _seed(db, ws, other_ws):
    rows = ((1526, ws, "wholesale-terms-2026.md", None), (1530, ws, "task-minimum-order.md", "agent_output"),
            (1600, other_ws, "wholesale-terms-2026.md", None))
    for doc_id, workspace, name, source_type in rows:
        db.execute(text("INSERT INTO documents VALUES (:id, CAST(:ws AS uuid), :name, :st, 'completed', :at)"),
                   {"id": doc_id, "ws": workspace, "name": name, "st": source_type, "at": NOW})
    chunks = [(1526, i, c) for i, c in enumerate(STORED_1526)] + [(1530, 0, REPORT), (1600, 0, "another tenant")]
    for doc_id, index, content in reversed(chunks):
        db.execute(text("INSERT INTO document_chunks (document_id, chunk_index, content) VALUES (:d, :i, :c)"),
                   {"d": doc_id, "i": index, "c": content})


@pytest.fixture
def workspace(db, monkeypatch):
    import core.database.database as database

    ws, other_ws = str(uuid4()), str(uuid4())
    _seed(db, ws, other_ws)
    monkeypatch.setattr(database, "SessionLocal", lambda: _Borrowed(db))
    monkeypatch.setattr(config, "RAG_WHOLE_DOCUMENT_MAX_TOKENS", 800, raising=False)
    facts = {"1526": (False, NOW), "1530": (True, NOW)}
    monkeypatch.setattr(RAGService, "_document_ranking_facts",
                        classmethod(lambda cls, ids: {i: facts[i] for i in ids if i in facts}))
    return ws


def _result(*chunks):
    return RAGResult(chunks=list(chunks), formatted_context="", total_tokens=0, sources=[], query=QUESTION)


def _passage(doc_id, content, source):
    return {"content": content, "source_file": source, "similarity": 0.79, "tokens": 40,
            "document_id": str(doc_id), "metadata": {}}


def test_the_documents_text_is_read_in_order_and_only_in_the_workspace(workspace):
    texts = document_texts(workspace, [1526, 1600])

    assert list(texts) == ["1526"]
    assert texts["1526"].index("## Delivery") < texts["1526"].index("Carriage is charged per drop")
    assert "another tenant" not in texts["1526"]


def test_the_prices_passage_becomes_the_whole_terms_with_the_carriage_charges(workspace):
    prices = _passage(1526, STORED_1526[1], "wholesale-terms-2026.md")
    report = _passage(1530, REPORT, "task-minimum-order.md")
    result = asyncio.run(whole_where_short(RAGService.__new__(RAGService), _result(prices, report), workspace))

    whole = result.chunks[0]["content"]
    assert "Carriage is charged per drop: - under 12 kg: **£8.50**" in whole and "## Returns" in whole
    assert result.chunks[1]["content"] == REPORT
    assert "- 12 kg and over: **£5.00**" in result.formatted_context
    assert result.total_tokens > 80


def test_a_second_passage_of_the_same_document_is_dropped(workspace):
    first = _passage(1526, STORED_1526[1], "wholesale-terms-2026.md")
    second = _passage(1526, STORED_1526[4], "wholesale-terms-2026.md")
    result = asyncio.run(whole_where_short(RAGService.__new__(RAGService), _result(first, second), workspace))

    assert len(result.chunks) == 1 and "## Ordering" in result.chunks[0]["content"]


def test_a_longer_document_and_an_agents_report_stay_passages(workspace, monkeypatch):
    monkeypatch.setattr(config, "RAG_WHOLE_DOCUMENT_MAX_TOKENS", 50, raising=False)
    prices = _passage(1526, STORED_1526[1], "wholesale-terms-2026.md")
    report = _passage(1530, REPORT, "task-minimum-order.md")
    before = _result(prices, report)

    assert asyncio.run(whole_where_short(RAGService.__new__(RAGService), before, workspace)) is before


def test_retrieval_hands_over_the_whole_document(workspace):
    prices = _passage(1526, STORED_1526[1], "wholesale-terms-2026.md")

    async def selected(self, query, **kwargs):
        return _result(prices)

    retrieve = short_documents_whole(selected)
    result = asyncio.run(retrieve(RAGService.__new__(RAGService), QUESTION, max_chunks=5, workspace_id=workspace))

    assert "- under 12 kg: **£8.50**" in result.chunks[0]["content"]
