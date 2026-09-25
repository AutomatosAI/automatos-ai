"""F077/F078 (refresh-2 retest) — the prompt names the connected database and
sends a number question to it.

With harbourline_shop connected, Auto got 2 of 7 shop questions right. It
answered "how many subscribers" from memory and the documents, and on "my three
best-selling products" it asked "Could you please provide the database ID?".
Nothing in its prompt named the database. The documents section said every
question about the business was a search_knowledge first. The action catalog
shows parameter names, never their descriptions, so platform_query_data's "omit
it when the workspace has one database" never reached the model. Now the section
names the active databases (at most five, PRD-231) and says a number goes to the
database, and platform_query_data's own description carries the one-database
rule. Without a database the F085 text is unchanged
(test_f085_the_prompt_says_documents_exist).

Refresh-3 retest: still 2 of 7, and the data tool was called 0 times. The
sentence named platform_query_data, reachable only through platform_execute,
where flash sends params={} (F027); smart_query_database, which Auto holds as a
first-class tool, went unnamed. And F085-A's prefetch put a summary document's
figures (415 for 400) in front of every number question, headed "answer from
them". The sentence now names smart_query_database, and with a database the
prefetched passages say a number comes from it.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.context.sections.documents_inventory import DocumentsInventorySection, documents_summary

TABLES = ("documents", "database_knowledge_sources")
NUMBERS_TO_ONE = ("For numbers about the business (counts, totals, rankings, trends), call "
                  "smart_query_database with the question; with one database no name is needed.")
NUMBERS_TO_ONE_OF = ("For numbers about the business (counts, totals, rankings, trends), call "
                     "smart_query_database with the question and the database's name (it lists them when "
                     "none is named).")
DOCUMENTS_BESIDE_DATA = ("For what a document says or how the product works, search them with "
                         "search_knowledge first and name the file you used.")


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        for table in TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{table}"))
            session.execute(text(f"CREATE TEMP TABLE {table} (LIKE public.{table} INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        for table in TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{table}"))
        session.commit()
        session.close()


def _doc(db, doc_id, ws, name, *, day=1):
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, status, upload_date) "
                    "VALUES (:id, CAST(:ws AS uuid), :name, 'completed', make_timestamp(2026, 9, :day, 12, 0, 0))"),
               {"id": doc_id, "ws": ws, "name": name, "day": day})


def _database(db, source_id, ws, name, *, active=True, minute=0):
    db.execute(text("INSERT INTO database_knowledge_sources "
                    "(id, workspace_id, tenant_id, name, credential_id, dialect, is_active, created_at) "
                    "VALUES (:id, CAST(:ws AS uuid), 1, :name, 1, 'postgresql', :active, "
                    "make_timestamp(2026, 9, 22, 12, :minute, 0))"),
               {"id": source_id, "ws": ws, "name": name, "active": active, "minute": minute})


def test_one_connected_database_is_named_and_a_number_goes_to_platform_query_data(db):
    ws = str(uuid4())
    _doc(db, 1, ws, "brand-voice.md", day=1)
    _doc(db, 2, ws, "wholesale-price-list.csv", day=20)
    _database(db, 36, ws, "harbourline_shop")
    assert documents_summary(db, ws) == (
        "## Documents and data in this workspace\n"
        "This workspace holds 2 of the owner's documents (wholesale-price-list.csv, brand-voice.md). "
        f"{DOCUMENTS_BESIDE_DATA}\n"
        f"This workspace has 1 connected database (harbourline_shop). {NUMBERS_TO_ONE}")


def test_a_database_without_documents_gets_the_database_sentences_alone(db):
    ws = str(uuid4())
    _database(db, 36, ws, "harbourline_shop")
    assert documents_summary(db, ws) == (
        "## Documents and data in this workspace\n"
        f"This workspace has 1 connected database (harbourline_shop). {NUMBERS_TO_ONE}")


def test_several_databases_are_named_newest_first_up_to_five_then_counted(db):
    ws, crowded = str(uuid4()), str(uuid4())
    _database(db, 36, ws, "harbourline_shop", minute=1)
    _database(db, 37, ws, "crm", minute=2)
    assert documents_summary(db, ws).endswith(
        f"This workspace has 2 connected databases (crm, harbourline_shop). {NUMBERS_TO_ONE_OF}")
    for i in range(7):
        _database(db, 50 + i, crowded, f"shard_{i}", minute=i)
    assert documents_summary(db, crowded).endswith(f"This workspace has 7 connected databases. {NUMBERS_TO_ONE_OF}")


def test_only_this_workspaces_active_databases_are_named(db):
    ws = str(uuid4())
    _database(db, 36, ws, "harbourline_shop", minute=1)
    _database(db, 37, ws, "old_shop", active=False, minute=2)
    _database(db, 38, str(uuid4()), "someone_elses_crm", minute=3)
    assert documents_summary(db, ws).endswith(f"This workspace has 1 connected database (harbourline_shop). {NUMBERS_TO_ONE}")


def test_ten_long_titles_and_five_databases_fit_the_section_budget_whole(db):
    from core.context_guard import count_tokens

    ws = str(uuid4())
    for i in range(1, 13):
        _doc(db, i, ws, f"green-coffee-wholesale-price-list-autumn-2026-v{i:02d}.csv", day=i)
    for i in range(5):
        _database(db, 60 + i, ws, f"harbourline_shop_region_{i}", minute=i)
    summary = documents_summary(db, ws)
    assert summary.endswith(NUMBERS_TO_ONE_OF)
    assert count_tokens(summary) <= DocumentsInventorySection.max_tokens


def test_the_catalog_line_for_platform_query_data_carries_the_one_database_rule():
    """The catalog renders an action's description and parameter NAMES only."""
    from modules.tools.discovery.action_registry import ActionRegistry, get_action_registry

    line = ActionRegistry._format_action_line(get_action_registry().get("platform_query_data"))
    assert "With one database connected, pass only the question: that one is used." in line
    assert "Name a database (database_id) only when several are connected." in line


# ── refresh-3 retest: the database was offered, never called (0 of 7) ──────

def _prefetch(monkeypatch, databases):
    import asyncio as _asyncio

    from consumers.chatbot import knowledge_prefetch as kp
    from modules.context.sections import documents_inventory as inv

    monkeypatch.setattr(kp, "documents_in", lambda db, ws: 3)
    monkeypatch.setattr(inv, "connected_databases", lambda db, ws: databases)

    async def _search(args):
        return {"raw_result": {"results": [
            {"filename": "subscriber-numbers-august.md", "similarity": 0.91, "content": "415 active subscribers"}]}}

    got = _asyncio.run(kp.prefetch(None, uuid4(), "How many active subscribers are on each plan?",
                                   search=_search, enabled=True, limit=5, min_score=0.3))
    return kp, got.message["content"]


def test_with_a_database_the_prefetched_passages_send_a_number_to_it(monkeypatch):
    kp, content = _prefetch(monkeypatch, ["harbourline_shop"])
    assert content.startswith(f"{kp.PREFETCH_HEADER} {kp.PREFETCH_DATABASE_NOTE}\n\n")
    assert "subscriber-numbers-august.md" in content


def test_without_one_the_passages_go_as_they_did(monkeypatch):
    kp, content = _prefetch(monkeypatch, [])
    assert content.startswith(f"{kp.PREFETCH_HEADER}\n\n")
