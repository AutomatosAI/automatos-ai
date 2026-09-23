"""F085-B (night 3) — the prompt says the workspace has documents, and the tool
texts send a question to them.

Night 3's RAG test: the product's own 54-page manual sat in the workspace and
Auto searched it once in 40 questions (20 find_tools calls). Nothing said
documents existed; search_knowledge called itself "local docs only", while
find_tools and the actions section said every capability was a catalog search
away. Now a section names what the workspace holds, and all three texts send a
QUESTION to search_knowledge and an ACTION to find_tools.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.context.sections.documents_inventory import DocumentsInventorySection, documents_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text("DROP TABLE IF EXISTS pg_temp.documents"))
        session.execute(text("CREATE TEMP TABLE documents (LIKE public.documents INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        session.execute(text("DROP TABLE IF EXISTS pg_temp.documents"))
        session.commit()
        session.close()


def _doc(db, doc_id, ws, name, *, report=False, day=1, status="completed"):
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, source_type, status, upload_date) "
                    "VALUES (:id, CAST(:ws AS uuid), :name, :st, :status, make_timestamp(2026, 9, :day, 12, 0, 0))"),
               {"id": doc_id, "ws": ws, "name": name, "st": "agent_output" if report else None,
                "status": status, "day": day})


def test_the_section_names_the_owners_newest_documents_and_counts_the_reports(db):
    ws = str(uuid4())
    _doc(db, 1, ws, "brand-voice.md", day=1)
    _doc(db, 2, ws, "green-coffee-list-autumn-2026.csv", day=22)
    _doc(db, 3, ws, "q3-report.md", report=True, day=20)
    _doc(db, 4, ws, "half-uploaded.pdf", status="processing")          # not searchable yet
    assert documents_summary(db, ws) == (
        "## Documents in this workspace\n"
        "This workspace holds 2 of the owner's documents (green-coffee-list-autumn-2026.csv, brand-voice.md) "
        "and 1 report its agents saved. For a question about the business, a document, or how the product "
        "works, search them with search_knowledge first and name the file you used.")


def test_a_long_list_is_cut_to_the_newest_ten_and_an_empty_workspace_has_no_section(db):
    ws = str(uuid4())
    for i in range(1, 13):
        _doc(db, i, ws, f"page-{i:02d}.md", day=i)
    summary = documents_summary(db, ws)
    assert "(page-12.md, page-11.md," in summary and ", +2 more)" in summary and "page-02.md" not in summary
    assert documents_summary(db, str(uuid4())) is None
    rendered = asyncio.run(DocumentsInventorySection().render(NS(db_session=db, workspace_id=str(uuid4()))))
    assert rendered == ""


def test_a_workspace_holding_only_reports_counts_them_as_reports(db):
    ws = str(uuid4())
    for i in range(3):
        _doc(db, 10 + i, ws, f"report-{i}.md", report=True, day=i + 1)
    assert documents_summary(db, ws) == (
        "## Documents in this workspace\n"
        "This workspace holds 3 reports its agents saved. For a question about the business, a document, or "
        "how the product works, search them with search_knowledge first and name the file you used.")


def test_the_section_is_on_autos_prompt_after_the_cached_prefix():
    from modules.context.modes import MODE_CONFIGS, ContextMode
    from modules.context.sections import SECTION_REGISTRY
    from modules.context.service import VOLATILE_SECTIONS

    assert "documents_inventory" in MODE_CONFIGS[ContextMode.CHATBOT].sections
    assert SECTION_REGISTRY["documents_inventory"] is DocumentsInventorySection
    assert "documents_inventory" in VOLATILE_SECTIONS        # counts move whenever an agent saves a report


def test_all_three_texts_send_a_question_to_the_documents():
    find_tools = (ROOT / "modules/tools/discovery/actions_capabilities.py").read_text()
    search = (ROOT / "modules/tools/registry/tool_registry.py").read_text()
    actions = (ROOT / "modules/context/sections/platform_actions.py").read_text()
    assert "This finds tools, not answers" in find_tools and "search_knowledge first" in find_tools
    assert "never assume something is impossible without checking here first" not in find_tools
    assert "Use it FIRST for any question" in search and "searches local docs only" not in search
    assert "For a QUESTION about the business" in actions and "`search_knowledge` first" in actions
