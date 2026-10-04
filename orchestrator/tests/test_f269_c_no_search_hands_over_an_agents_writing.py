"""F269 (night 9): what an agent or a mission wrote is never handed over as the owner's source.

Night 8 left an agent's documents out of Auto's retrieval-first passages and only marked
them in the model's own search. Night 9, chat 9a128fe3: "What payment terms do we give
cafés?", then "Where did that come from?" → "a document in our knowledge base titled
'2026-10-04_125617_e5c675_task-payment-terms-for-cafés.md'", ticket #1851's own answer
filed as document #1547 (MORNING-REPORT.md L90). platform_search_documents,
platform_grep_documents, platform_read_document and platform_list_documents had no check
at all, and asked which papers were the owner's, Auto said "You haven't uploaded any
documents yourself yet" (L97).
"""
from __future__ import annotations

import inspect

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

OWNERS_TERMS = "Cafés pay within 30 days of the invoice date. Quay Coffee House is on 14 days."
AGENTS_ANSWER = ("14-day terms: Lantern Bakehouse, Quay Coffee House (moves to 30-day from November), "
                 "Marsh Espresso Bar, Tide Espresso Bar, Pier Bakehouse, Gull Espresso Bar.")
REPORT_1547 = "2026-10-04_125617_e5c675_task-payment-terms-for-cafés.md"


@pytest.fixture
def papers(db_session, seed_workspace):
    """The owner's wholesale terms, #1851's filed answer and a mission's output."""
    from core.models.core import Document

    ws = UUID(seed_workspace())

    def document(filename, source_type=None):
        made = Document(filename=filename, workspace_id=ws, status="completed", source_type=source_type)
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, terms=document("wholesale-terms-2026.md", "upload"),
              report=document(REPORT_1547, "agent_output"),
              mission=document("mission-output-calculate-september-s-retail-takings.md", "agent_output"))


def _handler(action):
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    return PLATFORM_HANDLERS[action]


def test_platform_search_documents_returns_only_the_owners_passages(monkeypatch, papers):
    from modules.rag import service as rag_service

    chunks = [{"document_id": papers.report.id, "source_file": REPORT_1547, "content": AGENTS_ANSWER,
               "similarity": 0.92},
              {"document_id": str(papers.terms.id), "source_file": "wholesale-terms-2026.md",
               "content": OWNERS_TERMS, "similarity": 0.81}]

    class _Rag:
        async def retrieve(self, **kwargs):
            return NS(chunks=chunks, sources_map=[], sources=[REPORT_1547, "wholesale-terms-2026.md"])

    monkeypatch.setattr(rag_service, "RAGService", _Rag)

    found = asyncio.run(_handler("platform_search_documents")(papers.db, papers.ws,
                                                                {"query": "payment terms for cafés"}))

    assert [p["content"] for p in found["results"]] == [OWNERS_TERMS]
    assert found["count"] == 1 and found["sources"] == ["wholesale-terms-2026.md"]


def test_platform_grep_documents_returns_only_the_owners_matches(papers):
    from services.agents_writing import owners_passages_only

    async def grep(db, workspace_id, params):
        return {"success": True, "matches": [{"document_id": papers.mission.id, "snippet": "£1,455.00"},
                                             {"document_id": papers.terms.id, "snippet": OWNERS_TERMS}], "count": 2}

    found = asyncio.run(owners_passages_only("matches")(grep)(papers.db, papers.ws, {"pattern": "terms"}))

    assert [m["snippet"] for m in found["matches"]] == [OWNERS_TERMS] and found["count"] == 1


def test_reading_an_agents_document_says_an_agent_wrote_it(papers):
    from services.agents_writing import says_an_agent_wrote_it
    from services.draft_guides import AGENTS_WRITING

    async def read(db, workspace_id, params):
        texts = {papers.report.id: AGENTS_ANSWER, papers.terms.id: OWNERS_TERMS}
        return {"success": True, "document_id": params["document_id"], "content": texts[params["document_id"]]}

    reader = says_an_agent_wrote_it(read)
    report = asyncio.run(reader(papers.db, papers.ws, {"document_id": papers.report.id}))
    terms = asyncio.run(reader(papers.db, papers.ws, {"document_id": papers.terms.id}))

    assert report["content"] == f"{AGENTS_WRITING}\n{AGENTS_ANSWER}" and report["written_by_an_agent"] is True
    assert terms["content"] == OWNERS_TERMS and terms["written_by_an_agent"] is False


def test_the_document_list_says_which_an_agent_wrote(papers):
    listed = asyncio.run(_handler("platform_list_documents")(papers.db, papers.ws, {}))

    written = {d["filename"]: d["written_by_an_agent"] for d in listed["documents"]}
    assert written == {"wholesale-terms-2026.md": False, REPORT_1547: True,
                       "mission-output-calculate-september-s-retail-takings.md": True}


def test_every_document_action_runs_through_the_check():
    from modules.tools.discovery import handlers_documents as documents

    for action, handler in (("platform_list_documents", documents.list_documents),
                            ("platform_read_document", documents.read_document),
                            ("platform_grep_documents", documents.grep_documents),
                            ("platform_search_documents", documents.search_documents)):
        wrapped = _handler(action)
        # F305 stacks the past-work scope on search: the action is wrapped, whatever the depth.
        assert wrapped is not handler and inspect.unwrap(wrapped) is inspect.unwrap(handler)


def test_autos_answer_to_an_agents_question_leaves_out_an_agents_documents(monkeypatch, papers):
    """An agent's question mid-mission is answered from the owner's corpus only."""
    from services import orchestrator_answers as answers

    async def corpus(subject, question):
        return [{"text": AGENTS_ANSWER, "source": {"type": "corpus", "document_id": str(papers.report.id)}},
                {"text": OWNERS_TERMS, "source": {"type": "corpus", "document_id": str(papers.terms.id)}}]

    async def nothing(subject, question):
        return []

    monkeypatch.setattr(answers, "_corpus_blocks", corpus)
    monkeypatch.setattr(answers, "_field_blocks", nothing)
    monkeypatch.setattr(answers, "_memory_blocks", nothing)
    monkeypatch.setattr(answers, "_fleet_blocks", lambda db, subject, question: [])
    subject = answers.ClarificationSubject(run_id="run-0035", workspace_id=papers.ws)

    blocks = asyncio.run(answers._external_blocks(papers.db, subject, "Which cafés are on 14-day terms?"))

    assert [b["text"] for b in blocks] == [OWNERS_TERMS]
