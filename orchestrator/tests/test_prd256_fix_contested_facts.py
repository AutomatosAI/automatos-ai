"""P256-FIX-N13-3 (F410, night 13): a corrected fact wins over its superseded copies.

Night 13: the owner told Auto 'Club dispatch day from November is Wednesday' (a stored memory)
and wrote it into a ruling document; asked cold, Auto said Monday on 4 of 17 asks, because five
older documents still said Monday. Now, where the answer context is assembled
(``services.contested_facts``): the owner's standing rule comes first and outranks any document,
every passage carries its document's date and is listed newest first (marked older than the
owner's newest rule where it is), and the same ask always reads the same context.
"""
from __future__ import annotations

import asyncio
import random
from datetime import date, datetime
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import uuid4

from sqlalchemy.orm import Session

from consumers.chatbot import knowledge_prefetch as kp
from services import contested_facts as cf

OWNER = "user:7"
RULE = "From November we post Wednesdays."
RULE_ROW = {"memory": RULE, "created_at": "2026-10-11T22:05:00+00:00",
            "metadata": {"type": "preference", "owner": OWNER, "stated_by": "owner"}}
ASK = "What day do we post the club boxes from November?"
MONDAY_FILES = ["packing-checklist.md", "club-faq.md", "dispatch-calendar.md", "courier-sheet.md", "club-terms.md"]
OLDER_THAN_THE_RULE = cf.OLDER_THAN_THE_RULES.format(stated="2026-10-11")


def _passage(name, doc_id, when, *, says="Club boxes go out on Monday.", similarity=0.8):
    meta = {cf.DATE_KEY: when} if when else {}
    return {"filename": name, "source": name, "title": name, "content": says, "excerpt": says,
            "similarity": similarity, "document_id": doc_id, "metadata": meta}


MONDAYS = [_passage(name, 11 + i, f"2026-09-0{i + 1}", similarity=0.8 - i / 100) for i, name in enumerate(MONDAY_FILES)]
RULING = _passage("dispatch-ruling.md", 30, "2026-10-11", says="From November club boxes go out on Wednesday.",
                  similarity=0.6)


class _Search:
    def __init__(self, results):
        self.results = results

    async def __call__(self, args):
        return {"success": True, "raw_result": {"success": True, "results": list(self.results)}, "frontend_data": {}}


class _MemoryService:
    """The one shared memory service, as the standing rules read it (its durable store's scroll)."""

    def __init__(self, rows):
        self._durable = NS(get_where_any=AsyncMock(return_value=rows))

    @staticmethod
    def namespace(workspace_id):
        return NS(resolve=lambda _agent: f"ws:{workspace_id}")


def _prefetch(monkeypatch, passages, *, viewer=OWNER, rows=(RULE_ROW,)):
    import modules.memory.unified_memory_service as ums

    monkeypatch.setattr(kp, "documents_in", lambda db, ws: 6)
    monkeypatch.setattr(kp, "_has_database", lambda db, ws: False)
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: _MemoryService(list(rows)))
    got = asyncio.run(kp.prefetch(None, uuid4(), ASK, search=_Search(passages), enabled=True, limit=5,
                                  min_score=0.3, viewer=viewer))
    return got.message["content"]


def _sources(content):
    return [line for line in content.splitlines() if line.startswith("[Source ")]


# ---------------------------------------------------------------------------
# The owner's stated rule outranks any document
# ---------------------------------------------------------------------------

def test_the_owners_rule_comes_first_and_the_monday_passages_are_marked_older(monkeypatch):
    content = _prefetch(monkeypatch, MONDAYS)
    assert content.startswith(kp.PREFETCH_HEADER)
    assert f"- {RULE}" in content and cf.RULES_OUTRANK_DOCUMENTS in content
    sources = _sources(content)
    assert len(sources) >= 4                                            # format_for_llm's top four
    assert content.index(f"- {RULE}") < content.index(sources[0])      # night 13: the rule was outnumbered
    assert all(OLDER_THAN_THE_RULE in line for line in sources)
    assert cf.RANKING_RULE in content                                   # the winning value, the conflict in one line


def test_a_newer_document_than_the_rule_is_not_marked_older_than_it(monkeypatch):
    content = _prefetch(monkeypatch, [MONDAYS[0], RULING])
    first, second = _sources(content)
    assert first.startswith("[Source 1: dispatch-ruling.md]") and "dated 2026-10-11, newest" in first
    assert second.startswith("[Source 2: packing-checklist.md]") and OLDER_THAN_THE_RULE in second


def test_no_person_or_no_rule_puts_no_rules_above_the_passages(monkeypatch):
    for viewer, rows in ((None, (RULE_ROW,)), (OWNER, ())):
        content = _prefetch(monkeypatch, MONDAYS, viewer=viewer, rows=rows)
        assert RULE not in content and "## Standing Rules" not in content
        assert not any(OLDER_THAN_THE_RULE in line for line in _sources(content))


def test_the_standing_rules_block_says_the_rules_outrank_documents(monkeypatch):
    import modules.memory.unified_memory_service as ums
    from modules.context.sections.memory import owners_rules, standing_rules_block

    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: _MemoryService([RULE_ROW]))
    block = asyncio.run(standing_rules_block(uuid4(), viewer_subject_id=OWNER, widget_mode=False))
    assert f"- {RULE}" in block and cf.RULES_OUTRANK_DOCUMENTS in block
    rules = asyncio.run(owners_rules(uuid4(), viewer_subject_id=OWNER, widget_mode=False))
    assert (rules.texts, rules.stated) == ((RULE,), "2026-10-11")


# ---------------------------------------------------------------------------
# Among documents the newest outranks the older
# ---------------------------------------------------------------------------

def test_two_dated_documents_that_disagree_put_the_newer_first():
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    older = _passage("packing-checklist.md", 3, "2026-09-02", similarity=0.9)
    text = ToolResultFormatter.format_for_llm({"success": True, "results": [older, RULING]}, "search_knowledge")
    first, second = _sources(text)
    assert first == "[Source 1: dispatch-ruling.md] (document 30; dated 2026-10-11, newest; 60.0%)"
    assert second == "[Source 2: packing-checklist.md] (document 3; dated 2026-09-02, older; 90.0%)"
    assert text.index(RULING["content"]) < text.index(older["content"])


def test_undated_passages_keep_the_old_source_line():
    line = cf.ranked([_passage("notes.md", 5, None, similarity=0.5)])
    from modules.tools.formatting.llm_summary_parts import source_line

    assert source_line(1, line[0]) == "\n[Source 1: notes.md] (document 5; 50.0%)"


def test_ten_identical_asks_read_one_context(monkeypatch):
    tied = [_passage(name, 40 + i, f"2026-09-0{i + 1}", similarity=0.7) for i, name in enumerate(MONDAY_FILES)]
    contents = set()
    for seed in range(10):
        arrived = [*tied, RULING]
        random.Random(seed).shuffle(arrived)          # the search's order of score ties; five tie for four places
        contents.add(_prefetch(monkeypatch, arrived))
    assert len(contents) == 1
    assert len(_sources(contents.pop())) == 4


def test_score_ties_fall_to_the_document_never_to_arrival():
    first, second = _passage("b.md", 9, None, similarity=0.7), _passage("a.md", 8, None, similarity=0.7)
    assert cf.in_score_order([first, second]) == cf.in_score_order([second, first]) == [second, first]


# ---------------------------------------------------------------------------
# A passage's date: its name's, else its upload's, read in the workspace
# ---------------------------------------------------------------------------

def test_a_documents_date_is_its_names_else_its_upload_day():
    uploaded = datetime(2026, 9, 3, 14, 30)
    assert cf.document_date("club-ruling-2026-10-11.md", uploaded) == "2026-10-11"
    assert cf.document_date("stock-2026-13-45.md", uploaded) == "2026-09-03"     # not a date: the upload's
    assert cf.document_date("notes.md", date(2026, 8, 1)) == "2026-08-01"
    assert cf.document_date("notes.md", None) is None


def test_document_dates_read_only_the_callers_workspace():
    workspace = uuid4()
    seen = []

    class _Query:
        def filter(self, *clauses):
            seen.extend(str(c) for c in clauses)
            return self

        def all(self):
            return [NS(id=3, filename="u_83f.md", original_filename="packing-checklist.md",
                       upload_date=datetime(2026, 9, 2, 8))]

    got = cf.document_dates(NS(query=lambda *columns: _Query()), workspace, ["3", 3, None, "x"])
    assert got == {3: "2026-09-02"}
    assert any("documents.workspace_id" in clause for clause in seen)


def test_dated_stamps_each_passage_and_leaves_the_rest(monkeypatch):
    monkeypatch.setattr(cf, "document_dates", lambda db, ws, ids: {3: "2026-09-02"})
    result = {"success": True, "results": [_passage("a.md", 3, None), _passage("b.md", 4, None)]}
    got = cf.dated(Session(), uuid4(), result)
    assert [cf.passage_date(p) for p in got["results"]] == ["2026-09-02", None]
    assert result["results"][0]["metadata"] == {}                       # the caller's result is not changed
    assert cf.dated(object(), uuid4(), result) is result                # no session: as it was
    assert cf.dated(Session(), None, result) is result                  # no workspace: as it was


def test_platform_search_documents_lists_newest_first(monkeypatch):
    monkeypatch.setattr(cf, "document_dates", lambda db, ws, ids: {3: "2026-09-02", 30: "2026-10-11"})

    async def handler(db, workspace_id, params):
        return {"success": True, "count": 2, "results": [
            {"document_id": 3, "file_name": "packing-checklist.md", "score": 0.9, "content": "Monday"},
            {"document_id": 30, "file_name": "dispatch-ruling.md", "score": 0.6, "content": "Wednesday"}]}

    got = asyncio.run(cf.dates_and_ranks("results")(handler)(Session(), uuid4(), {"query": "dispatch day"}))
    assert [p["document_id"] for p in got["results"]] == [30, 3]
    assert [p["metadata"][cf.STANDING_KEY] for p in got["results"]] == [cf.NEWEST, cf.OLDER]


def test_the_model_s_own_search_is_dated_after_the_agents_writing_is_left_out(monkeypatch):
    import services.agents_writing as agents_writing
    from modules.tools.execution import exec_platform

    calls = []
    monkeypatch.setattr(agents_writing, "owners_search", lambda db, result, agent: {**result, "owners": True})
    monkeypatch.setattr(cf, "dated_for_agent", lambda db, result, agent: calls.append((result, agent)) or result)
    executor = NS(platform_tools=NS(db="db", execute_tool=AsyncMock(return_value={"success": True, "results": []})))
    asyncio.run(exec_platform.execute_platform_tool(executor, "search_knowledge", {"query": "dispatch day"}, 294))
    assert calls == [({"success": True, "results": [], "owners": True}, 294)]


def test_the_chat_turn_hands_the_prefetch_its_person(monkeypatch):
    from consumers.chatbot import service

    seen = {}

    async def prefetch(*args, **kwargs):
        seen.update(kwargs)
        return None

    from consumers.chatbot.streaming import get_streaming_handler

    monkeypatch.setattr(kp, "prefetch", prefetch)
    svc = service.StreamingChatService.__new__(service.StreamingChatService)
    svc.db, svc.workspace_id, svc.streaming_handler = object(), "ws", get_streaming_handler()
    svc._turn_document_ids, svc._turn_chunk_ids, svc._viewer_subject_id = set(), set(), OWNER
    svc.tool_router = NS(execute_and_format=AsyncMock())
    llm_messages = [{"role": "system", "content": "You are Auto."}]

    async def collect():
        return [f async for f in svc._retrieval_first(ASK, llm_messages, NS(agent_id=294), "chat-1", [])]

    assert asyncio.run(collect()) == [] and seen["viewer"] == OWNER
