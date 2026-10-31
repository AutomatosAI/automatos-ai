"""P256-FIX-N17-11 (F443, night 16): the backend log says what a document search found and what the chat cut.

Ask 318 (the November price letter, 22:02 on 14 Oct, chat fb259dec) read three
platform_search_documents answers and quoted the October terms. Whether the November sheet was
among the passages, or was cut from the answer after, could not be told: the search logged nothing
of its passages and the chat's cut of a tool's answer to 2,000 tokens left no trace. Now:
- each document search logs, at INFO, the turn's conversation, the query, and each passage's file,
  score and date in the order the answer lists them (``search_passages_log``);
- each cut of a tool's answer logs '[tool-cut] <tool>: kept N of M tokens' (``consumers/chatbot/tool_cut``).
"""
from __future__ import annotations

import asyncio
import logging

from consumers.chatbot import service, tool_cut
from modules.tools.discovery import handlers_documents
from modules.tools.discovery.search_passages_log import passages_logged
from modules.tools.execution.turn_owner_words import held_conversation_id, owner_words_held

CHAT = "fb259dec-0000-4000-8000-000000000318"
QUERY = "November wholesale prices"
PASSAGES = [
    {"document_id": 31, "file_name": "harbourline-wholesale-terms-2026-11.md", "score": 0.71,
     "metadata": {"document_date": "2026-11-01", "standing": "newest"}},
    {"document_id": 30, "file_name": "wholesale-terms-2026-10.md", "score": 0.83,
     "metadata": {"document_date": "2026-10-01", "standing": "older"}},
]


async def _search(db, workspace_id, params):
    return {"success": True, "count": len(PASSAGES), "results": PASSAGES}


def test_a_document_search_logs_its_conversation_query_and_passages_in_order(caplog):
    with caplog.at_level(logging.INFO, logger="modules.tools.discovery.search_passages_log"):
        with owner_words_held({"conversation_id": CHAT, "user_query": "I need the November price letter"}):
            assert held_conversation_id() == CHAT
            result = asyncio.run(passages_logged(_search)(None, "ws", {"query": QUERY}))

    assert result["results"] == PASSAGES and held_conversation_id() is None          # released after the action
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[search-documents]")]
    assert line == (f"[search-documents] conversation={CHAT} query={QUERY!r} passages=2: "
                    "harbourline-wholesale-terms-2026-11.md score=0.71 date=2026-11-01; "
                    "wholesale-terms-2026-10.md score=0.83 date=2026-10-01")


def test_a_failed_or_empty_search_logs_what_it_holds(caplog):
    async def failed(db, workspace_id, params):
        return {"success": False, "error": "document search failed"}

    async def empty(db, workspace_id, params):
        return {"success": True, "results": [], "count": 0}

    with caplog.at_level(logging.INFO, logger="modules.tools.discovery.search_passages_log"):
        asyncio.run(passages_logged(failed)(None, "ws", {"query": QUERY}))
        asyncio.run(passages_logged(empty)(None, "ws", {"query": QUERY}))

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[search-documents]")]
    assert lines == [f"[search-documents] conversation=None query={QUERY!r} passages=0: none"]


def test_platform_search_documents_is_logged_after_it_is_dated_and_ranked():
    handler = handlers_documents.search_documents
    assert handler.__code__.co_qualname == "passages_logged.<locals>.wrapped"
    assert handler.__wrapped__.__code__.co_qualname == "dates_and_ranks.<locals>.wrap.<locals>.wrapped"


# ── The chat's cut of a tool's answer ───────────────────────────────────────────────────

class _Loop:
    @tool_cut.logs_what_the_cut_keeps
    def track(self, result, tool_name, last_tool_name, streak):
        return streak + 1


def test_a_cut_is_logged_with_what_it_keeps(caplog):
    long_answer = {"success": True, "llm_context": "passage " * 3000}

    with caplog.at_level(logging.INFO, logger="consumers.chatbot.tool_cut"):
        assert _Loop().track(long_answer, "platform_search_documents", None, 0) == 1     # the method still runs
        _Loop().track({"success": True, "llm_context": "short"}, "platform_list_tasks", None, 0)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[tool-cut]")]
    tokens = tool_cut.answer_tokens(long_answer)
    assert tokens > tool_cut.TOOL_RESULT_TOKEN_BUDGET
    assert lines == [f"[tool-cut] platform_search_documents: kept {tool_cut.TOOL_RESULT_TOKEN_BUDGET} of {tokens} tokens"]


def test_the_chat_loop_logs_its_cut_at_the_budget_it_cuts_to():
    assert service._TOOL_RESULT_TOKEN_BUDGET == tool_cut.TOOL_RESULT_TOKEN_BUDGET == 2000
    tracker = service.StreamingChatService._track_search_spiral
    assert tracker.__code__.co_qualname == "logs_what_the_cut_keeps.<locals>.wrapped"
    raw = {"raw_result": {"results": ["passage " * 3000]}}                            # no llm_context: the raw result
    assert tool_cut.answer_tokens(raw) > tool_cut.TOOL_RESULT_TOKEN_BUDGET
    assert tool_cut.answer_tokens({"llm_context": "short"}) == 0 == tool_cut.answer_tokens("not a result")
