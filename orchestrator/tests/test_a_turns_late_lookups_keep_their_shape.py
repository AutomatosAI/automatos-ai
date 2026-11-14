"""Voice latency D-1: what a turn's two late lookups hand the turn, written down before the move.

Just before its first model call a chat turn looks up its connected apps
(``StreamingChatService._inject_composio_tools``) and searches its documents
(``StreamingChatService._retrieval_first``). D-1 splits each lookup from the code that merges
its answer into the turn, with no switch, so this file pins what each method gives the turn
today: the messages, the tools, the returned result, the frames, ``prefetched``, the turn's
retrieval record and the warning lines. It was written against the methods as they stood
before the move and must pass with no edit after it.

Every expected value is a literal read from the code, never computed by a second copy of it.
The methods are the real ones, on a ``StreamingChatService.__new__`` object, with the fakes the
F105 and F085 tests use: fake Composio services on the real lookup threads, a fake
``execute_and_format``. Two renderers outside the move are faked so the literals stay short:
the passage formatter (``knowledge_prefetch._passages``) and the database-sources read
(``knowledge_prefetch._has_database``).

DB-free: no database, model, SDK or socket.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import uuid
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from config import config
from consumers.chatbot import knowledge_prefetch as kp
from modules.tools.services.composio_hint_service import ComposioHintResult
from modules.tools.services.composio_tool_service import ComposioToolResult

# --------------------------------------------------------------------------- shared

WARNED = ("[F085]", "Composio tool injection failed", "[composio-lookup]")


def _warnings(caplog):
    """The warning lines the two methods write, in order."""
    said = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    return [line for line in said if line.startswith(WARNED)]


# --------------------------------------------------------------------------- the apps lookup

WS = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
AGENT = 1
PROMPT = "Email the café its invoice."
GIVE_UP_SECONDS = 5.0
COMPOSIO_EXECUTE = {"type": "function", "function": {
    "name": "composio_execute",
    "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
}}
LIST_AGENTS = {"type": "function", "function": {
    "name": "platform_list_agents", "parameters": {"type": "object", "properties": {}},
}}
SEND_EMAIL = {"type": "function", "function": {
    "name": "GMAIL_SEND_EMAIL", "description": "Send an email.",
    "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
}}
SCOPE_MESSAGE = {"role": "system", "content": (
    "You have these external apps connected: gmail.\n"
    "The relevant actions are available as individual tools with their "
    "parameter schemas. Call them directly by name — do NOT use composio_execute."
)}
HINT_MESSAGE = {"role": "system", "content": "Use GMAIL_SEND_EMAIL.\nIts params: to, subject, body."}
MESSAGES = [{"role": "system", "content": "identity"}, {"role": "system", "content": "context"},
            {"role": "user", "content": PROMPT}]


def _tools_found():
    return ComposioToolResult(tools=[SEND_EMAIL], action_set={"GMAIL_SEND_EMAIL"}, entity_id="entity-c1",
                              app_names=["gmail"], strategy="sdk_search", search_ms=12)


def _hint_lines():
    return ComposioHintResult(hint_lines=["Use GMAIL_SEND_EMAIL.", "Its params: to, subject, body."],
                              allowed_apps=["GMAIL"], matched_actions=["GMAIL_SEND_EMAIL"],
                              strategy_used="token_filtered")


def _answer(what):
    if isinstance(what, BaseException):
        raise what
    return what


@pytest.fixture
def composio(monkeypatch):
    """Fake Composio services on the real lookup threads. ``composio.found`` and
    ``composio.hints`` are what each lookup answers (an exception is raised);
    ``composio.asked`` is every lookup made, in order, with its arguments."""
    script = NS(found=ComposioToolResult(), hints=ComposioHintResult(), asked=[])

    class _ToolService:
        def __init__(self, db):
            pass

        def get_tools_for_step(self, **kwargs):
            script.asked.append(("tools", kwargs))
            return _answer(script.found)

    class _HintService:
        def __init__(self, db):
            pass

        def build_hints(self, **kwargs):
            script.asked.append(("hints", kwargs))
            return _answer(script.hints)

    monkeypatch.setattr(config, "COMPOSIO_LOOKUP_TIMEOUT_SECONDS", GIVE_UP_SECONDS, raising=False)
    monkeypatch.setattr("modules.tools.services.composio_tool_service.ComposioToolService", _ToolService)
    monkeypatch.setattr("modules.tools.services.composio_hint_service.ComposioHintService", _HintService)
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: MagicMock(name="the lookup's session"))
    return script


async def _inject(given, *, latest_text=PROMPT, skip_composio=False, assessment=None):
    """The real ``_inject_composio_tools`` on a bare service: (messages, tools returned, result)."""
    from consumers.chatbot.service import StreamingChatService

    chat = StreamingChatService.__new__(StreamingChatService)
    chat.db, chat.workspace_id = MagicMock(name="the turn's session"), WS
    messages = copy.deepcopy(MESSAGES)
    tools, found = await chat._inject_composio_tools(
        messages, given, latest_text, AGENT, None, skip_composio, assessment)
    return messages, tools, found


@pytest.mark.asyncio
async def test_tools_found_replace_the_generic_tool_and_the_scope_message_goes_third(composio, caplog):
    composio.found = _tools_found()
    given = [COMPOSIO_EXECUTE, LIST_AGENTS]

    with caplog.at_level(logging.WARNING):
        messages, tools, found = await _inject(given, assessment=NS(tool_hints=["gmail", "send email"]))

    assert tools == [LIST_AGENTS, SEND_EMAIL]
    assert given == [COMPOSIO_EXECUTE, LIST_AGENTS]          # the list given is not changed in place
    assert found is composio.found
    assert messages == [MESSAGES[0], MESSAGES[1], SCOPE_MESSAGE, MESSAGES[2]]
    assert composio.asked == [("tools", {"agent_id": AGENT, "workspace_id": WS, "task_prompt": "gmail send email",
                                         "tool_hints": ["gmail", "send email"]})]
    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_tools_found_for_a_turn_with_no_tools_are_the_turns_tools(composio):
    composio.found = _tools_found()

    messages, tools, found = await _inject(None)

    assert tools == [SEND_EMAIL] and tools is composio.found.tools
    assert found is composio.found
    assert messages == [MESSAGES[0], MESSAGES[1], SCOPE_MESSAGE, MESSAGES[2]]
    assert composio.asked == [("tools", {"agent_id": AGENT, "workspace_id": WS, "task_prompt": PROMPT,
                                         "tool_hints": []})]


@pytest.mark.asyncio
async def test_no_tools_and_hint_lines_put_the_hints_third(composio, caplog):
    composio.hints = _hint_lines()
    given = [COMPOSIO_EXECUTE, LIST_AGENTS]

    with caplog.at_level(logging.WARNING):
        messages, tools, found = await _inject(given)

    assert tools is given and tools == [COMPOSIO_EXECUTE, LIST_AGENTS]
    assert found is composio.found                            # the search's own (empty) result
    assert messages == [MESSAGES[0], MESSAGES[1], HINT_MESSAGE, MESSAGES[2]]
    assert composio.asked == [
        ("tools", {"agent_id": AGENT, "workspace_id": WS, "task_prompt": PROMPT, "tool_hints": []}),
        ("hints", {"agent_id": AGENT, "prompt": PROMPT, "workspace_id": WS}),
    ]
    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_no_tools_and_no_hint_lines_leave_the_turn_as_it_was(composio, caplog):
    given = [COMPOSIO_EXECUTE, LIST_AGENTS]

    with caplog.at_level(logging.WARNING):
        messages, tools, found = await _inject(given)

    assert tools is given and tools == [COMPOSIO_EXECUTE, LIST_AGENTS]
    assert found is composio.found
    assert messages == MESSAGES
    assert [name for name, _ in composio.asked] == ["tools", "hints"]
    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_a_search_that_gives_up_leaves_the_turn_without_composio(composio, caplog):
    from core.composio.off_loop import ComposioLookupTimeout

    composio.found = ComposioLookupTimeout("no answer from Composio")
    given = [COMPOSIO_EXECUTE, LIST_AGENTS]

    with caplog.at_level(logging.WARNING):
        messages, tools, found = await _inject(given)

    assert tools is given and found is None
    assert messages == MESSAGES
    assert [name for name, _ in composio.asked] == ["tools"]  # the hints are not asked after it
    assert _warnings(caplog) == [
        "[composio-lookup] chat turn (agent 1): tool search gave up after 5s waiting on the Composio SDK "
        "(before naming an app); this turn goes on without Composio tools",
    ]


@pytest.mark.asyncio
async def test_a_failed_hint_lookup_still_returns_the_searchs_result(composio, caplog):
    composio.hints = RuntimeError("the hint lookup broke")
    given = [COMPOSIO_EXECUTE, LIST_AGENTS]

    with caplog.at_level(logging.WARNING):
        messages, tools, found = await _inject(given)

    assert tools is given and found is composio.found
    assert messages == MESSAGES
    assert [name for name, _ in composio.asked] == ["tools", "hints"]
    assert _warnings(caplog) == ["Composio tool injection failed for agent 1: the hint lookup broke"]


@pytest.mark.asyncio
async def test_a_scope_message_that_fails_after_the_merge_still_returns_the_merged_tools(
        composio, caplog, monkeypatch):
    def _broken(app_names):
        raise RuntimeError("the scope message broke")

    monkeypatch.setattr("api.recipe_executor._composio_scope_message", _broken)
    composio.found = _tools_found()
    given = [COMPOSIO_EXECUTE, LIST_AGENTS]

    with caplog.at_level(logging.WARNING):
        messages, tools, found = await _inject(given)

    assert tools == [LIST_AGENTS, SEND_EMAIL]                 # merged before the message failed
    assert given == [COMPOSIO_EXECUTE, LIST_AGENTS]
    assert found is composio.found
    assert messages == MESSAGES
    assert _warnings(caplog) == ["Composio tool injection failed for agent 1: the scope message broke"]


@pytest.mark.asyncio
@pytest.mark.parametrize("turn", [{"skip_composio": True}, {"latest_text": ""}])
async def test_a_turn_that_skips_the_apps_or_says_nothing_makes_no_lookup(composio, caplog, turn):
    given = [COMPOSIO_EXECUTE, LIST_AGENTS]

    with caplog.at_level(logging.WARNING):
        messages, tools, found = await _inject(given, **turn)

    assert tools is given and found is None
    assert messages == MESSAGES
    assert composio.asked == []
    assert _warnings(caplog) == []


# --------------------------------------------------------------------------- the knowledge prefetch

ASK = "Which word does the brand voice guide ban?"
ASK_AGAIN = "Which word does the brand voice guide prefer?"
INSTRUCTION = "Create an agent called OPS"
DRAFTING = ("I need the November price letter to go out to the cafes this week. Write it from the new wholesale "
            "terms: the price per kilo and the minimum order.")
HIT = {"filename": "brand-voice.md", "source": "brand-voice.md", "title": "brand-voice.md", "similarity": 0.62,
       "content": "Never write 'moreish'.", "excerpt": "Never write 'moreish'.", "document_id": 720}
WEAK = {"filename": "q3-report.md", "source": "q3-report.md", "title": "q3-report.md", "similarity": 0.12,
        "content": "Q3 went well.", "excerpt": "Q3 went well.", "document_id": 900}
NOVEMBER = {"filename": "harbourline-wholesale-terms-2026-11.md", "source": "harbourline-wholesale-terms-2026-11.md",
            "title": "harbourline-wholesale-terms-2026-11.md", "similarity": 0.71, "document_id": 731,
            "content": "From 1 November: £26/kg, minimum order 8 kg.", "excerpt": "From 1 November: £26/kg."}
IDENTITY = {"role": "system", "content": "You are Auto."}
PASSAGES = 5
MIN_SCORE = 0.3
CALL_ID = re.compile(r"^retrieval-first-[0-9a-f]{12}$")
MASKED_ID = "<call id>"
MASKED_MS = "<ms>"


def _searched_as(query, limit=PASSAGES):
    """One ``execute_and_format`` call the turn makes for ``query``, whole."""
    return {"tool_name": "search_knowledge", "tool_args": {"query": query, "limit": limit}, "agent_id": 294,
            "workspace_id": "ws", "original_intent": query,
            "caller_context": {"user_query": query, "conversation_id": "chat-1", "retrieval_first": True}}


def _chat(monkeypatch, *, dial_on=True, documents=2, answers=((HIT,),)):
    """A bare service whose searches answer ``answers`` in turn (an exception is raised).
    ``documents`` is the workspace's document count, or the exception the count raises."""
    from consumers.chatbot import service
    from consumers.chatbot.streaming import get_streaming_handler

    searched, left = [], list(answers)

    async def execute_and_format(**kw):
        searched.append(kw)
        answer = _answer(left.pop(0))
        return {"success": True, "raw_result": {"success": True, "results": list(answer)}, "frontend_data": {}}

    monkeypatch.setattr(kp, "documents_in", lambda db, ws: _answer(documents))
    monkeypatch.setattr(kp, "_has_database", lambda db, ws: False)
    monkeypatch.setattr(kp, "_passages", lambda kept: "<passages: " + ", ".join(p["filename"] for p in kept) + ">")
    monkeypatch.setattr(type(config), "CHATBOT_KNOWLEDGE_PREFETCH", property(lambda self: dial_on))
    monkeypatch.setattr(config, "KNOWLEDGE_PREFETCH_PASSAGES", PASSAGES, raising=False)
    monkeypatch.setattr(config, "KNOWLEDGE_PREFETCH_MIN_SCORE", MIN_SCORE, raising=False)
    svc = service.StreamingChatService.__new__(service.StreamingChatService)
    svc.db, svc.workspace_id, svc.streaming_handler = object(), "ws", get_streaming_handler()
    svc._turn_document_ids, svc._turn_chunk_ids = set(), set()
    svc.tool_router = NS(execute_and_format=execute_and_format)
    return svc, searched


def _turn(svc, message):
    """The real ``_retrieval_first``: (frames, llm_messages, prefetched)."""
    llm_messages, prefetched = [dict(IDENTITY)], []

    async def collect():
        return [f async for f in svc._retrieval_first(message, llm_messages, NS(agent_id=294), "chat-1", prefetched)]

    return asyncio.run(collect()), llm_messages, prefetched


def _masked(frames):
    """The frames as data, the call id and the search's duration masked (both are checked first)."""
    events = [json.loads(frame[len("d:"):]) for frame in frames]
    assert all(frame.startswith("d:") and frame.endswith("\n") for frame in frames)
    assert len({event["data"]["toolCallId"] for event in events}) <= 1     # one call id for the pair
    for event in events:
        assert CALL_ID.match(event["data"]["toolCallId"])
        event["data"]["toolCallId"] = MASKED_ID
        if "durationMs" in event["data"]:
            assert isinstance(event["data"]["durationMs"], int) and event["data"]["durationMs"] >= 0
            event["data"]["durationMs"] = MASKED_MS
    return events


def _frames_for(query, summary):
    """The two activity-trail frames of one automatic search, as data."""
    return [
        {"type": "tool-start", "data": {"toolCallId": MASKED_ID, "toolName": "search_knowledge",
                                        "input": {"query": query, "limit": PASSAGES, "automatic": True}}},
        {"type": "tool-end", "data": {"toolCallId": MASKED_ID, "toolName": "search_knowledge", "success": True,
                                      "durationMs": MASKED_MS, "summary": summary}},
    ]


def _untouched(svc, frames, llm_messages, prefetched):
    """The turn as it was: no frame, no message, nothing recorded as run."""
    return (frames, llm_messages, prefetched) == ([], [IDENTITY], [])


def test_a_question_with_passages_shows_the_search_and_hands_the_passages_over(monkeypatch, caplog):
    svc, searched = _chat(monkeypatch, answers=((HIT, WEAK),))

    with caplog.at_level(logging.WARNING):
        frames, llm_messages, prefetched = _turn(svc, ASK)

    assert searched == [_searched_as(ASK)]
    assert _masked(frames) == _frames_for(ASK, "1 passage from brand-voice.md — searched automatically")
    assert prefetched == [("search_knowledge", {"query": ASK, "limit": 5})]
    assert llm_messages == [IDENTITY, {"role": "system",
                                       "content": f"{kp.PREFETCH_HEADER}\n\n<passages: brand-voice.md>"}]
    assert svc._turn_document_ids == {720, 900}               # every passage the search returned
    assert _warnings(caplog) == []


def test_a_question_with_nothing_above_the_floor_shows_the_search_and_adds_no_message(monkeypatch, caplog):
    svc, searched = _chat(monkeypatch, answers=((WEAK,),))

    with caplog.at_level(logging.WARNING):
        frames, llm_messages, prefetched = _turn(svc, ASK)

    assert searched == [_searched_as(ASK)]
    assert _masked(frames) == _frames_for(
        ASK, "searched automatically — nothing above the relevance floor (1 found)")
    assert prefetched == [("search_knowledge", {"query": ASK, "limit": 5})]
    assert llm_messages == [IDENTITY]
    assert svc._turn_document_ids == {900}
    assert _warnings(caplog) == []


def test_a_drafting_turn_is_searched_whole_under_the_drafting_header(monkeypatch, caplog):
    svc, searched = _chat(monkeypatch, answers=((NOVEMBER,),))

    with caplog.at_level(logging.WARNING):
        frames, llm_messages, prefetched = _turn(svc, DRAFTING)

    assert searched == [_searched_as(DRAFTING)]
    assert _masked(frames) == _frames_for(
        DRAFTING, "1 passage from harbourline-wholesale-terms-2026-11.md — searched automatically")
    assert prefetched == [("search_knowledge", {"query": DRAFTING, "limit": 5})]
    assert llm_messages == [IDENTITY, {
        "role": "system",
        "content": f"{kp.DRAFTING_HEADER}\n\n<passages: harbourline-wholesale-terms-2026-11.md>"}]
    assert svc._turn_document_ids == {731}
    assert _warnings(caplog) == []


@pytest.mark.parametrize("message, chat", [
    (INSTRUCTION, {}),                       # not a question, and drafts nothing
    (ASK, {"dial_on": False}),               # the dial off
    (ASK, {"documents": 0}),                 # no documents in the workspace
])
def test_a_turn_that_does_not_qualify_is_not_searched(monkeypatch, caplog, message, chat):
    svc, searched = _chat(monkeypatch, **chat)

    with caplog.at_level(logging.WARNING):
        frames, llm_messages, prefetched = _turn(svc, message)

    assert _untouched(svc, frames, llm_messages, prefetched)
    assert searched == [] and svc._turn_document_ids == set()
    assert _warnings(caplog) == []


def test_a_count_that_cannot_be_read_leaves_the_turn_as_it_was(monkeypatch, caplog):
    svc, searched = _chat(monkeypatch, documents=RuntimeError("the count failed"))

    with caplog.at_level(logging.WARNING):
        frames, llm_messages, prefetched = _turn(svc, ASK)

    assert _untouched(svc, frames, llm_messages, prefetched)
    assert searched == [] and svc._turn_document_ids == set()
    assert _warnings(caplog) == ["[F085] retrieval first skipped: could not count documents"]


def test_a_first_search_that_fails_leaves_the_turn_as_it_was(monkeypatch, caplog):
    svc, searched = _chat(monkeypatch, answers=(RuntimeError("embedding timed out"),))

    with caplog.at_level(logging.WARNING):
        frames, llm_messages, prefetched = _turn(svc, ASK)

    assert _untouched(svc, frames, llm_messages, prefetched)
    assert searched == [_searched_as(ASK)] and svc._turn_document_ids == set()
    assert _warnings(caplog) == ["[F085] retrieval first failed"]


def test_a_second_question_whose_search_fails_keeps_the_first_search_on_the_record(monkeypatch, caplog):
    svc, searched = _chat(monkeypatch, answers=((HIT,), RuntimeError("embedding timed out")))
    both = f"{ASK} {ASK_AGAIN}"

    with caplog.at_level(logging.WARNING):
        frames, llm_messages, prefetched = _turn(svc, both)

    assert _untouched(svc, frames, llm_messages, prefetched)
    assert [(call["tool_args"], call["original_intent"]) for call in searched] == [
        ({"query": ASK, "limit": 2}, both), ({"query": ASK_AGAIN, "limit": 2}, both)]
    assert svc._turn_document_ids == {720}                    # the first question's search was recorded
    assert _warnings(caplog) == ["[F085] retrieval first failed"]


def test_rules_that_fail_after_a_search_keep_the_search_on_the_record(monkeypatch, caplog):
    async def _broken(*args, **kwargs):
        raise RuntimeError("the owner's rules could not be read")

    svc, searched = _chat(monkeypatch, answers=((HIT,),))
    monkeypatch.setattr(kp, "_rules_first", _broken)

    with caplog.at_level(logging.WARNING):
        frames, llm_messages, prefetched = _turn(svc, ASK)

    assert _untouched(svc, frames, llm_messages, prefetched)
    assert searched == [_searched_as(ASK)]
    assert svc._turn_document_ids == {720}                    # recorded before the rules failed
    assert _warnings(caplog) == ["[F085] retrieval first skipped"]
