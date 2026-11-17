"""Voice latency D-1: a turn's two late lookups, each split from its merge.

``test_a_turns_late_lookups_keep_their_shape.py`` pins what the two methods hand the turn,
before and after the move. This file is about the pieces the move made:

* ``app_lookup.look_up_apps`` only reads: the tool search, then the action hints when the
  search offers no tools. A lookup that gives up is not a failure; one that raises is, and the
  search's result is kept when only the hints failed.
* ``app_lookup.apply_app_lookup`` only merges, into a new tool list, and returns the merged
  list even when the scope message fails after it.
* ``early_lookups.run_prefetch`` hands back every search's result and records nothing on the
  turn; it never raises, also when ``prefetch`` does.
* ``early_lookups.the_turns_prefetch`` records each result on the turn once: also when
  nothing was found, and when it then raises.

DB-free: no database, model, SDK or socket.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace as NS

import pytest

from config import config
from consumers.chatbot import app_lookup, early_lookups
from consumers.chatbot import knowledge_prefetch as kp
from consumers.chatbot.app_lookup import AppLookup
from core.composio.off_loop import ComposioLookupTimeout
from modules.tools.services.composio_hint_service import ComposioHintResult
from modules.tools.services.composio_tool_service import ComposioToolResult

WS = "6a0f3c1e-2b4d-4e8f-9a1b-c20000000001"
AGENT = 11
ASKED = "Email John the notes"
SEARCH_STEP = "chat turn (agent 11): tool search"
HINTS_STEP = "chat turn (agent 11): action hints"
FAILED_LINE = "Composio tool injection failed for agent 11: "
COMPOSIO_EXECUTE = {"type": "function", "function": {
    "name": "composio_execute",
    "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
}}
LIST_AGENTS = {"type": "function", "function": {
    "name": "platform_list_agents", "parameters": {"type": "object", "properties": {}},
}}
SEARCH_KNOWLEDGE = {"type": "function", "function": {
    "name": "search_knowledge", "parameters": {"type": "object", "properties": {}},
}}
SEND_EMAIL = {"type": "function", "function": {
    "name": "GMAIL_SEND_EMAIL", "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
}}
MESSAGES = [{"role": "system", "content": "identity"}, {"role": "system", "content": "context"},
            {"role": "user", "content": ASKED}]
HINT_LINES = ["Use GMAIL_SEND_EMAIL.", "Its params: to, subject, body."]


def _tools_found():
    return ComposioToolResult(tools=[SEND_EMAIL], action_set={"GMAIL_SEND_EMAIL"}, app_names=["gmail"],
                              strategy="sdk_search", search_ms=12)


def _warned(caplog, logger_name):
    return [record.getMessage() for record in caplog.records
            if record.name == logger_name and record.levelno >= logging.WARNING]


def _answer(what):
    if isinstance(what, BaseException):
        raise what
    return what


# --------------------------------------------------------------------------- look_up_apps

@pytest.fixture
def composio(monkeypatch):
    """A fake ``composio_lookup`` (no thread, no session) over fake services.
    ``composio.answers`` maps a step to its answer (an exception is raised);
    ``composio.steps`` is every lookup made, in order; ``composio.asked`` its arguments."""
    script = NS(answers={SEARCH_STEP: ComposioToolResult(), HINTS_STEP: ComposioHintResult()}, steps=[], asked=[])

    class _ToolService:
        def __init__(self, db):
            pass

        def get_tools_for_step(self, **kwargs):
            script.asked.append(kwargs)
            return _answer(script.answers[SEARCH_STEP])

    class _HintService:
        def __init__(self, db):
            pass

        def build_hints(self, **kwargs):
            script.asked.append(kwargs)
            return _answer(script.answers[HINTS_STEP])

    async def _lookup(read, *, step):
        script.steps.append(step)
        return read(None)

    monkeypatch.setattr("core.composio.off_loop.composio_lookup", _lookup)
    monkeypatch.setattr("modules.tools.services.composio_tool_service.ComposioToolService", _ToolService)
    monkeypatch.setattr("modules.tools.services.composio_hint_service.ComposioHintService", _HintService)
    return script


def _look_up(tool_hints=()):
    return asyncio.run(app_lookup.look_up_apps(AGENT, WS, ASKED, list(tool_hints)))


def test_tools_found_are_handed_back_and_the_hints_are_not_asked(composio):
    composio.answers[SEARCH_STEP] = _tools_found()

    lookup = _look_up(["gmail", "send email"])

    assert lookup == AppLookup(found=composio.answers[SEARCH_STEP])
    assert lookup.found is composio.answers[SEARCH_STEP]
    assert (lookup.hints, lookup.failed) == (None, False)
    assert composio.steps == [SEARCH_STEP]
    assert composio.asked == [{"agent_id": AGENT, "workspace_id": WS, "task_prompt": "gmail send email",
                               "tool_hints": ["gmail", "send email"]}]


def test_no_tools_asks_the_hints_next(composio):
    composio.answers[HINTS_STEP] = ComposioHintResult(hint_lines=list(HINT_LINES))

    lookup = _look_up()

    assert lookup.found is composio.answers[SEARCH_STEP]
    assert lookup.hints is composio.answers[HINTS_STEP]
    assert lookup.failed is False
    assert composio.steps == [SEARCH_STEP, HINTS_STEP]
    assert composio.asked == [
        {"agent_id": AGENT, "workspace_id": WS, "task_prompt": ASKED, "tool_hints": []},   # no hints: the message
        {"agent_id": AGENT, "prompt": ASKED, "workspace_id": WS},
    ]


def test_a_search_that_gives_up_is_no_failure_and_the_hints_are_not_asked(composio, caplog):
    composio.answers[SEARCH_STEP] = ComposioLookupTimeout("no answer from Composio")

    with caplog.at_level(logging.WARNING):
        lookup = _look_up()

    assert lookup == AppLookup()
    assert (lookup.found, lookup.hints, lookup.failed) == (None, None, False)
    assert composio.steps == [SEARCH_STEP]
    assert _warned(caplog, app_lookup.logger.name) == []      # warned where it gave up, not here


def test_hints_that_give_up_keep_the_searchs_result_and_are_no_failure(composio):
    composio.answers[HINTS_STEP] = ComposioLookupTimeout("no answer from Composio")

    lookup = _look_up()

    assert lookup.found is composio.answers[SEARCH_STEP]
    assert (lookup.hints, lookup.failed) == (None, False)


def test_a_failed_hint_lookup_keeps_the_searchs_result_and_says_it_failed(composio, caplog):
    composio.answers[HINTS_STEP] = RuntimeError("the hint lookup broke")

    with caplog.at_level(logging.WARNING):
        lookup = _look_up()

    assert lookup.found is composio.answers[SEARCH_STEP]
    assert (lookup.hints, lookup.failed) == (None, True)
    assert composio.steps == [SEARCH_STEP, HINTS_STEP]
    assert _warned(caplog, app_lookup.logger.name) == [FAILED_LINE + "the hint lookup broke"]


def test_a_failed_search_finds_nothing_and_says_it_failed(composio, caplog):
    composio.answers[SEARCH_STEP] = RuntimeError("the search broke")

    with caplog.at_level(logging.WARNING):
        lookup = _look_up()

    assert lookup == AppLookup(failed=True)
    assert composio.steps == [SEARCH_STEP]
    assert _warned(caplog, app_lookup.logger.name) == [FAILED_LINE + "the search broke"]


def test_the_turns_app_lookup_asks_for_the_turns_workspace(composio):
    composio.answers[SEARCH_STEP] = _tools_found()

    lookup = asyncio.run(early_lookups.the_turns_app_lookup(NS(workspace_id=WS), AGENT, ASKED, []))

    assert lookup.found is composio.answers[SEARCH_STEP]
    assert composio.asked[0]["workspace_id"] == WS


@pytest.mark.parametrize("assessment, hints", [
    (None, []),
    (NS(), []),                                         # an assessment that carries no hints
    (NS(tool_hints=["gmail"]), ["gmail"]),
])
def test_the_tool_hints_are_the_assessments_or_none(assessment, hints):
    assert app_lookup.tool_hints_of(assessment) == hints


@pytest.mark.parametrize("turn, asks", [
    ((ASKED, AGENT, WS, False), True),
    (("", AGENT, WS, False), False),                    # nothing said
    ((ASKED, None, WS, False), False),                  # no agent
    ((ASKED, AGENT, None, False), False),               # no workspace
    ((ASKED, AGENT, WS, True), False),                  # the turn skips its apps
])
def test_which_turns_ask_the_apps(turn, asks):
    assert app_lookup.asks_the_apps(*turn) is asks


# --------------------------------------------------------------------------- apply_app_lookup

def test_the_typed_tools_replace_the_generic_one_and_every_other_tool_keeps_its_place():
    given, messages = [LIST_AGENTS, COMPOSIO_EXECUTE, SEARCH_KNOWLEDGE], copy.deepcopy(MESSAGES)
    kept = list(given)

    tools = app_lookup.apply_app_lookup(messages, given, AppLookup(found=_tools_found()), AGENT)

    assert tools == [LIST_AGENTS, SEARCH_KNOWLEDGE, SEND_EMAIL]
    assert tools is not given and given == kept               # the list given is not changed
    assert [m["role"] for m in messages] == ["system", "system", "system", "user"]
    assert "gmail" in messages[2]["content"] and "do NOT use composio_execute" in messages[2]["content"]
    assert messages[:2] + messages[3:] == MESSAGES


def test_hint_lines_go_third_and_the_tools_stay():
    given, messages = [LIST_AGENTS, COMPOSIO_EXECUTE], copy.deepcopy(MESSAGES)
    lookup = AppLookup(found=ComposioToolResult(), hints=ComposioHintResult(hint_lines=list(HINT_LINES)))

    tools = app_lookup.apply_app_lookup(messages, given, lookup, AGENT)

    assert tools is given
    assert messages == [MESSAGES[0], MESSAGES[1],
                        {"role": "system", "content": "Use GMAIL_SEND_EMAIL.\nIts params: to, subject, body."},
                        MESSAGES[2]]


@pytest.mark.parametrize("lookup", [
    AppLookup(),                                                            # gave up
    AppLookup(failed=True),                                                 # the search raised
    AppLookup(found=ComposioToolResult(), hints=ComposioHintResult()),      # nothing to offer, nothing to hint
    AppLookup(found=ComposioToolResult(), failed=True),                     # the hints raised
])
def test_a_lookup_with_nothing_to_merge_leaves_the_turn_as_it_was(lookup):
    given, messages = [LIST_AGENTS, COMPOSIO_EXECUTE], copy.deepcopy(MESSAGES)

    tools = app_lookup.apply_app_lookup(messages, given, lookup, AGENT)

    assert tools is given and given == [LIST_AGENTS, COMPOSIO_EXECUTE]
    assert messages == MESSAGES


def test_a_scope_message_that_fails_still_returns_the_merged_tools(monkeypatch, caplog):
    def _broken(app_names):
        raise RuntimeError("the scope message broke")

    monkeypatch.setattr("api.recipe_executor._composio_scope_message", _broken)
    given, messages = [LIST_AGENTS, COMPOSIO_EXECUTE], copy.deepcopy(MESSAGES)

    with caplog.at_level(logging.WARNING):
        tools = app_lookup.apply_app_lookup(messages, given, AppLookup(found=_tools_found()), AGENT)

    assert tools == [LIST_AGENTS, SEND_EMAIL]
    assert given == [LIST_AGENTS, COMPOSIO_EXECUTE]
    assert messages == MESSAGES
    assert _warned(caplog, app_lookup.logger.name) == [FAILED_LINE + "the scope message broke"]


# --------------------------------------------------------------------------- run_prefetch, the_turns_prefetch

QUESTION = "Which word does the brand voice guide ban?"
HIT = {"filename": "brand-voice.md", "source": "brand-voice.md", "title": "brand-voice.md", "similarity": 0.62,
       "content": "Never write 'moreish'.", "excerpt": "Never write 'moreish'.", "document_id": 720}
SEARCH_ARGS = {"query": QUESTION, "limit": 5}
THE_TURNS_SESSION = "the turn's session"
ANOTHER_SESSION = "a session of the lookup's own"
RUNTIME = NS(agent_id=294)
CHAT_ID = "chat-1"


def _result(*passages):
    return {"success": True, "raw_result": {"success": True, "results": list(passages)}, "frontend_data": {}}


def _chat(monkeypatch, *, search=None):
    """A bare service over a fake ``execute_and_format`` (``search`` is its answer; an exception
    is raised). The count, the dial and the two renderers outside this story are faked."""
    from consumers.chatbot import service
    from consumers.chatbot.streaming import get_streaming_handler

    searched = []
    answer = _result(HIT) if search is None else search

    async def execute_and_format(**kw):
        searched.append(kw)
        return _answer(answer)

    monkeypatch.setattr(kp, "documents_in", lambda db, ws: 2)
    monkeypatch.setattr(kp, "_has_database", lambda db, ws: False)
    monkeypatch.setattr(kp, "_passages", lambda kept: "<passages>")
    monkeypatch.setattr(type(config), "CHATBOT_KNOWLEDGE_PREFETCH", property(lambda self: True))
    monkeypatch.setattr(config, "KNOWLEDGE_PREFETCH_PASSAGES", 5, raising=False)
    monkeypatch.setattr(config, "KNOWLEDGE_PREFETCH_MIN_SCORE", 0.3, raising=False)
    svc = service.StreamingChatService.__new__(service.StreamingChatService)
    svc.db, svc.workspace_id, svc.streaming_handler = THE_TURNS_SESSION, "ws", get_streaming_handler()
    svc._turn_document_ids, svc._turn_chunk_ids = set(), set()
    svc.tool_router = NS(execute_and_format=execute_and_format)
    return svc, searched


def _prefetch_that(after_one_search):
    """A ``prefetch`` that runs one search and then does ``after_one_search(result)``."""
    seen = {}

    async def prefetch(db, workspace_id, message, *, search, **kwargs):
        seen.update(kwargs, db=db, workspace_id=workspace_id, message=message)
        return after_one_search(await search(dict(SEARCH_ARGS)))

    return prefetch, seen


def _raise(error):
    raise error


def test_run_prefetch_hands_back_what_was_found_and_records_nothing_on_the_turn(monkeypatch):
    svc, searched = _chat(monkeypatch)

    got = asyncio.run(early_lookups.run_prefetch(svc, ANOTHER_SESSION, QUESTION, RUNTIME, CHAT_ID, {}))

    assert isinstance(got.found, kp.Prefetch) and got.found.passages == 1
    assert got.results == (_result(HIT),)
    assert (got.failed, got.error) == (False, None)
    assert svc._turn_document_ids == set()                    # recording is the caller's
    assert searched == [{
        "tool_name": "search_knowledge", "tool_args": SEARCH_ARGS, "agent_id": 294, "workspace_id": "ws",
        "original_intent": QUESTION,
        "caller_context": {"user_query": QUESTION, "conversation_id": CHAT_ID, "retrieval_first": True},
    }]


def test_run_prefetch_reads_on_the_session_it_is_given(monkeypatch):
    svc, _ = _chat(monkeypatch)
    prefetch, seen = _prefetch_that(lambda result: None)
    filtered_on = []
    monkeypatch.setattr(kp, "prefetch", prefetch)
    monkeypatch.setattr("services.draft_guides.owners_own",
                        lambda db, result, workspace_id=None: filtered_on.append((db, workspace_id)) or result)
    svc._viewer_subject_id = "user:7"

    got = asyncio.run(early_lookups.run_prefetch(
        svc, ANOTHER_SESSION, QUESTION, RUNTIME, CHAT_ID, {"question_only": False, "drafting": True}))

    assert got == early_lookups.Searched(None, (_result(HIT),))
    assert filtered_on == [(ANOTHER_SESSION, "ws")]           # never the turn's own session
    assert seen == {"db": ANOTHER_SESSION, "workspace_id": "ws", "message": QUESTION, "viewer": "user:7",
                    "enabled": True, "question_only": False, "drafting": True, "limit": 5, "min_score": 0.3}


def test_run_prefetch_returns_what_was_searched_when_prefetch_raises(monkeypatch):
    svc, _ = _chat(monkeypatch)
    broke = RuntimeError("the owner's rules could not be read")
    prefetch, _seen = _prefetch_that(lambda result: _raise(broke))
    monkeypatch.setattr(kp, "prefetch", prefetch)

    got = asyncio.run(early_lookups.run_prefetch(svc, THE_TURNS_SESSION, QUESTION, RUNTIME, CHAT_ID, {}))

    assert got == early_lookups.Searched(found=None, results=(_result(HIT),), failed=True, error=broke)
    assert got.error is broke


def test_run_prefetch_says_a_failed_search_failed_and_raises_nothing(monkeypatch, caplog):
    svc, searched = _chat(monkeypatch, search=RuntimeError("embedding timed out"))

    with caplog.at_level(logging.WARNING):
        got = asyncio.run(early_lookups.run_prefetch(svc, THE_TURNS_SESSION, QUESTION, RUNTIME, CHAT_ID, {}))

    assert got == early_lookups.Searched(found=None, results=(), failed=True, error=None)
    assert len(searched) == 1
    assert _warned(caplog, kp.logger.name) == ["[F085] retrieval first failed"]   # prefetch's own line, as before


def test_run_prefetch_puts_each_result_in_the_callers_list_as_it_arrives(monkeypatch):
    svc, _ = _chat(monkeypatch)
    arrived = []
    prefetch, _seen = _prefetch_that(lambda result: arrived.append(list(into)))
    into = []
    monkeypatch.setattr(kp, "prefetch", prefetch)

    got = asyncio.run(early_lookups.run_prefetch(svc, THE_TURNS_SESSION, QUESTION, RUNTIME, CHAT_ID, {}, into=into))

    assert arrived == [[_result(HIT)]]                        # there before prefetch went on
    assert into == [_result(HIT)] and got.results == (_result(HIT),)


def test_the_turns_prefetch_searches_on_the_turns_session_and_records_each_result_once(monkeypatch):
    svc, _ = _chat(monkeypatch)
    recorded = []
    collect = svc._collect_tool_retrieval
    svc._collect_tool_retrieval = lambda name, result: recorded.append(name) or collect(name, result)
    prefetch, seen = _prefetch_that(lambda result: None)       # nothing found
    monkeypatch.setattr(kp, "prefetch", prefetch)

    found = asyncio.run(early_lookups.the_turns_prefetch(svc, QUESTION, RUNTIME, CHAT_ID, {}))

    assert found is None
    assert seen["db"] == THE_TURNS_SESSION
    assert recorded == ["search_knowledge"]
    assert svc._turn_document_ids == {720}


def test_the_turns_prefetch_hands_back_what_prefetch_found(monkeypatch):
    svc, _ = _chat(monkeypatch)

    found = asyncio.run(early_lookups.the_turns_prefetch(svc, QUESTION, RUNTIME, CHAT_ID, {}))

    assert isinstance(found, kp.Prefetch)
    assert (found.args, found.passages, found.files) == (SEARCH_ARGS, 1, ["brand-voice.md"])
    assert svc._turn_document_ids == {720}


def test_the_turns_prefetch_records_the_search_and_then_raises_what_prefetch_raised(monkeypatch):
    svc, _ = _chat(monkeypatch)
    broke = RuntimeError("the owner's rules could not be read")
    prefetch, _seen = _prefetch_that(lambda result: _raise(broke))
    monkeypatch.setattr(kp, "prefetch", prefetch)

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(early_lookups.the_turns_prefetch(svc, QUESTION, RUNTIME, CHAT_ID, {}))

    assert raised.value is broke
    assert svc._turn_document_ids == {720}


def test_a_cancelled_prefetch_still_records_what_it_had_searched(monkeypatch):
    svc, _ = _chat(monkeypatch)
    prefetch, _seen = _prefetch_that(lambda result: _raise(asyncio.CancelledError()))
    monkeypatch.setattr(kp, "prefetch", prefetch)

    async def turn():
        with pytest.raises(asyncio.CancelledError):
            await early_lookups.the_turns_prefetch(svc, QUESTION, RUNTIME, CHAT_ID, {})

    asyncio.run(turn())

    assert svc._turn_document_ids == {720}


# --------------------------------------------------------------------------- the two methods keep only the merge

def _innermost(method):
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    return method


def test_the_two_methods_ask_this_module_for_their_lookup():
    from consumers.chatbot.service import StreamingChatService

    inject = _innermost(StreamingChatService._inject_composio_tools).__code__
    retrieve = _innermost(StreamingChatService._retrieval_first).__code__

    assert "the_turns_app_lookup" in inject.co_names + inject.co_varnames
    assert "composio_lookup" not in inject.co_names + inject.co_varnames
    assert "the_turns_prefetch" in retrieve.co_names + retrieve.co_varnames
    assert "prefetch" not in retrieve.co_names + retrieve.co_varnames
