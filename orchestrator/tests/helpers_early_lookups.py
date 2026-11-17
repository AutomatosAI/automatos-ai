"""Voice latency D-2: the fakes and the turn stand-ins of ``test_a_turn_starts_its_lookups_early``.

``World`` fakes both of a turn's late lookups and counts what ran: the Composio services
behind a fake ``composio_lookup``, the document search, the count, the dial, the sessions the
early prefetch opens for itself, and the owner's-documents filter (which says where it ran).
The service is a bare ``StreamingChatService.__new__`` object, as the F105 and F085 tests
build one, and the late steps go through its real ``_inject_composio_tools`` and
``_retrieval_first``.

No SDK, no network, no database.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import re
import threading
import time
from types import SimpleNamespace as NS

from config import config
from consumers.chatbot import early_lookups as el
from consumers.chatbot import knowledge_prefetch as kp
from modules.tools.services.composio_hint_service import ComposioHintResult
from modules.tools.services.composio_tool_service import ComposioToolResult

WS = "ws"
AGENT = 7
CHAT_ID = "chat-1"
ASK = "Which word does the brand voice guide ban?"
SEARCH_STEP = "chat turn (agent 7): tool search"
HINTS_STEP = "chat turn (agent 7): action hints"
SEARCH_ARGS = {"query": ASK, "limit": 5}
COMPOSIO_EXECUTE = {"type": "function", "function": {
    "name": "composio_execute",
    "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
}}
LIST_AGENTS = {"type": "function", "function": {
    "name": "platform_list_agents", "parameters": {"type": "object", "properties": {}},
}}
SEND_EMAIL = {"type": "function", "function": {
    "name": "GMAIL_SEND_EMAIL", "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
}}
TOOLS = [COMPOSIO_EXECUTE, LIST_AGENTS]
MESSAGES = [{"role": "user", "content": ASK}]
LLM_MESSAGES = [{"role": "system", "content": "identity"}, {"role": "system", "content": "context"},
                {"role": "user", "content": ASK}]
HIT = {"filename": "brand-voice.md", "source": "brand-voice.md", "title": "brand-voice.md", "similarity": 0.62,
       "content": "Never write 'moreish'.", "excerpt": "Never write 'moreish'.", "document_id": 720}
RUNTIME = NS(agent_id=294, metadata=NS(name="Auto", agent_type="assistant", description="", skills=[]))
CALL_ID = re.compile(r"^retrieval-first-[0-9a-f]{12}$")
HERE = "here"        # where a fake ran: on the test's own thread (the loop's)...
THREAD = "thread"    # ...or on another


class OwnSession:
    """A session an early read opens for itself: it only knows whether it was closed."""

    closed = False

    def close(self):
        self.closed = True


class TurnSession:
    """Stands in for the turn's own session: every read asked of it is noted and refused, so
    the steps around the prefetch read nothing (as on the ``object()`` the F085 tests use)."""

    def __init__(self):
        self.asked = []

    def __getattr__(self, name):
        self.asked.append(name)
        raise AttributeError(name)


def next_answer(scripted, otherwise):
    """The next scripted answer (an exception is raised), or ``otherwise`` when none is left."""
    answer = scripted.pop(0) if scripted else otherwise
    if isinstance(answer, BaseException):
        raise answer
    return answer


def search_result(*passages):
    return {"success": True, "raw_result": {"success": True, "results": list(passages)}, "frontend_data": {}}


def tools_found():
    return ComposioToolResult(tools=[SEND_EMAIL], action_set={"GMAIL_SEND_EMAIL"}, app_names=["gmail"],
                              strategy="sdk_search", search_ms=12)


class World:
    """Both lookups, faked and counted. ``found`` and ``hints`` are what the Composio lookups
    answer, ``passages`` what a search finds, ``documents`` the count, ``dial`` the prefetch
    dial, ``room`` what the pool guard says. ``tool_answers``, ``search_answers`` and
    ``count_answers`` script the next calls (an exception is raised); ``on_lookup`` and
    ``on_search`` are awaited inside each fake. What ran is in ``lookups`` (steps), ``asked``
    (the tool search's arguments), ``searches``, ``counts`` and ``filtered`` (where each ran),
    and ``sessions`` (every session the early prefetch opened for itself)."""

    def __init__(self):
        self.found, self.hints, self.passages = ComposioToolResult(), ComposioHintResult(), (HIT,)
        self.documents, self.dial, self.room = 2, True, True
        self.tool_answers, self.search_answers, self.count_answers = [], [], []
        self.on_lookup = self.on_search = None
        self.lookups, self.asked, self.searches = [], [], []
        self.counts, self.filtered, self.sessions = [], [], []
        self.thread = threading.get_ident()

    def where(self):
        return HERE if threading.get_ident() == self.thread else THREAD

    def forget(self):
        """Empty the counters (after a first run whose calls are not the test's subject)."""
        for ran in (self.lookups, self.asked, self.searches, self.counts, self.filtered, self.sessions):
            ran.clear()

    async def lookup(self, read, *, step):
        """Stands in for ``core.composio.off_loop.composio_lookup``: no thread, no session."""
        self.lookups.append(step)
        if self.on_lookup is not None:
            await self.on_lookup()
        return read(None)

    async def execute_and_format(self, **kw):
        self.searches.append(kw)
        if self.on_search is not None:
            await self.on_search()
        return search_result(*next_answer(self.search_answers, self.passages))

    def count(self, db, workspace_id):
        self.counts.append(self.where())
        return next_answer(self.count_answers, self.documents)

    def owners_own(self, db, result, workspace_id=None):
        self.filtered.append((self.where(), db))
        return result

    def open_session(self):
        self.sessions.append(OwnSession())
        return self.sessions[-1]

    def tool_service(self):
        world = self

        class _ToolService:
            def __init__(self, db):
                pass

            def get_tools_for_step(self, **kwargs):
                world.asked.append(kwargs)
                return next_answer(world.tool_answers, world.found)

        return _ToolService

    def hint_service(self):
        world = self

        class _HintService:
            def __init__(self, db):
                pass

            def build_hints(self, **kwargs):
                return world.hints

        return _HintService

    def install(self, monkeypatch):
        """Put the fakes in place, with ``CHAT_EARLY_LOOKUPS`` on."""
        monkeypatch.setattr("core.composio.off_loop.composio_lookup", self.lookup)
        monkeypatch.setattr("modules.tools.services.composio_tool_service.ComposioToolService", self.tool_service())
        monkeypatch.setattr("modules.tools.services.composio_hint_service.ComposioHintService", self.hint_service())
        monkeypatch.setattr("services.draft_guides.owners_own", self.owners_own)
        monkeypatch.setattr(kp, "documents_in", self.count)
        monkeypatch.setattr(kp, "_has_database", lambda db, ws: False)
        monkeypatch.setattr(kp, "_passages", lambda kept: "<passages: " + ", ".join(p["filename"] for p in kept) + ">")
        monkeypatch.setattr(type(config), "CHATBOT_KNOWLEDGE_PREFETCH", property(lambda _config: self.dial))
        monkeypatch.setattr(config, "KNOWLEDGE_PREFETCH_PASSAGES", 5, raising=False)
        monkeypatch.setattr(config, "KNOWLEDGE_PREFETCH_MIN_SCORE", 0.3, raising=False)
        monkeypatch.setattr(config, "CHAT_EARLY_LOOKUPS", True)
        monkeypatch.setattr(el, "_a_session", self.open_session)
        monkeypatch.setattr(el, "pool_has_room", lambda: self.room)
        return self


def a_chat(world):
    """A bare service over the fakes."""
    from consumers.chatbot import service
    from consumers.chatbot.prompt_analyzer import get_prompt_analyzer
    from consumers.chatbot.streaming import get_streaming_handler

    chat = service.StreamingChatService.__new__(service.StreamingChatService)
    chat.db, chat.workspace_id, chat.widget_mode = TurnSession(), WS, False
    chat.streaming_handler, chat.prompt_analyzer = get_streaming_handler(), get_prompt_analyzer()
    chat._turn_document_ids, chat._turn_chunk_ids = set(), set()
    chat.tool_router = NS(execute_and_format=world.execute_and_format)
    return chat


def an_ask(text=ASK, **given):
    """The turn's arguments, as the turn wrapper keeps them."""
    ask = el.TurnAsk(chat_id=CHAT_ID, messages=[{"role": "user", "content": text}], agent_id=AGENT,
                     skip_composio=False, complexity_assessment=None, force_text_only=False)
    return dataclasses.replace(ask, **given)


def masked(frames):
    """The frames as data, the call id and the search's duration masked (both are checked first)."""
    events = [json.loads(frame[len("d:"):]) for frame in frames]
    assert all(frame.startswith("d:") and frame.endswith("\n") for frame in frames)
    for event in events:
        assert CALL_ID.match(event["data"]["toolCallId"])
        event["data"]["toolCallId"] = "<call id>"
        if "durationMs" in event["data"]:
            assert isinstance(event["data"]["durationMs"], int) and event["data"]["durationMs"] >= 0
            event["data"]["durationMs"] = "<ms>"
    return events


async def late_steps(chat, *, assessment=None):
    """The turn's two late steps, through the real methods: everything the model would be
    given, the frames, and the turn's retrieval record."""
    messages, prefetched = copy.deepcopy(LLM_MESSAGES), []
    tools, found = await chat._inject_composio_tools(messages, list(TOOLS), ASK, AGENT, RUNTIME, False, assessment)
    frames = [frame async for frame in chat._retrieval_first(ASK, messages, RUNTIME, CHAT_ID, prefetched)]
    return {"messages": messages, "tools": tools, "found": found, "prefetched": prefetched,
            "frames": masked(frames), "recorded": set(chat._turn_document_ids)}


def a_turn(world, *, early, ask=None, between=None, **late):
    """One turn's late steps on a new bare service. ``early``: its lookups start first, as they
    do once the agent is active, and ``between`` (the turn's tools and prompt) runs before the
    late steps. Returns the service, what the late steps gave, and what became of the early
    apps lookup and of the early prefetch."""
    chat = a_chat(world)

    async def go():
        if not early:
            return await late_steps(chat, **late), None
        chat._turn_ask = ask or an_ask()
        chat._early = el.start(chat, chat._turn_ask, RUNTIME)
        await asyncio.sleep(0)
        try:
            if between is not None:
                await between(chat)
            given = await late_steps(chat, **late)
            return given, (chat._early.apps_use, chat._early.knowledge_use)
        finally:
            await el._drop(chat)

    given, uses = asyncio.run(go())
    return chat, given, uses


def given_in_place(world, **late):
    """What the same late steps give with nothing started early (the switch-off run); the
    fakes' counters are emptied again afterwards."""
    _, given, _ = a_turn(world, early=False, **late)
    world.forget()
    return given


async def the_early_lookups_end(chat):
    """The turn's tools and prompt take long enough for every early lookup to end."""
    await asyncio.wait([task for task in (chat._early.apps, chat._early.knowledge) if task is not None])


async def until(happened, within=5.0):
    """Wait, a few milliseconds at a time, until ``happened()``; fail after ``within`` seconds."""
    deadline = time.monotonic() + within
    while not happened():
        assert time.monotonic() < deadline, "what the test waits for never happened"
        await asyncio.sleep(0.005)


def only_the_first_waits():
    """A hook for a fake: its first call waits until it is cancelled; later calls go straight on."""
    calls = []

    async def hook():
        calls.append(None)
        if len(calls) == 1:
            await asyncio.Event().wait()

    return hook


def said(caplog, level=logging.DEBUG):
    return [record.getMessage() for record in caplog.records if record.levelno >= level]


async def drain(frames):
    return [frame async for frame in frames]


@el.once_the_agent_is_active
async def load_agent_context(chat, agent_runtime):
    """Stands in for ``_load_agent_context``, under the real wrapper."""
    return {"persona": ""}


def a_wrapped_turn(then):
    """A stand-in for the turn with the real one's argument names, under the real wrapper: it
    sends the chat id, loads its agent's context (the agent is active), then awaits ``then(chat)``."""
    @el.starts_its_lookups_early
    async def turn(chat, chat_id, messages, agent_id, user_id, use_orchestrator_llm=False, skip_composio=False,
                   complexity_assessment=None, force_text_only=False):
        yield "chat-id"
        await load_agent_context(chat, RUNTIME)
        await then(chat)
        yield "answered"

    return turn
