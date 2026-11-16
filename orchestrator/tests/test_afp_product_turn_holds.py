"""PRD-274 §5.5: what a product's turn (a server key's own, or an end user's) does once its writes are held.

- A tool that stores memory says nothing was saved when its write was held, never "Stored" (F103).
- A product's turn makes no on-demand checkpoint: no distil, no ``chats.summary`` write.
- The idle sweep checkpoints an end user's conversation as their turn, into their own namespace, recalled for
  them only. A key's own conversation and a widget visitor's it never reads (tests/test_afp_keys_own_turn...).
- A key's own turn captures no tool outcome, and marks none as seen.
- A product's turn still knows what the workspace can look up.
- A turn that ends early, closed or by an error, leaves no hold behind it.
- Every memory write a turn can make reaches no store on a key's own turn; on a person's, each one does.
- A Playbook run started in a product's turn runs under its hold. No real model, provider or database.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from consumers.chatbot.lean_request import LeanRequest
from core.security.end_user_turn import (EndUser, current_end_user, end_user_turn, keys_own_turn,
                                         on_a_keys_own_turn)
from modules.memory.durable_store import DurableMemoryStore
from modules.memory.unified_memory_service import MemoryNamespace, UnifiedMemoryService
from modules.memory.write_contract import DEFAULT_FACT_TYPE

WS = "44444444-4444-4444-4444-444444444444"
LEARNER, OTHER = EndUser(WS, "learner-7", "record-1"), EndUser(WS, "learner-8", "record-2")
KEYS_OWN = LeanRequest(server_key=True)
FOR_THE_LEARNER = LeanRequest(server_key=True, end_user=LEARNER)
PERSON = LeanRequest()
NS_ = MemoryNamespace(WS)
FACT = "The tutor's lessons are 40 minutes long."
CHECKPOINT = ('{"topic": "Photosynthesis", "last_summary": "We went over the light reactions.", '
              '"next_step": "Practise the Calvin cycle.", "decisions": ["Revise with worked examples."], '
              '"open_questions": ["Which exam board is it?"]}')
MESSAGES = [{"role": "user" if i % 2 == 0 else "assistant", "parts": [{"type": "text", "text": f"message {i}"}]}
            for i in range(6)]


@pytest.fixture(autouse=True)
def consent(monkeypatch):
    """An end user's consent, as their writes check it before and after: on record unless a test says."""
    answers = []

    async def consented(end_user):
        return answers.pop(0) if answers else True

    monkeypatch.setattr("core.security.end_user_turn._consented", consented)
    return answers


@contextmanager
def _turn(lean):
    """The scopes ``stream_response_with_agent`` opens for the turn ``lean`` names."""
    with end_user_turn(getattr(lean, "end_user", None)), keys_own_turn(lean):
        yield


async def _settled():
    """Wait for every task the turn started, and the tasks those started in turn."""
    while True:
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


class _Plane:
    """The memory plane on its real code, recorded where each write leaves the process: the durable store's
    ``add`` (at its dedup lookup, which sees the final namespace), an L2 row, the L1 session, a conversation's
    ``chats.summary`` and a flywheel ingest."""

    def __init__(self, monkeypatch):
        self.written = []
        store = object.__new__(DurableMemoryStore)
        store._enabled, store._bootstrap_done = True, True

        async def deduped(user_id, content_hash, metadata):
            self.written.append(("durable", user_id, (metadata or {}).get("end_user_id")))
            return {"success": True, "id": "existing"}

        store._deduped = deduped
        service = object.__new__(UnifiedMemoryService)
        service._durable = store
        service._store_short_term_sync = lambda *row: self.written.append(("l2", row[2])) or "row-1"
        conn = NS(get=lambda key: None, setex=lambda key, ttl, value: self.written.append(("session", key)))
        service._get_redis = lambda: NS(get_redis=lambda: conn)

        async def no_cache(workspace_id):
            return None

        async def nothing_found(**kwargs):
            return []

        service._invalidate_search_cache, service.search_long_term = no_cache, nothing_found
        self.store, self.service = store, service
        monkeypatch.setattr("modules.memory.unified_memory_service.get_unified_memory_service", lambda: service)
        monkeypatch.setattr("config.config.QDRANT_URL", "http://qdrant.test", raising=False)

    def chat(self, chat_id="chat-1"):
        """A conversation whose ``chats.summary`` write is recorded."""
        plane = self

        class _Chat:
            id, user_id = chat_id, 9

            @property
            def summary(self):
                return None

            @summary.setter
            def summary(self, value):
                plane.written.append(("summary", chat_id))

        return _Chat()


def _checkpoint_seams(monkeypatch, chats):
    """``run_thread_checkpoint``'s reads, the conversations ``chats`` names by id and their messages."""
    from modules.memory import thread_checkpoint as tc

    async def distil(prompt, **kwargs):
        return CHECKPOINT

    monkeypatch.setattr(tc, "_load_chat", lambda db, ws, chat_id: chats.get(chat_id))
    monkeypatch.setattr(tc, "_load_messages", lambda db, chat_id: MESSAGES)
    monkeypatch.setattr(tc, "_distill_checkpoint", distil)
    return NS(commit=lambda: None, rollback=lambda: None, close=lambda: None)


# ── 1. a held write is never "Stored" (F103) ─────────────────────────────────

HELD = [(KEYS_OWN, False), (FOR_THE_LEARNER, True)]
HELD_IDS = ["a keys own turn", "an end users turn after their erase started"]


@pytest.mark.parametrize("lean, erasing", HELD, ids=HELD_IDS)
def test_storing_a_memory_on_a_held_turn_says_it_was_not_saved(monkeypatch, consent, lean, erasing):
    from modules.tools.discovery.handlers_workspace import store_memory

    plane = _Plane(monkeypatch)
    if erasing:
        consent.append(False)  # their erase started before the write
    with _turn(lean):
        answer = asyncio.run(store_memory(None, WS, {"content": FACT}))
    assert answer["success"] is False and answer["error"].startswith("Memory NOT saved")
    assert answer["error"].endswith("Tell them it was not stored.") and plane.written == []


def test_storing_a_memory_on_a_persons_turn_still_says_stored(monkeypatch):
    from modules.tools.discovery.handlers_workspace import store_memory

    plane = _Plane(monkeypatch)
    answer = asyncio.run(store_memory(None, WS, {"content": FACT}))
    assert answer["success"] is True and answer["message"].startswith("Stored in memory")
    assert plane.written == [("durable", NS_.workspace(), None)]


@pytest.mark.parametrize("lean, erasing", HELD, ids=HELD_IDS)
def test_the_memory_tool_never_says_created_over_a_held_write(monkeypatch, consent, lean, erasing):
    from modules.memory.memory_tool import DurableMemoryStoreBackend, MemoryToolBackend

    plane = _Plane(monkeypatch)
    deleted = []

    async def delete_long_term(**kwargs):
        deleted.append(kwargs)

    plane.service.delete_long_term = delete_long_term
    if erasing:
        consent.append(False)
    tool = MemoryToolBackend(DurableMemoryStoreBackend(plane.service), workspace_id=WS)
    with _turn(lean):
        said = asyncio.run(tool.handle({"command": "create", "path": "/memories/plan.md", "file_text": "Week 1"}))
    assert said.startswith("Error: memory is not kept on this turn") and said.endswith("nothing was saved")
    assert plane.written == [] and deleted == []


# ── 2. no on-demand checkpoint on a product's turn ───────────────────────────

@pytest.mark.parametrize("lean", [KEYS_OWN, FOR_THE_LEARNER], ids=["a keys own turn", "an end users turn"])
def test_a_products_turn_makes_no_on_demand_checkpoint(monkeypatch, lean):
    from modules.memory.thread_checkpoint import NOT_ON_A_PRODUCTS_TURN
    from modules.tools.discovery.handlers_workspace import checkpoint_thread

    plane = _Plane(monkeypatch)
    asked = []
    db = _checkpoint_seams(monkeypatch, {"another-chat": plane.chat("another-chat")})
    monkeypatch.setattr("modules.memory.thread_checkpoint._load_chat", lambda *a: asked.append(a))
    with _turn(lean):  # the model names another conversation of the workspace
        answer = asyncio.run(checkpoint_thread(db, WS, {"chat_id": "another-chat", "_origin_chat_id": "chat-1"}))
    assert answer == {"success": False, "error": NOT_ON_A_PRODUCTS_TURN}
    assert asked == [] and plane.written == []  # nothing read, no distil, no summary, no memory


def test_a_persons_turn_still_checkpoints_on_demand(monkeypatch):
    from modules.tools.discovery.handlers_workspace import checkpoint_thread

    plane = _Plane(monkeypatch)
    db = _checkpoint_seams(monkeypatch, {"chat-1": plane.chat()})
    with _turn(PERSON):
        answer = asyncio.run(checkpoint_thread(db, WS, {"_origin_chat_id": "chat-1"}))
    assert answer["success"] is True and ("summary", "chat-1") in plane.written


# ── 3. the idle sweep checkpoints an end user's conversation as theirs ───────

def _sweep(monkeypatch, plane, rows, chats, record_id="record-1"):
    from services import memory_jobs

    db = _checkpoint_seams(monkeypatch, chats)
    monkeypatch.setattr(memory_jobs, "_on_own_session", lambda work: work(object()))
    monkeypatch.setattr(memory_jobs, "_checkpoint_candidates", lambda db: rows)
    monkeypatch.setattr("core.models.end_users.consent_record_id", lambda db, ws, end_user_id: record_id)
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: db)
    asyncio.run(memory_jobs.MemoryJobScheduler()._run_thread_checkpoints())


def _recalled_namespaces(store, turn):
    """The namespaces one workspace search reads during ``turn`` (an end user, or None)."""
    seen = []

    async def query(query, flt, limit):
        seen.append(next(c for c in flt.must if c.key == "namespace").match)
        return []

    store._query, store._record_search = query, (lambda *args, **kwargs: None)
    with end_user_turn(turn):
        asyncio.run(store.search("exam board", NS_.workspace(), workspace_id=WS))
    match = seen[0]
    return set(getattr(match, "any", None) or [match.value])


def test_the_sweep_checkpoints_an_end_users_conversation_into_their_namespace(monkeypatch):
    plane = _Plane(monkeypatch)
    theirs, persons = plane.chat("their-chat"), plane.chat("persons-chat")
    _sweep(monkeypatch, plane, [("their-chat", WS, "learner-7"), ("persons-chat", WS, None)],
           {"their-chat": theirs, "persons-chat": persons})
    durable = [w for w in plane.written if w[0] == "durable"]
    assert {(w[1], w[2]) for w in durable} == {(NS_.end_user("learner-7"), "learner-7"),  # theirs, tagged
                                               (NS_.workspace(), None)}                  # the person's, as before
    assert ("summary", "their-chat") in plane.written  # their resume (§5.7) reads it on their conversation
    assert NS_.end_user("learner-7") not in _recalled_namespaces(plane.store, OTHER)  # another end user's recall
    assert NS_.end_user("learner-7") not in _recalled_namespaces(plane.store, None)   # a person's recall
    assert NS_.end_user("learner-7") in _recalled_namespaces(plane.store, LEARNER)    # theirs


def test_the_sweep_checkpoints_nothing_for_an_end_user_with_no_consent_on_record(monkeypatch):
    plane = _Plane(monkeypatch)
    _sweep(monkeypatch, plane, [("their-chat", WS, "learner-7")], {"their-chat": plane.chat("their-chat")},
           record_id=None)
    assert plane.written == []


# ── 4. a key's own turn captures no tool outcome, and marks none as seen ─────

def test_a_keys_own_turn_captures_no_tool_outcome_and_marks_none_seen():
    from modules.memory import tool_outcome_capture as capture

    failed = {"tool_name": "composio_execute", "parameters": {"action": "KEYS_OWN_TURN_PROBE"},
              "result": {"success": False, "error": "a key's own turn probe failed"}, "workspace_id": WS}
    outcome_hash = capture.build_tool_outcome(**failed)["metadata"]["outcome_hash"]

    async def turn():
        with keys_own_turn(KEYS_OWN):
            return capture.capture_tool_outcome(**failed, agent_id=42)

    try:
        assert asyncio.run(turn()) is None
        assert capture.should_dedupe(outcome_hash) is False  # never marked: a later turn still writes it
    finally:
        capture._SEEN_HASHES.pop(outcome_hash, None)


# ── 5. a product's turn still knows what the workspace can look up ───────────

AWARENESS = "## What you can look up\n- 12 documents, 1 database"


@pytest.mark.parametrize("lean", [KEYS_OWN, FOR_THE_LEARNER], ids=["a keys own turn", "an end users turn"])
def test_a_products_turn_still_knows_what_the_workspace_can_look_up(monkeypatch, lean):
    from modules.context.sections.base import SectionContext
    from modules.context.sections.memory import MemorySection

    async def awareness(self, ctx):
        return AWARENESS

    async def no_session(**kwargs):
        return None

    smart = NS(retrieve_memories=AsyncMock(return_value=NS(formatted_context="- Lessons are 40 minutes long",
                                                           memories=[], user_context=None)))
    monkeypatch.setattr(MemorySection, "_knowledge_awareness", awareness)
    monkeypatch.setattr("consumers.chatbot.smart_memory.get_smart_memory_manager", lambda: smart)
    monkeypatch.setattr("modules.memory.unified_memory_service.get_unified_memory_service",
                        lambda: NS(get_session=no_session))
    ctx = SectionContext(agent=NS(id=42), workspace_id=WS, widget_mode=False,
                         kwargs={"query": "How long is a lesson?", "chat_id": "chat-1"})
    with _turn(lean):
        text = asyncio.run(MemorySection().render(ctx))
    assert AWARENESS in text and "Lessons are 40 minutes long" in text


# ── 6. a turn that ends early leaves no hold behind ──────────────────────────

def _streaming_service(monkeypatch, ends):
    from consumers.chatbot.service import StreamingChatService

    service = object.__new__(StreamingChatService)
    service.widget_mode, service.widget_scopes, service.widget_team, service.widget_agent_lock = False, None, None, None
    service.db, service.workspace_id, service.streaming_handler = None, None, None

    async def scoped(*args, **kwargs):
        yield "first"
        if ends == "by an error":
            raise RuntimeError("the model went away")
        yield "second"

    monkeypatch.setattr(service, "_stream_response_with_agent_scoped", scoped)
    monkeypatch.setattr(service, "_hidden_action_scope", lambda agent_id: nullcontext())
    return service


def _person_service(store):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.widget_mode, svc.streaming_handler = False, get_streaming_handler()
    svc._smart_chat = NS(store=store, orchestrator=NS(memory_manager=NS(_last_l3_facts_stored=0)))
    return svc


@pytest.mark.parametrize("ends", ["closed", "by an error"])
def test_a_turn_that_ends_early_leaves_no_hold_behind(monkeypatch, ends):
    service = _streaming_service(monkeypatch, ends)
    store = AsyncMock(return_value=True)

    async def turns():
        for lean in (KEYS_OWN, FOR_THE_LEARNER):
            turn = service.stream_response_with_agent("c1", [], 42, 9, lean_request=lean)
            assert await turn.__anext__() == "first" and (on_a_keys_own_turn() or current_end_user())
            if ends == "closed":
                await turn.aclose()  # the reader went away mid-turn
            else:
                with pytest.raises(RuntimeError):
                    await turn.__anext__()
            assert on_a_keys_own_turn() is False and current_end_user() is None
        # the next turn in the same context, a person's, stores as always
        return [c async for c in _person_service(store)._remember_the_turn(FACT, "Noted.", "chat-2", 9)]

    asyncio.run(turns())
    assert store.await_args.kwargs["subject_id"] == "user:9"


# ── 7. every write a turn can make reaches no store on a key's own turn ──────

def _smart_memory(service):
    from consumers.chatbot.smart_memory import SmartMemoryManager

    manager = object.__new__(SmartMemoryManager)
    manager._unified_service, manager._cache = service, {}
    manager._last_tier, manager._last_l3_facts_stored = None, 0
    manager._distill_durable_facts = AsyncMock(return_value=[{"fact": FACT, "type": DEFAULT_FACT_TYPE,
                                                              "importance": 0.7}])
    manager._get_store_max_chars = lambda: 6000
    return manager


async def _every_write(plane, db, manager):
    """Each memory write a chat turn can make, from where the turn makes it."""
    from core.context_guard import ContextGuard
    from modules.memory.memory_tool import DurableMemoryStoreBackend, MemoryToolBackend
    from modules.memory.tool_outcome_capture import capture_tool_outcome, write_tool_outcome
    from modules.tools.discovery.handlers_workspace import checkpoint_thread, store_memory
    from services.knowledge_flywheel import ingest_agent_output

    service = plane.service
    await manager.store_conversation(WS, 42, "Explain photosynthesis, step by step.", "Light, water, CO2.",
                                     chat_id="chat-1")
    await manager.store_daily_summary(WS, "Explain photosynthesis, step by step.", "Light, water, CO2.")
    await service.update_session(workspace_id=WS, conversation_id="chat-1", user_msg="hi", assistant_msg="hello")
    await service.store_long_term(WS, FACT)
    await write_tool_outcome({"fact": "a tool's outcome", "metadata": {}}, workspace_id=WS, agent_id=42,
                             service=service)
    capture_tool_outcome(tool_name="composio_execute", parameters={"action": "EVERY_WRITE_PROBE"},
                         result={"success": False, "error": "every-write probe"}, workspace_id=WS, agent_id=42)
    await ContextGuard.__new__(ContextGuard)._flush_to_memory("Earlier: the exam is in June.", WS, 42)
    await store_memory(None, WS, {"content": FACT})
    await MemoryToolBackend(DurableMemoryStoreBackend(service), WS).handle(
        {"command": "create", "path": "/memories/plan.md", "file_text": "Week 1"})
    await checkpoint_thread(db, WS, {"_origin_chat_id": "chat-1"})
    await ingest_agent_output(None, WS, content="# Notes", filename="notes.md", source="generated_document")
    await _settled()


@pytest.mark.parametrize("lean", [KEYS_OWN, PERSON], ids=["a keys own turn", "a persons turn"])
def test_every_write_a_turn_can_make_reaches_no_store_on_a_keys_own_turn(monkeypatch, lean):
    """The behavioural guard: each write leaves the process through one of these exits, so a new write
    that is not held fails here. A person's turn reaches each exit, so the test sees every one."""
    from modules.memory import tool_outcome_capture
    from services import knowledge_flywheel

    plane = _Plane(monkeypatch)
    db = _checkpoint_seams(monkeypatch, {"chat-1": plane.chat()})
    manager = _smart_memory(plane.service)
    monkeypatch.setattr("consumers.chatbot.smart_memory.get_smart_memory_manager", lambda: manager)
    monkeypatch.setattr(knowledge_flywheel, "flywheel_enabled", lambda db, ws: plane.written.append(("flywheel",)))
    monkeypatch.setattr(tool_outcome_capture, "_SEEN_HASHES", type(tool_outcome_capture._SEEN_HASHES)())

    async def turn():
        with _turn(lean):
            await _every_write(plane, db, manager)

    asyncio.run(turn())
    exits = {w[0] for w in plane.written}
    assert exits == (set() if lean is KEYS_OWN else {"durable", "l2", "session", "summary", "flywheel"})


# ── 8. a Playbook run started in a product's turn runs under its hold ────────

def test_a_playbook_run_started_in_a_products_turn_runs_under_its_hold(monkeypatch):
    from api import recipe_executor
    from services.playbook_engine import get_playbook_engine

    seen = []

    async def execute_recipe_direct(**kwargs):
        seen.append((on_a_keys_own_turn(), current_end_user()))

    monkeypatch.setattr(recipe_executor, "execute_recipe_direct", execute_recipe_direct)

    async def turn(lean):
        with _turn(lean):
            get_playbook_engine().launch(recipe_execution_id="exec-1", recipe_id=1, workspace_id=WS, input_data={})
        await _settled()  # the run goes on after the turn's scopes close

    for lean in (KEYS_OWN, FOR_THE_LEARNER, PERSON):
        asyncio.run(turn(lean))
    assert seen == [(True, None), (False, LEARNER), (False, None)]
