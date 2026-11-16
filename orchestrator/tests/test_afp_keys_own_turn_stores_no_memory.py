"""PRD-274 (made while building, 16 Nov 2026): a server key's own chat turn stores no memory, and every
product turn follows §5.5 for what it reads and for the work done after it.

A key's turn that names no end user is nobody's, as a widget turn is an anonymous visitor's (F154). What
was said in it is no fact about the workspace, so nothing of it is written where the workspace's other
turns would recall it: no distilled facts, no L2 row or mirror, no daily log, no L1 session. The rule is
held where memory is written (``core.security.end_user_turn.kept_from_a_keys_own_turn``). A key's turn for
an end user still writes to their namespace; a person's turn, a lean agent's included, stores as before.

A product's turn (a key's own, or an end user's) reads the workspace's durable facts, never its other
conversations or its daily logs; nothing it makes is ingested into the workspace's knowledge; and the idle
checkpoint sweep never reads a key's own conversation or a widget visitor's (an end user's it checkpoints as
theirs: tests/test_afp_product_turn_holds.py).
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from consumers.chatbot.lean_request import NOTHING_ASKED, LeanRequest, TurnModel
from consumers.chatbot.smart_memory import SmartMemoryManager
from core.security.end_user_turn import (KEPT_FROM_A_KEYS_OWN_TURN, KEYS_OWN, EndUser, current_end_user,
                                         end_user_turn, keys_own_turn, on_a_keys_own_turn)
from modules.memory.durable_store import DurableMemoryStore
from modules.memory.unified_memory_service import MemoryNamespace, UnifiedMemoryService
from modules.memory.write_contract import DEFAULT_FACT_TYPE

WS = "33333333-3333-3333-3333-333333333333"
ASKED = "Can you explain photosynthesis again, step by step?"
SAID = "Plants turn light, water and carbon dioxide into sugar and oxygen."
LEARNER = EndUser(WS, "learner-7", "record-1")
KEYS_OWN_REQUEST = LeanRequest(server_key=True)
FOR_THE_LEARNER = LeanRequest(server_key=True, end_user=LEARNER)
PERSONS_LEAN_REQUEST = LeanRequest(model=TurnModel("openrouter", "acme/fast-1"))
NS_ = MemoryNamespace(WS)


@pytest.fixture(autouse=True)
def consented(monkeypatch):
    """The end user's consent, as their writes check it: on record."""
    async def on_record(end_user):
        return True

    monkeypatch.setattr("core.security.end_user_turn._consented", on_record)


# ── whose turn it is ──────────────────────────────────────────────────────────

def test_a_keys_request_says_a_key_sent_it(monkeypatch):
    from api import chat_server_key

    monkeypatch.setattr(chat_server_key, "_refuse_unless_allowed", lambda db, ctx, request: None)
    monkeypatch.setattr("core.models.end_users.consent_record_id", lambda db, ws, end_user_id: "record-1")
    checked = chat_server_key.what_a_server_key_may_ask(lambda db, ctx, request: NOTHING_ASKED)
    key, person = NS(auth_type="sdk_key", workspace_id=WS), NS(auth_type="clerk", workspace_id=WS)

    assert checked(None, key, NS(endUserId=None)) == KEYS_OWN_REQUEST
    assert checked(None, key, NS(endUserId="learner-7")) == FOR_THE_LEARNER
    assert checked(None, person, NS(endUserId=None)) is NOTHING_ASKED


def test_only_a_keys_turn_for_no_end_user_is_its_own(monkeypatch):
    from consumers.chatbot.service import StreamingChatService

    service = object.__new__(StreamingChatService)
    service.widget_mode, service.widget_scopes, service.widget_team, service.widget_agent_lock = False, None, None, None
    service.db, service.workspace_id = None, None
    seen = []

    async def scoped(*args, **kwargs):
        seen.append((on_a_keys_own_turn(), current_end_user()))
        yield "chunk"

    monkeypatch.setattr(service, "_stream_response_with_agent_scoped", scoped)
    monkeypatch.setattr(service, "_hidden_action_scope", lambda agent_id: nullcontext())

    async def drain(lean):
        return [c async for c in service.stream_response_with_agent("c1", [], 42, 9, lean_request=lean)]

    for lean in (KEYS_OWN_REQUEST, FOR_THE_LEARNER, PERSONS_LEAN_REQUEST, None):
        asyncio.run(drain(lean))
    assert seen == [(True, None), (False, LEARNER), (False, None), (False, None)]
    assert on_a_keys_own_turn() is False


# ── the turn's own store, after the reply ─────────────────────────────────────

def _turn_service(store, widget=False):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.widget_mode, svc.streaming_handler = widget, get_streaming_handler()
    svc._smart_chat = NS(store=store, orchestrator=NS(memory_manager=NS(_last_l3_facts_stored=1,
                                                                         _last_tier="global")))
    return svc


async def _remembered(svc, lean):
    with end_user_turn(getattr(lean, "end_user", None)), keys_own_turn(lean):
        return [chunk async for chunk in svc._remember_the_turn(ASKED, SAID, "chat-1", 9)]


@pytest.mark.asyncio
async def test_a_keys_own_turn_distils_nothing_and_says_nothing_was_stored():
    store = AsyncMock(return_value=True)
    assert await _remembered(_turn_service(store), KEYS_OWN_REQUEST) == []
    store.assert_not_awaited()  # no distil: no model call, nothing written


@pytest.mark.asyncio
async def test_a_keys_turn_for_its_end_user_still_stores_as_theirs():
    store = AsyncMock(return_value=True)
    frames = await _remembered(_turn_service(store), FOR_THE_LEARNER)
    assert store.await_args.kwargs["subject_id"] == "end_user:learner-7"
    assert len(frames) == 1 and "memory-stored" in frames[0]


@pytest.mark.asyncio
async def test_a_persons_turn_with_a_lean_agent_still_stores():
    store = AsyncMock(return_value=True)
    frames = await _remembered(_turn_service(store), PERSONS_LEAN_REQUEST)
    assert store.await_args.kwargs["subject_id"] == "user:9"
    assert len(frames) == 1 and "memory-stored" in frames[0]


@pytest.mark.asyncio
async def test_a_widget_turn_still_stores_nothing():
    store = AsyncMock(return_value=True)
    assert await _remembered(_turn_service(store, widget=True), NOTHING_ASKED) == []
    store.assert_not_awaited()


# ── where memory is written ───────────────────────────────────────────────────

def _memory_plane():
    """The real write path of a chat turn's memory, recorded where each write leaves the process: the
    durable store (at its dedup lookup, which sees the final namespace), the L2 table and the L1 session."""
    written = []
    store = object.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done = True, True

    async def deduped(user_id, content_hash, metadata):
        written.append(("durable", user_id))
        return {"success": True, "id": "existing"}

    store._deduped = deduped
    service = object.__new__(UnifiedMemoryService)
    service._durable = store

    def l2_row(workspace_id, content, content_type, agent_id, importance, metadata):
        written.append(("l2", content_type))
        return "row-1"

    service._store_short_term_sync = l2_row
    conn = NS(get=lambda key: None, setex=lambda key, ttl, value: written.append(("session", key)))
    service._get_redis = lambda: NS(get_redis=lambda: conn)
    return store, service, written


def _orchestrator(service):
    """The turn's ``SmartChatOrchestrator`` on the plane, its distiller finding one durable fact."""
    from consumers.chatbot.smart_orchestrator import SmartChatOrchestrator

    manager = object.__new__(SmartMemoryManager)
    manager._unified_service, manager._cache = service, {}
    manager._last_tier, manager._last_l3_facts_stored = None, 0
    manager._distill_durable_facts = AsyncMock(return_value=[
        {"fact": "The learner is revising photosynthesis for an exam.", "type": DEFAULT_FACT_TYPE, "importance": 0.7}])
    manager._get_store_max_chars = lambda: 6000
    orchestrator = object.__new__(SmartChatOrchestrator)
    orchestrator.workspace_id, orchestrator.agent_id, orchestrator.widget_mode = WS, 42, False
    orchestrator.memory_manager, orchestrator._unified_memory = manager, service
    return orchestrator


async def _settled():
    """Wait for every write the turn scheduled, and the writes those scheduled in turn."""
    while True:
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


EVERY_WRITE_OF_A_TURN = {
    ("durable", NS_.workspace()),     # the distilled fact
    ("l2", "transcript"),             # the L2 transcript ...
    ("durable", NS_.l2()),            # ... and its mirror (the daily log's L2 row's too)
    ("durable", NS_.daily()),         # the daily log
    ("l2", "heartbeat_log"),          # the daily log's L2 row
    ("session", NS_.session("chat-1")),  # the L1 session
}


async def _post_response(service, lean):
    """``StreamingChatService._post_response`` on the plane, for the turn ``lean`` names."""
    from consumers.chatbot.integration import SmartChatIntegration
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.widget_mode, svc.streaming_handler = False, get_streaming_handler()
    svc._smart_chat = SmartChatIntegration.__new__(SmartChatIntegration)
    svc._smart_chat.orchestrator = _orchestrator(service)
    svc.db = NS(query=lambda *a: NS(filter=lambda *a: NS(first=lambda: None)))  # no agent row to count on
    with end_user_turn(getattr(lean, "end_user", None)), keys_own_turn(lean):
        frames = [c async for c in svc._post_response(ASKED, SAID, "chat-1", NS(), 42, NS(usage=None), None,
                                                      user_id=9)]
        await _settled()
    return frames


@pytest.fixture
def no_live_eval(monkeypatch):
    from core.services.futureagi_service import futureagi_service

    monkeypatch.setattr(futureagi_service, "schedule_live_eval", lambda *a, **k: None)


@pytest.mark.asyncio
async def test_everything_after_a_keys_own_reply_writes_no_memory(no_live_eval):
    """The guard: the whole post-response path of a key's own turn, through the real memory plane, makes
    no write. A new write added to that path reaches one of these exits, and fails this test unless held."""
    _store, service, written = _memory_plane()
    assert await _post_response(service, KEYS_OWN_REQUEST) == [] and written == []
    await _settled()
    assert written == []


@pytest.mark.asyncio
async def test_a_persons_turn_writes_every_layer_as_before(no_live_eval):
    """The same path on a person's lean turn writes every layer, so the test above sees each exit."""
    _store, service, written = _memory_plane()
    await _post_response(service, PERSONS_LEAN_REQUEST)
    assert set(written) == EVERY_WRITE_OF_A_TURN


@pytest.mark.asyncio
async def test_a_write_made_during_a_keys_own_turn_is_not_made(monkeypatch):
    """A tool's outcome, the context guard's flush, a flywheel ingest: each is made during the turn, not
    after it, through a held write. Each answers as when it stores nothing."""
    from modules.memory.tool_outcome_capture import write_tool_outcome
    from services import knowledge_flywheel

    store, service, written = _memory_plane()
    manager = object.__new__(SmartMemoryManager)

    def never(*args, **kwargs):
        raise AssertionError("the flywheel was asked on a key's own turn")

    monkeypatch.setattr(knowledge_flywheel, "flywheel_enabled", never)
    with keys_own_turn(KEYS_OWN_REQUEST):
        assert await store.add([{"role": "user", "content": "x"}], user_id=NS_.workspace(), workspace_id=WS) == {
            "success": False, "skipped": KEYS_OWN}
        assert await write_tool_outcome({"fact": "a tool's outcome", "metadata": {}}, workspace_id=WS,
                                        agent_id=42, service=service) is True  # best-effort: it answers, stores nothing
        assert await service.store_short_term(WS, "a transcript", content_type="exchange") is None
        assert await service.store_daily_log(WS, "today") == {"success": False, "skipped": KEYS_OWN}
        assert await manager.store_daily_summary(WS, "hello there", "hi") is False
        await service.update_session(workspace_id=WS, conversation_id="chat-1", user_msg="hi", assistant_msg="hello")
        assert await knowledge_flywheel.ingest_agent_output(
            None, WS, content="# Notes", filename="notes.md", source="generated_document") is None
        await _settled()
    assert written == []


@pytest.mark.asyncio
async def test_a_keys_turn_for_its_end_user_still_writes_to_their_namespace():
    store, _service, written = _memory_plane()
    with end_user_turn(LEARNER), keys_own_turn(FOR_THE_LEARNER):
        await store.add([{"role": "user", "content": "prefers worked examples"}], user_id=NS_.workspace(),
                        workspace_id=WS)
    assert written == [("durable", NS_.end_user("learner-7"))]


HELD_WRITES = ("modules.memory.durable_store:DurableMemoryStore.add",
               "modules.memory.unified_memory_service:UnifiedMemoryService.store_short_term",
               "modules.memory.unified_memory_service:UnifiedMemoryService.store_daily_log",
               "modules.memory.unified_memory_service:UnifiedMemoryService.update_session",
               "consumers.chatbot.smart_memory:SmartMemoryManager.store_daily_summary",
               "services.knowledge_flywheel:ingest_agent_output")


@pytest.mark.parametrize("where", HELD_WRITES)
def test_every_write_where_memory_leaves_the_process_is_held(where):
    import importlib

    module, _, name = where.partition(":")
    write = importlib.import_module(module)
    for part in name.split("."):
        write = getattr(write, part)
    assert getattr(write, KEPT_FROM_A_KEYS_OWN_TURN, False), f"{where} is not held on a key's own turn"


# ── what a product's turn reads (§5.5) ────────────────────────────────────────

OTHER_CONVERSATION = "Owner said: keep the margin target at 40% until March"
DAILY_LOG = "Owner priced the Q4 plan with the accountant"
DURABLE_FACT = "Lessons are 40 minutes long"


class _Router:
    """The unified memory service as the section reads it: the Context Router's bundle (another
    conversation's L2 and the daily logs) and the conversation's own L1 session."""

    def __init__(self):
        self.calls = []

    async def retrieve_context(self, **kwargs):
        self.calls.append(kwargs)
        return NS(long_term_memories=[], session_summary=None, daily_logs=DAILY_LOG,
                  temporal_results=[{"memory": OTHER_CONVERSATION}], knowledge_awareness=None,
                  total_tokens_estimate=40, signals=[])

    async def get_session(self, **kwargs):
        return None


class _SmartMemory:
    def __init__(self):
        self.recalls, self.daily_logs = [], 0

    async def retrieve_memories(self, **kwargs):
        self.recalls.append(kwargs)
        return NS(formatted_context=f"- {DURABLE_FACT}", memories=[{"memory": DURABLE_FACT}], user_context=None)

    async def get_daily_logs(self, **kwargs):
        self.daily_logs += 1
        return DAILY_LOG


def _section_for(monkeypatch, lean):
    from modules.context.sections.base import SectionContext
    from modules.context.sections.memory import MemorySection

    router, smart = _Router(), _SmartMemory()
    monkeypatch.setattr("modules.memory.unified_memory_service.get_unified_memory_service", lambda: router)
    monkeypatch.setattr("consumers.chatbot.smart_memory.get_smart_memory_manager", lambda: smart)
    ctx = SectionContext(agent=NS(id=42), workspace_id=WS, widget_mode=False,
                         kwargs={"query": "How long is a lesson?", "chat_id": "chat-1"})
    with end_user_turn(getattr(lean, "end_user", None)), keys_own_turn(lean):
        text = asyncio.run(MemorySection().render(ctx))
    return text, router, smart


@pytest.mark.parametrize("lean", [KEYS_OWN_REQUEST, FOR_THE_LEARNER], ids=["a keys own turn", "an end users turn"])
def test_a_products_turn_reads_no_other_conversation_and_no_daily_log(monkeypatch, lean):
    text, router, smart = _section_for(monkeypatch, lean)
    assert router.calls == [] and smart.daily_logs == 0              # no Context Router L2, no daily log
    assert [recall["chat_transcripts"] for recall in smart.recalls] == [False]  # no transcript promoted to L3
    assert DURABLE_FACT in text
    assert OTHER_CONVERSATION not in text and DAILY_LOG not in text


def test_a_persons_turn_reads_the_workspace_as_before(monkeypatch):
    text, router, _smart = _section_for(monkeypatch, PERSONS_LEAN_REQUEST)
    assert len(router.calls) == 1 and OTHER_CONVERSATION in text and DAILY_LOG in text


@pytest.mark.asyncio
async def test_nothing_an_end_users_turn_makes_is_ingested(monkeypatch):
    from services import knowledge_flywheel

    def never(*args, **kwargs):
        raise AssertionError("the flywheel was asked on an end user's turn")

    monkeypatch.setattr(knowledge_flywheel, "flywheel_enabled", never)
    with end_user_turn(LEARNER), keys_own_turn(FOR_THE_LEARNER):
        assert await knowledge_flywheel.ingest_agent_output(
            None, WS, content="# Their plan", filename="plan.md", source="generated_document") is None


# ── the idle checkpoint sweep: never a key's own or a visitor's conversation ──

@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    with eng.begin() as conn:  # a test database built before these
        conn.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS end_user_id VARCHAR(128)"))
        conn.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS widget_key_id UUID"))
        conn.execute(text("ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS model_allow_list JSONB"))
        conn.execute(text("ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS service_user_id INTEGER "
                          "REFERENCES users(id) ON DELETE SET NULL"))
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'keys-own-turn')"), {"id": ws})
    s.commit()
    s.close()
    made = []
    yield ws, made
    s = new_session.sweep()
    users = s.execute(text("SELECT service_user_id FROM sdk_api_keys WHERE workspace_id = CAST(:id AS uuid) "
                           "AND service_user_id IS NOT NULL"), {"id": ws}).scalars().all()
    for table in ("chats", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    if users or made:
        s.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": [*users, *made]})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _keys_service_user(new_session, ws):
    """The users row a server key of ``ws`` runs its turns as, the key revoked: revoking keeps its
    service user, and its conversations stay its own."""
    from api.chat_server_key import service_user_for
    from core.auth.scopes import AGENTS_CHAT
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        made = ApiKeyService.create_api_key(s, uuid.UUID(ws), "tutor", key_type="server", permissions=[AGENTS_CHAT])
        user_id = service_user_for(s, made["id"])
        assert ApiKeyService.revoke_api_key(s, uuid.UUID(str(made["id"])), uuid.UUID(ws))
        return user_id
    finally:
        s.close()


def _person(new_session, made):
    from core.models.core import User

    s = new_session()
    try:
        name = f"desk-{uuid.uuid4().hex[:8]}"
        user = User(username=name, email=f"{name}@example.com", name="A person", is_active=True)
        s.add(user)
        s.commit()
        made.append(user.id)
        return user.id
    finally:
        s.close()


def _idle_chat(new_session, ws, user_id, **tags):
    """A conversation of four messages, idle past the sweep's threshold; ``tags`` are its
    ``end_user_id`` or ``widget_key_id``."""
    from consumers.chatbot import ChatService

    s = new_session()
    try:
        chats = ChatService(s)
        chat_id = str(chats.create_chat(user_id=user_id, title=f"chat {uuid.uuid4().hex[:6]}",
                                        workspace_id=uuid.UUID(ws), **tags).id)
        for role in ("user", "assistant", "user", "assistant"):
            chats.save_message(chat_id=chat_id, role=role, parts=[{"type": "text", "text": f"the {role} speaks"}],
                               workspace_id=ws)
        s.execute(text("UPDATE chats SET updated_at = NOW() - INTERVAL '40 minutes' WHERE id = CAST(:id AS uuid)"),
                  {"id": chat_id})
        s.commit()
        return chat_id
    finally:
        s.close()


def _a_keys_own(new_session, ws, person):
    return _idle_chat(new_session, ws, _keys_service_user(new_session, ws))


def _a_widget_visitors(new_session, ws, person):
    return _idle_chat(new_session, ws, person, widget_key_id=uuid.uuid4())  # a widget's chats are a person's row


def _candidates(new_session, ws, monkeypatch):
    """The sweep's batch for ``ws``: ``{chat id: end user id}``."""
    from config import config
    from services import memory_jobs

    for name, value in (("THREAD_CHECKPOINT_IDLE_MINUTES", 30), ("THREAD_CHECKPOINT_LOOKBACK_HOURS", 48),
                        ("THREAD_CHECKPOINT_MIN_MESSAGES", 4), ("THREAD_CHECKPOINT_BATCH", 100000)):
        monkeypatch.setattr(config, name, value, raising=False)
    s = new_session()
    try:
        return {str(chat_id): end_user_id for chat_id, workspace_id, end_user_id
                in memory_jobs._checkpoint_candidates(s) if str(workspace_id) == ws}
    finally:
        s.close()


@pytest.mark.parametrize("never_swept", [_a_keys_own, _a_widget_visitors], ids=["a keys own", "a widget visitors"])
def test_the_idle_checkpoint_sweep_never_reads_a_keys_own_or_a_visitors_conversation(new_session, workspace,
                                                                                     monkeypatch, never_swept):
    ws, made = workspace
    person = _person(new_session, made)
    skipped = never_swept(new_session, ws, person)
    persons = _idle_chat(new_session, ws, person)
    assert _candidates(new_session, ws, monkeypatch) == {persons: None} and skipped


def test_the_idle_checkpoint_sweep_takes_an_end_users_conversation_with_its_end_user(new_session, workspace,
                                                                                     monkeypatch):
    ws, made = workspace
    theirs = _idle_chat(new_session, ws, _keys_service_user(new_session, ws), end_user_id="learner-7")
    persons = _idle_chat(new_session, ws, _person(new_session, made))
    assert _candidates(new_session, ws, monkeypatch) == {theirs: "learner-7", persons: None}
