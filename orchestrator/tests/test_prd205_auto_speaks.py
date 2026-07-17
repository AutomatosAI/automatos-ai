"""PRD-205 — Auto Speaks: background→chat delivery.

* S1/S2 — ChatMessenger posts assistant messages from background producers:
  Clerk-string resolution at the seam (#513), workspace-scoped chat
  validation, per-(workspace,user) Auto-thread fallback, fail-soft wrapper.
* S3 — additive columns exist and surface (messages.source, chats.kind,
  watches.origin_chat_id, agent_scheduled_tasks.origin_chat_id).
* S4 — origin capture is server-injected and unspoofable.
* S5 — watcher verdicts/actions/escalations also land in the conversation.
* S6 — scheduled-task output is delivered, not discarded (the PRD-77 fix).
* S7 — chat_changed frames route by name; notify_chat_event emits.
* S8 — /vote and /agents resolve (route-order regression, the PRD-220
  /search class).

DB-backed where the seam is the DB (messenger/thread) — skips cleanly
without Postgres; everything else pure/monkeypatched at the boundary.
"""
from __future__ import annotations

import asyncio
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from core.database.database import get_database_url  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (the PRD-204 suite idiom: skip without Postgres, sweep on teardown)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto-speaks suite needs Postgres with the 205 schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace_and_user(engine, new_session):
    ws_id = str(uuid.uuid4())
    clerk_id = f"user_test_{uuid.uuid4().hex[:10]}"
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd205-auto-speaks"},
    )
    row = s.execute(
        text(
            "INSERT INTO users (username, email, clerk_user_id) "
            "VALUES (:u, :e, :c) RETURNING id"
        ),
        {"u": clerk_id, "e": f"{clerk_id}@test.local", "c": clerk_id},
    ).first()
    user_int_id = int(row[0])
    s.commit()
    s.close()

    yield ws_id, clerk_id, user_int_id

    s = new_session.sweep()
    for stmt in (
        "DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM users WHERE clerk_user_id = :c",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id, "c": clerk_id})
    s.commit()
    s.close()


# ---------------------------------------------------------------------------
# S1/S2 — the messenger seam and the Auto thread
# ---------------------------------------------------------------------------

def test_auto_thread_find_or_create_is_idempotent(workspace_and_user, new_session):
    from services.chat_messenger import AUTO_CHAT_TITLE, find_or_create_auto_chat

    ws_id, _clerk, user_int_id = workspace_and_user
    db = new_session()
    try:
        first = find_or_create_auto_chat(db, ws_id, user_int_id)
        second = find_or_create_auto_chat(db, ws_id, user_int_id)
        assert first.id == second.id
        assert first.kind == "auto"
        assert first.title == AUTO_CHAT_TITLE
        assert first.visibility == "private"
    finally:
        db.close()


def test_post_background_message_falls_back_to_auto_thread(workspace_and_user, new_session):
    """No chat_id + resolvable Clerk id → the Auto thread; the message
    carries the persisted badge source and the exact in-turn parts shape."""
    from core.models.core import Message
    from services.chat_messenger import BACKGROUND_LABEL, post_background_message

    ws_id, clerk_id, _user = workspace_and_user
    db = new_session()
    try:
        message = post_background_message(
            db,
            workspace_id=ws_id,
            text="Watched it. 8.4/10, passed.",
            source={"origin": "watcher", "event": "watch_verdict"},
            clerk_user_id=clerk_id,
            link_type="watch",
            link_id="w-1",
        )
        assert message is not None
        saved = db.query(Message).filter(Message.id == message.id).first()
        assert saved.role == "assistant"
        assert saved.parts == [{"type": "text", "text": "Watched it. 8.4/10, passed."}]
        assert saved.source["label"] == BACKGROUND_LABEL
        assert saved.source["origin"] == "watcher"
        assert saved.source["link_type"] == "watch"

        chat = saved.chat
        assert chat.kind == "auto"
        assert str(chat.workspace_id) == ws_id
    finally:
        db.close()


def test_post_background_message_rejects_foreign_workspace_chat(
    workspace_and_user, new_session
):
    """A chat_id outside the workspace is never written to — it falls back
    to the Auto thread instead of leaking across chats."""
    from services.chat_messenger import post_background_message

    ws_id, clerk_id, _user = workspace_and_user
    db = new_session()
    try:
        foreign_chat_id = str(uuid.uuid4())  # not a chat in this workspace
        message = post_background_message(
            db,
            workspace_id=ws_id,
            text="hello",
            source={"origin": "watcher"},
            chat_id=foreign_chat_id,
            clerk_user_id=clerk_id,
        )
        assert message is not None
        assert str(message.chat_id) != foreign_chat_id
        assert message.chat.kind == "auto"
    finally:
        db.close()


def test_post_background_message_drops_without_target(workspace_and_user, new_session):
    """No valid chat AND no resolvable user → dropped (returns None), never
    a guessatorial write."""
    from services.chat_messenger import post_background_message

    ws_id, _clerk, _user = workspace_and_user
    db = new_session()
    try:
        assert (
            post_background_message(
                db,
                workspace_id=ws_id,
                text="orphan",
                source={"origin": "watcher"},
                clerk_user_id="user_does_not_exist",
            )
            is None
        )
    finally:
        db.close()


def test_deliver_background_message_never_raises():
    """The producer entrypoint is fail-soft: a poisoned db never propagates
    into a watcher tick or scheduled task."""
    from services.chat_messenger import deliver_background_message

    boom = MagicMock()
    boom.query.side_effect = RuntimeError("db down")
    assert (
        deliver_background_message(
            boom,
            workspace_id=str(uuid.uuid4()),
            text="x",
            source={"origin": "watcher"},
            clerk_user_id="user_x",
        )
        is None
    )


# ---------------------------------------------------------------------------
# S4 — origin capture is server-injected and unspoofable
# ---------------------------------------------------------------------------

def test_origin_chat_id_parser():
    from modules.tools.discovery.handlers_watches import _origin_chat_id

    good = uuid.uuid4()
    assert _origin_chat_id({"_origin_chat_id": str(good)}) == good
    assert _origin_chat_id({"_origin_chat_id": "not-a-uuid"}) is None
    assert _origin_chat_id({}) is None


def test_executor_injects_and_overwrites_origin():
    """Source-level pin: the origin comes from caller_context and OVERWRITES
    any LLM-supplied param. Since 2026-07-17 _created_by is hardened the
    same way (strip-then-inject — Gerard's call on the #563 flag), so BOTH
    attribution keys are unspoofable via tool args."""
    src = (
        _orchestrator_root
        / "modules" / "tools" / "discovery" / "platform_executor.py"
    ).read_text()
    block = re.search(
        r"_WATCH_ORIGIN_ACTIONS = \((?P<actions>[^)]*)\).*?"
        r"if action_name in _WATCH_ORIGIN_ACTIONS:(?P<body>.*?)\n\n",
        src,
        re.S,
    )
    assert block, "origin injection block missing"
    for action in (
        "platform_create_watch",
        "platform_create_mission",
        "platform_execute_playbook",
        "platform_execute_recipe",
        "platform_schedule_task",
    ):
        assert action in block.group("actions")
    assert 'caller_context or {}).get("conversation_id")' in block.group("body")
    assert '"_origin_chat_id" not in params' not in block.group("body")


def test_executor_created_by_is_strip_then_inject():
    """The 2026-07-17 hardening: _created_by is stripped from params before
    context injection — the old caller-preserving guard is gone, so a
    spoofed tool arg can never claim mission attribution on headless
    paths."""
    src = (
        _orchestrator_root
        / "modules" / "tools" / "discovery" / "platform_executor.py"
    ).read_text()
    block = re.search(
        r"if action_name in _MISSION_ATTRIBUTED:(?P<body>.*?)\n\n", src, re.S
    )
    assert block, "mission attribution block missing"
    assert '"_created_by" not in params' not in block.group("body")
    assert 'k != "_created_by"' in block.group("body")


def test_executor_strips_spoofed_origin_without_context(monkeypatch):
    """A caller-supplied _origin_chat_id must never reach a handler: the
    headless paths (board dispatcher, workflows) carry no conversation_id in
    caller_context, so inject-on-truthy alone would let the spoofed tool arg
    survive there. The executor strips the key FIRST, then injects only the
    context value."""
    import modules.tools.discovery as discovery_pkg
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    registry = MagicMock()
    registry.get.return_value = None  # no action_def -> permission gates no-op
    monkeypatch.setattr(discovery_pkg, "get_action_registry", lambda: registry)

    seen = []

    async def _handler(db, workspace_id, params):
        seen.append(params)
        return {"success": True}

    executor = PlatformActionExecutor(MagicMock(), uuid.uuid4())
    executor._handlers["platform_create_watch"] = _handler

    spoofed = str(uuid.uuid4())

    # Headless: no caller_context at all -> the spoofed arg is dropped.
    result = asyncio.run(
        executor.execute(
            "platform_create_watch",
            {"title": "w", "_origin_chat_id": spoofed},
            caller_context=None,
        )
    )
    assert result == {"success": True}
    assert "_origin_chat_id" not in seen[0]
    assert seen[0]["title"] == "w"

    # Context without a conversation (board dispatcher shape) -> still dropped.
    asyncio.run(
        executor.execute(
            "platform_create_watch",
            {"title": "w", "_origin_chat_id": spoofed},
            caller_context={"user_id": "user_abc"},
        )
    )
    assert "_origin_chat_id" not in seen[1]

    # Chat path: the context value wins over the spoofed arg.
    origin = str(uuid.uuid4())
    asyncio.run(
        executor.execute(
            "platform_create_watch",
            {"title": "w", "_origin_chat_id": spoofed},
            caller_context={"conversation_id": origin},
        )
    )
    assert seen[2]["_origin_chat_id"] == origin


# ---------------------------------------------------------------------------
# S5 — the watcher speaks (bell first, chat second, both fail-soft)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("event_type", ["watch_verdict", "watch_action", "watch_escalation"])
def test_watch_events_post_to_chat(monkeypatch, event_type):
    import services.chat_messenger as messenger
    from core.services.notification_dispatcher import NotificationDispatcher
    from services.watch_notifications import dispatch_watch_notification

    async def _dispatch(self, **kwargs):
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _dispatch)

    delivered = []
    monkeypatch.setattr(
        messenger, "deliver_background_message",
        lambda db, **kw: delivered.append(kw) or None,
    )

    origin = uuid.uuid4()
    watch = SimpleNamespace(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        created_by="user_abc",
        origin_chat_id=origin,
        title="Watch: nightly report",
        quality_threshold=0.7,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None  # clerk unresolvable → workspace-wide bell

    ok = asyncio.run(
        dispatch_watch_notification(
            db, watch, event_type=event_type, title="t", message="m", status="ok"
        )
    )
    assert ok is True
    assert len(delivered) == 1
    assert delivered[0]["chat_id"] == str(origin)
    assert delivered[0]["clerk_user_id"] == "user_abc"
    assert delivered[0]["source"]["event"] == event_type
    assert delivered[0]["link_type"] == "watch"


def test_non_watch_events_do_not_post_to_chat(monkeypatch):
    import services.chat_messenger as messenger
    from core.services.notification_dispatcher import NotificationDispatcher
    from services.watch_notifications import dispatch_watch_notification

    async def _dispatch(self, **kwargs):
        return {}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _dispatch)
    delivered = []
    monkeypatch.setattr(
        messenger, "deliver_background_message",
        lambda db, **kw: delivered.append(kw),
    )
    watch = SimpleNamespace(id=1, workspace_id=uuid.uuid4(), created_by=None)
    db = MagicMock()
    asyncio.run(
        dispatch_watch_notification(
            db, watch, event_type="watch_created", title="t", message=None
        )
    )
    assert delivered == []


# ---------------------------------------------------------------------------
# S6 — scheduled-task output is delivered, not discarded
# ---------------------------------------------------------------------------

def _run_trigger(monkeypatch, *, result, origin_chat_id, task_id=7):
    import modules.agents.factory.agent_factory as factory_mod
    import services.chat_messenger as messenger
    from services.scheduled_task_service import ScheduledTaskService

    class FakeFactory:
        def __init__(self, db_session):
            pass

        async def execute_with_prompt(self, **kwargs):
            return result

    monkeypatch.setattr(factory_mod, "AgentFactory", FakeFactory)
    delivered = []
    monkeypatch.setattr(
        messenger, "deliver_background_message",
        lambda db, **kw: delivered.append(kw),
    )
    asyncio.run(
        ScheduledTaskService._trigger_agent_chat(
            workspace_id=str(uuid.uuid4()),
            agent_id=1,
            message="[Scheduled Task #7] do the thing",
            db=MagicMock(),
            origin_chat_id=origin_chat_id,
            task_id=task_id,
        )
    )
    return delivered


def test_scheduled_output_delivered_to_origin_chat(monkeypatch):
    origin = str(uuid.uuid4())
    delivered = _run_trigger(
        monkeypatch, result={"result": "Here is the weekly digest."}, origin_chat_id=origin
    )
    assert len(delivered) == 1
    assert delivered[0]["chat_id"] == origin
    assert delivered[0]["text"] == "Here is the weekly digest."
    assert delivered[0]["source"] == {"origin": "scheduled_task"}
    assert delivered[0]["link_id"] == "7"


def test_scheduled_output_without_origin_stays_log_only(monkeypatch):
    delivered = _run_trigger(
        monkeypatch, result={"result": "text"}, origin_chat_id=None
    )
    assert delivered == []


def test_scheduled_empty_output_posts_nothing(monkeypatch):
    delivered = _run_trigger(
        monkeypatch, result={"result": "   "}, origin_chat_id=str(uuid.uuid4())
    )
    assert delivered == []


# ---------------------------------------------------------------------------
# S7 — frame routing + the NOTIFY emitter
# ---------------------------------------------------------------------------

def test_frame_for_payload_routes_chat_changed():
    import json

    from services.board_events import frame_for_payload

    ws = str(uuid.uuid4())
    chat_payload = json.dumps(
        {"workspace_id": ws, "chat_id": "c1", "user_id": 5, "event": "chat_changed"}
    )
    board_payload = json.dumps({"workspace_id": ws, "task_id": 1, "event": "task_changed"})
    foreign_payload = json.dumps(
        {"workspace_id": str(uuid.uuid4()), "event": "chat_changed"}
    )

    assert "event: chat_changed" in frame_for_payload(chat_payload, ws)
    assert "event: board_changed" in frame_for_payload(board_payload, ws)
    assert frame_for_payload(foreign_payload, ws) is None  # tenant gate holds


def test_notify_chat_event_emits_payload():
    import json

    from services.board_events import notify_chat_event

    db = MagicMock()
    ws = uuid.uuid4()
    notify_chat_event(db, workspace_id=ws, chat_id="c-9", user_id=3)
    args = db.execute.call_args
    params = args.args[1]
    payload = json.loads(params["payload"])
    assert payload == {
        "workspace_id": str(ws),
        "chat_id": "c-9",
        "user_id": 3,
        "event": "chat_changed",
    }


def test_history_and_get_chat_surface_kind(workspace_and_user, new_session):
    """S7 passthrough: /history rows and GET /{chat_id} carry ``chats.kind``
    so the UI can mark the Auto thread ('user' for ordinary chats)."""
    from api.chat import get_chat, get_chat_history
    from consumers.chatbot.service import ChatService
    from services.chat_messenger import find_or_create_auto_chat

    ws_id, _clerk, user_int_id = workspace_and_user
    db = new_session()
    try:
        regular = ChatService(db).create_chat(
            user_id=user_int_id,
            title="regular-kind",
            workspace_id=uuid.UUID(ws_id),
        )
        auto = find_or_create_auto_chat(db, ws_id, user_int_id)

        # get_user_id's fast path takes an integer ctx.user.id as-is, so a
        # SimpleNamespace principal exercises the real endpoint bodies.
        ctx = SimpleNamespace(
            workspace_id=uuid.UUID(ws_id),
            user=SimpleNamespace(id=user_int_id),
        )

        rows = asyncio.run(get_chat_history(limit=50, ctx=ctx, db=db))
        by_id = {row["id"]: row for row in rows}
        assert by_id[str(auto.id)]["kind"] == "auto"
        assert by_id[str(regular.id)]["kind"] == "user"

        payload = asyncio.run(get_chat(str(auto.id), ctx=ctx, db=db))
        assert payload["kind"] == "auto"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S8 — /vote and /agents resolve (the PRD-220 /search regression class)
# ---------------------------------------------------------------------------

def test_chat_router_literal_routes_precede_param_routes():
    """Declaration order IS the dispatch order in FastAPI: every literal
    chat route must be declared before the /{chat_id} catch-alls for its
    method, or it is dead (matched as chat_id='vote'/'agents'/'search')."""
    from api.chat import router

    order = [
        (sorted(r.methods)[0], r.path)
        for r in router.routes
        if hasattr(r, "methods")
    ]

    def index_of(method, path):
        return order.index((method, path))

    assert index_of("GET", "/api/chat/search") < index_of("GET", "/api/chat/{chat_id}")
    assert index_of("GET", "/api/chat/agents") < index_of("GET", "/api/chat/{chat_id}")
    assert index_of("PATCH", "/api/chat/vote") < index_of("PATCH", "/api/chat/{chat_id}")


# ---------------------------------------------------------------------------
# S3 — the additive columns exist on the models and in the one migration
# ---------------------------------------------------------------------------

def test_models_carry_the_205_columns():
    from core.models.core import Chat, Message
    from core.models.watches import Watch

    assert "kind" in Chat.__table__.columns
    assert Chat.__table__.columns["kind"].nullable is False
    assert "source" in Message.__table__.columns
    assert Message.__table__.columns["source"].nullable is True
    assert "origin_chat_id" in Watch.__table__.columns


def test_migration_is_additive_and_single_parent():
    src = (
        _orchestrator_root / "alembic" / "versions" / "prd205_auto_speaks.py"
    ).read_text()
    assert 'down_revision = "prd199_drop_fake_stats"' in src
    for col in ("source", "kind", "origin_chat_id"):
        assert col in src
    assert "agent_scheduled_tasks" in src
    assert "drop_table" not in src  # additive only
