"""PRD-227 P227-RVW-2 — mission-narration target must be server-injected and
owner-scoped, never caller-controlled (cross-user chat injection).

The review finding: the narration delivery target ``run.config['origin_chat_id']``
was caller-controllable on BOTH create paths — the tool handler preserved a
caller-supplied ``config['origin_chat_id']`` and fell back to ``config['chat_id']``
(handlers_missions.py), and the REST route copied ``body.config`` verbatim into
``run.config`` (api/missions.py). At delivery, ``post_background_message`` honoured
any chat matching only ``Chat.workspace_id`` — cross-tenant-safe but NOT
cross-user-safe — so a caller with ``missions:create`` (or a prompt-injected agent)
could aim every ``Auto · mission`` line at another workspace member's private chat.

Two layers, both proven here:
  * Layer 1 (strip at the entry points): the origin is set from the SERVER-injected
    ``_origin_chat_id`` ONLY; any caller-supplied ``origin_chat_id``/``chat_id`` is
    stripped from config before it becomes ``run.config`` (the executor's
    strip-then-inject discipline applied to the nested config), on the tool path
    (handlers_missions) and the REST path (api/missions).
  * Layer 2 (owner check at the single delivery choke point): ``post_background_message``
    honours a caller-supplied ``chat_id`` only when the chat is owned by the message's
    resolved owner (``clerk_user_id`` → ``users.id``); a workspace-valid chat owned by
    another user falls back to the owner's Auto thread. A producer with no resolvable
    owner (agent-created scheduled tasks, whose origin was server-captured) keeps the
    legacy workspace-only honour so it is not regressed.

PURE tests run everywhere (DB layer mocked); the ``@integration`` cross-user
round-trip skips cleanly without local Postgres (CI test.yml is the gate).
"""
from __future__ import annotations

import asyncio
import types
import uuid
from unittest.mock import MagicMock

import pytest

import modules.tools.discovery.handlers_missions as hm
import modules.tools.discovery.handlers_watches as hw
import services.chat_messenger as cm
import services.coordinator_service as cs_mod
from core.models.orchestration_enums import RunState


# ===========================================================================
# Layer 1 — the create paths strip caller-supplied origin (PURE)
# ===========================================================================

def _capture_tool_config(monkeypatch) -> dict:
    """Patch the tool-path create so we capture the config that becomes
    run.config, without touching a DB or the real coordinator/planner."""
    captured: dict = {}

    class _FakeCoordinator:
        async def create_mission(self, *, db, workspace_id, goal, created_by, config):
            captured["config"] = config
            captured["created_by"] = created_by
            return types.SimpleNamespace(
                id=uuid.uuid4(),
                state=RunState.AWAITING_APPROVAL.value,
                goal=goal,
                plan={"tasks": []},
            )

    monkeypatch.setattr(cs_mod, "CoordinatorService", _FakeCoordinator)
    monkeypatch.setattr(hw, "auto_create_watch", lambda *a, **k: None)
    monkeypatch.setattr(hm, "_recent_chat_context", lambda *a, **k: [])
    return captured


def test_tool_create_strips_caller_supplied_origin_and_chat_id(monkeypatch):
    """No server origin → a caller/LLM-supplied origin_chat_id AND chat_id are
    stripped from config; unrelated keys survive; origin is left UNSET (→ Auto)."""
    captured = _capture_tool_config(monkeypatch)
    params = {
        "goal": "Do the thing",
        "_created_by": "user_owner",
        "config": {
            "origin_chat_id": str(uuid.uuid4()),  # attacker: another user's chat
            "chat_id": str(uuid.uuid4()),          # attacker: the fallback vector
            "keep": "yes",
        },
    }
    result = asyncio.run(hm.create_mission(MagicMock(), uuid.uuid4(), params))

    cfg = captured["config"]
    assert "origin_chat_id" not in cfg, "caller origin_chat_id must be stripped"
    assert "chat_id" not in cfg, "caller chat_id (the fallback vector) must be stripped"
    assert cfg.get("keep") == "yes", "unrelated config keys must survive"
    assert result["success"] is True


def test_tool_create_uses_server_injected_origin_only(monkeypatch):
    """A server-injected _origin_chat_id sets the origin; a caller-supplied
    config['origin_chat_id'] never survives or overrides it."""
    captured = _capture_tool_config(monkeypatch)
    server_chat = str(uuid.uuid4())
    params = {
        "goal": "Do the thing",
        "_created_by": "user_owner",
        "_origin_chat_id": server_chat,  # executor-injected, non-spoofable
        "config": {
            "origin_chat_id": str(uuid.uuid4()),  # attacker value — must lose
            "chat_id": str(uuid.uuid4()),
        },
    }
    asyncio.run(hm.create_mission(MagicMock(), uuid.uuid4(), params))

    cfg = captured["config"]
    assert cfg["origin_chat_id"] == server_chat, "origin must be the SERVER value only"
    assert "chat_id" not in cfg


def test_rest_create_strips_caller_supplied_origin_and_chat_id(monkeypatch):
    """The REST route (api/missions.py) copies body.config verbatim; it must
    strip origin_chat_id/chat_id before they become run.config."""
    import api.missions as am

    captured: dict = {}

    class _FakeCoordinator:
        async def create_mission(self, *, db, workspace_id, goal, created_by, config):
            captured["config"] = config
            return types.SimpleNamespace(id=uuid.uuid4(), state=RunState.AWAITING_APPROVAL.value)

    monkeypatch.setattr(am, "get_coordinator_service", lambda: _FakeCoordinator())
    monkeypatch.setattr(am, "_run_to_response", lambda run: {"ok": True})

    body = am.MissionCreateRequest(
        goal="Do the thing",
        config={
            "origin_chat_id": str(uuid.uuid4()),  # attacker: another user's chat
            "chat_id": str(uuid.uuid4()),
            "keep": "yes",
        },
    )
    ctx = types.SimpleNamespace(workspace_id=uuid.uuid4(), user=types.SimpleNamespace(id="user_owner"))

    asyncio.run(am.create_mission(body, ctx, MagicMock()))

    cfg = captured["config"]
    assert "origin_chat_id" not in cfg, "REST caller origin_chat_id must be stripped"
    assert "chat_id" not in cfg, "REST caller chat_id must be stripped"
    assert cfg.get("keep") == "yes"


def test_blog_create_strips_caller_supplied_origin_and_chat_id(monkeypatch):
    """CRITICAL (security review): create_blog_post_from_topic is a FOURTH
    create_mission caller — and its created_by is an agent-id (unresolvable to a
    user), so a caller-supplied origin would slip through the messenger's no-owner
    branch straight into a victim's chat. It MUST strip origin_chat_id/chat_id."""
    import modules.tools.discovery.handlers_blog as hb

    captured: dict = {}

    class _FakeCoordinator:
        async def create_mission(self, *, db, workspace_id, goal, created_by, config):
            captured["config"] = config
            captured["created_by"] = created_by
            return types.SimpleNamespace(
                id=uuid.uuid4(), state=RunState.AWAITING_APPROVAL.value, plan={"tasks": []}
            )

    monkeypatch.setattr(cs_mod, "CoordinatorService", _FakeCoordinator)

    params = {
        "topic": "Agentic AI",
        "category": "AI & Automation",
        "_agent_id": "42",
        "config": {
            "origin_chat_id": str(uuid.uuid4()),  # attacker: another user's chat
            "chat_id": str(uuid.uuid4()),
            "keep": "yes",
        },
    }
    result = asyncio.run(hb.create_blog_post_from_topic(MagicMock(), uuid.uuid4(), params))

    cfg = captured["config"]
    assert "origin_chat_id" not in cfg, "blog path must strip caller origin_chat_id"
    assert "chat_id" not in cfg, "blog path must strip caller chat_id"
    assert cfg.get("keep") == "yes"
    assert result["success"] is True
    # created_by is an agent id — the no-owner delivery branch is exactly the one
    # this path would otherwise hit, so the strip is load-bearing here.
    assert captured["created_by"] == "42"


def test_import_plan_strips_caller_supplied_origin_and_chat_id(monkeypatch):
    """The /import-plan sibling endpoint strips caller origin too — parity with
    create_mission and defense-in-depth (its created_by is a verified user, but
    it must not diverge silently from its sibling)."""
    import api.missions as am

    captured: dict = {}

    class _FakeCoordinator:
        def import_plan(self, *, db, workspace_id, goal, plan, created_by, config):
            captured["config"] = config
            return types.SimpleNamespace(id=uuid.uuid4(), state=RunState.AWAITING_APPROVAL.value)

    monkeypatch.setattr(am, "get_coordinator_service", lambda: _FakeCoordinator())
    monkeypatch.setattr(am, "_run_to_response", lambda run: {"ok": True})

    body = am.PlanImportRequest(
        goal="Do the thing",
        plan={"tasks": [{"id": 1}]},
        config={
            "origin_chat_id": str(uuid.uuid4()),
            "chat_id": str(uuid.uuid4()),
            "keep": "yes",
        },
    )
    ctx = types.SimpleNamespace(workspace_id=uuid.uuid4(), user=types.SimpleNamespace(id="user_owner"))

    asyncio.run(am.import_mission_plan(body, ctx, MagicMock()))

    cfg = captured["config"]
    assert "origin_chat_id" not in cfg, "import-plan must strip caller origin_chat_id"
    assert "chat_id" not in cfg, "import-plan must strip caller chat_id"
    assert cfg.get("keep") == "yes"


# ===========================================================================
# Layer 2 — post_background_message owner check (PURE; DB query mocked)
# ===========================================================================

def _wire_messenger(monkeypatch, *, owner_int, candidate_chat):
    """Wire post_background_message so its Chat lookup returns ``candidate_chat``
    and the owner resolves to ``owner_int``; capture the Auto-thread fallback and
    the chat the message is actually saved into."""
    state = {"foac_uids": [], "saved_chat_id": None}

    auto_chat = types.SimpleNamespace(id=uuid.uuid4(), user_id=owner_int or -1)

    monkeypatch.setattr(cm, "_resolve_user_int_id", lambda db, cid: owner_int)

    def _foac(db, ws, uid):
        state["foac_uids"].append(uid)
        return auto_chat

    monkeypatch.setattr(cm, "find_or_create_auto_chat", _foac)

    class _FakeChatService:
        def __init__(self, db):
            pass

        def save_message(self, **kw):
            state["saved_chat_id"] = kw.get("chat_id")
            return types.SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr("consumers.chatbot.service.ChatService", _FakeChatService)
    monkeypatch.setattr("services.board_events.notify_chat_event", lambda db, **kw: None)

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = candidate_chat
    state["auto_chat_id"] = str(auto_chat.id)
    return db, state


def test_delivery_rejects_chat_owned_by_another_user(monkeypatch):
    """A workspace-valid chat owned by a DIFFERENT user is rejected — the message
    falls back to the OWNER's Auto thread, never the other user's chat."""
    OWNER, OTHER = 7, 99
    evil_chat = types.SimpleNamespace(id=uuid.uuid4(), user_id=OTHER)
    db, state = _wire_messenger(monkeypatch, owner_int=OWNER, candidate_chat=evil_chat)

    cm.post_background_message(
        db,
        workspace_id=uuid.uuid4(),
        text="Mission complete",
        source={"origin": "mission", "label": "Auto · mission"},
        chat_id=str(evil_chat.id),
        clerk_user_id="user_owner",
        link_type="mission",
        link_id=str(uuid.uuid4()),
    )

    assert state["foac_uids"] == [OWNER], "must fall back to the OWNER's Auto thread"
    assert state["saved_chat_id"] == state["auto_chat_id"]
    assert state["saved_chat_id"] != str(evil_chat.id), "must NOT write the other user's chat"


def test_delivery_honors_chat_owned_by_the_owner(monkeypatch):
    """The owner's own originating chat is honoured — no Auto-thread fallback."""
    OWNER = 7
    own_chat = types.SimpleNamespace(id=uuid.uuid4(), user_id=OWNER)
    db, state = _wire_messenger(monkeypatch, owner_int=OWNER, candidate_chat=own_chat)

    cm.post_background_message(
        db,
        workspace_id=uuid.uuid4(),
        text="Mission complete",
        source={"origin": "mission"},
        chat_id=str(own_chat.id),
        clerk_user_id="user_owner",
    )

    assert state["foac_uids"] == [], "owner's own chat is honoured — no fallback"
    assert state["saved_chat_id"] == str(own_chat.id)


def test_delivery_no_owner_keeps_legacy_workspace_honor(monkeypatch):
    """Regression guard: a producer with NO resolvable owner (agent-created
    scheduled tasks, server-captured origin) keeps the legacy workspace-only
    honour — the message is delivered, not dropped."""
    ws_chat = types.SimpleNamespace(id=uuid.uuid4(), user_id=55)
    db, state = _wire_messenger(monkeypatch, owner_int=None, candidate_chat=ws_chat)

    result = cm.post_background_message(
        db,
        workspace_id=uuid.uuid4(),
        text="Scheduled task output",
        source={"origin": "scheduled_task"},
        chat_id=str(ws_chat.id),
        # no clerk_user_id — scheduled tasks are agent-created
    )

    assert state["saved_chat_id"] == str(ws_chat.id), "no-owner producer keeps legacy honour"
    assert state["foac_uids"] == [], "no Auto-thread fallback when there is no owner"
    assert result is not None


# ===========================================================================
# @integration — real Postgres cross-user round-trip (skips cleanly without a
# DB; CI test.yml is the gate). Mirrors test_prd227_narration_session_isolation.
# ===========================================================================

@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM chats LIMIT 1"))
            c.execute(text("SELECT 1 FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"owner-scope suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def two_users(new_session):
    """Workspace + owner user (with an Auto thread target) + a DIFFERENT user who
    owns a private chat in the SAME workspace. Yields
    ``(ws_id, owner_clerk, other_chat_id)`` — the other user's chat is the
    cross-user injection target a bad origin_chat_id would aim at. Committed so a
    separate narration session sees them (PRD-158: workspaces seeded FIRST)."""
    from sqlalchemy import text

    from core.models.core import Chat, User

    ws_id = str(uuid.uuid4())
    owner_clerk = f"user_owner_{uuid.uuid4().hex[:8]}"
    other_clerk = f"user_other_{uuid.uuid4().hex[:8]}"

    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd227-rvw2"},
    )
    s.commit()

    owner = User(username=owner_clerk, email=f"{owner_clerk}@t.test", clerk_user_id=owner_clerk)
    other = User(username=other_clerk, email=f"{other_clerk}@t.test", clerk_user_id=other_clerk)
    s.add(owner)
    s.add(other)
    s.commit()

    other_chat = Chat(user_id=other.id, workspace_id=uuid.UUID(ws_id), title="other-private")
    s.add(other_chat)
    s.commit()
    other_chat_id = str(other_chat.id)
    s.close()

    yield ws_id, owner_clerk, other_chat_id

    s = new_session.sweep()
    s.execute(text("DELETE FROM messages WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    s.execute(text("DELETE FROM chats WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    s.execute(
        text("DELETE FROM users WHERE clerk_user_id IN (:o, :t)"),
        {"o": owner_clerk, "t": other_clerk},
    )
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws_id})
    s.commit()
    s.close()


def test_narration_never_lands_in_another_users_chat(two_users, new_session, engine):
    """A run whose config.origin_chat_id points at ANOTHER user's workspace chat
    (a value that bypassed layer 1) delivers to the creator's Auto thread — the
    other user's chat receives NOTHING. Drives the REAL messenger via the RVW-1
    independent session."""
    import services.coordinator_service as cs
    from core.models.core import Chat, Message

    ws_id, owner_clerk, other_chat_id = two_users

    run = types.SimpleNamespace(
        id=uuid.uuid4(),
        workspace_id=uuid.UUID(ws_id),
        goal="Cross-user narration probe",
        created_by=owner_clerk,          # the message's true owner
        state=RunState.COMPLETED.value,
        config={"origin_chat_id": other_chat_id},  # attacker-aimed at the other user
        plan={"tasks": [{"id": 1}]},
        stop_detail=None,
        stop_reason=None,
    )

    cs._narrate_run_terminal(new_session(), run)

    probe = new_session()
    # The other user's chat received NOTHING.
    other_msgs = probe.query(Message).filter(Message.chat_id == uuid.UUID(other_chat_id)).count()
    assert other_msgs == 0, "narration must NEVER land in another user's chat"

    # The creator's Auto thread received the line instead.
    auto = (
        probe.query(Chat)
        .filter(Chat.workspace_id == uuid.UUID(ws_id), Chat.kind == "auto")
        .first()
    )
    assert auto is not None, "creator's Auto thread must have been created for the fallback"
    auto_msgs = probe.query(Message).filter(Message.chat_id == auto.id).count()
    assert auto_msgs >= 1, "the narration line must land in the creator's Auto thread"
    probe.close()
