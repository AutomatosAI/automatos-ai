"""PRD-227 P227-RVW-3 — the REST "Launch Mission" (suggestion-card) path must
narrate back INTO the launching chat, not always the Auto thread.

The review finding: US-002 AC3 ("a chat-launched mission narrates into the
launching thread") was met only for the TOOL path (the executor server-injects
``_origin_chat_id`` → handlers_missions sets ``run.config['origin_chat_id']``).
The human "Launch Mission" suggestion card posts ``config={source:'chat',
chat_id:<launching chat>}`` to POST /api/missions; the RVW-2 layer-1 strip
removed that ``chat_id`` and the REST route re-injected NOTHING, so every
``Auto · mission`` line landed in the creator's Auto thread — never the chat the
user launched from.

The fix keeps the RVW-2 discipline (strip the caller keys first) and then
SERVER-sets ``origin_chat_id`` from the AUTHENTICATED request's launching chat —
but only for a Clerk caller whose owner ``post_background_message`` can resolve
(``created_by`` → ``clerk_user_id`` → ``users.id``). That messenger owner check
(layer 2) rejects a chat owned by another user (→ Auto thread), so honouring the
authenticated user's OWN chat is cross-user-safe and does NOT reopen RVW-2. A
non-Clerk caller (API key/SDK — no resolvable owner) gets no origin at all, so a
request key is never honoured workspace-wide.

PURE tests run everywhere (DB layer mocked); the ``@integration`` cross-user
round-trip skips cleanly without local Postgres (CI test.yml is the gate).
"""
from __future__ import annotations

import asyncio
import types
import uuid
from unittest.mock import MagicMock

import pytest

import api.missions as am
import services.chat_messenger as cm
import services.coordinator_service as cs
from core.models.orchestration_enums import RunState


def _clerk_ctx(clerk_id: str = "user_owner"):
    """A Clerk-authenticated RequestContext-shape (id == clerk_user_id, exactly
    as core/auth/hybrid.py builds it: id=info['clerk_user_id'] or email)."""
    return types.SimpleNamespace(
        workspace_id=uuid.uuid4(),
        user=types.SimpleNamespace(id=clerk_id, clerk_user_id=clerk_id),
    )


def _nonclerk_ctx():
    """A non-Clerk caller (API key/SDK): an id but NO resolvable clerk_user_id."""
    return types.SimpleNamespace(
        workspace_id=uuid.uuid4(),
        user=types.SimpleNamespace(id="api_key", clerk_user_id=None),
    )


# ===========================================================================
# Layer 1 — the REST create paths SERVER-SET the origin from the authenticated
# launching chat, and only for a resolvable Clerk caller (PURE)
# ===========================================================================

def _capture_rest_config(monkeypatch, *, is_import=False) -> dict:
    captured: dict = {}

    class _FakeCoordinator:
        async def create_mission(self, *, db, workspace_id, goal, created_by, config):
            captured["config"] = config
            captured["created_by"] = created_by
            return types.SimpleNamespace(id=uuid.uuid4(), state=RunState.AWAITING_APPROVAL.value)

        def import_plan(self, *, db, workspace_id, goal, plan, created_by, config):
            captured["config"] = config
            captured["created_by"] = created_by
            return types.SimpleNamespace(id=uuid.uuid4(), state=RunState.AWAITING_APPROVAL.value)

    monkeypatch.setattr(am, "get_coordinator_service", lambda: _FakeCoordinator())
    monkeypatch.setattr(am, "_run_to_response", lambda run: {"ok": True})
    return captured


def test_rest_create_sets_origin_from_clerk_launching_chat(monkeypatch):
    """A Clerk caller's config['chat_id'] (the suggestion-card launching chat) is
    stripped, then re-set as the SERVER origin_chat_id so the mission narrates
    back into that chat. The raw chat_id key does NOT survive in run.config."""
    captured = _capture_rest_config(monkeypatch)
    launch_chat = str(uuid.uuid4())

    body = am.MissionCreateRequest(
        goal="Ship the Q3 report",
        config={"source": "chat", "chat_id": launch_chat, "keep": "yes"},
    )
    asyncio.run(am.create_mission(body, _clerk_ctx("user_owner"), MagicMock()))

    cfg = captured["config"]
    assert cfg["origin_chat_id"] == launch_chat, "origin must be server-set from the launching chat"
    assert "chat_id" not in cfg, "the raw caller chat_id key must not survive in run.config"
    assert cfg.get("keep") == "yes", "unrelated config keys survive"
    assert captured["created_by"] == "user_owner", "created_by is the authenticated clerk id"


def test_rest_create_prefers_explicit_origin_chat_id_key(monkeypatch):
    """If the request already names origin_chat_id (not just chat_id), the Clerk
    path honours it as the server origin too — still owner-checked at delivery."""
    captured = _capture_rest_config(monkeypatch)
    launch_chat = str(uuid.uuid4())

    body = am.MissionCreateRequest(goal="x", config={"origin_chat_id": launch_chat})
    asyncio.run(am.create_mission(body, _clerk_ctx("user_owner"), MagicMock()))

    assert captured["config"]["origin_chat_id"] == launch_chat


def test_rest_create_no_origin_for_non_clerk_caller(monkeypatch):
    """A non-Clerk caller (no resolvable owner) gets NO origin — the request key is
    never honoured workspace-wide (that would reopen RVW-2 via the messenger's
    no-owner legacy branch). Narration falls back to the Auto thread."""
    captured = _capture_rest_config(monkeypatch)

    body = am.MissionCreateRequest(
        goal="x", config={"source": "chat", "chat_id": str(uuid.uuid4())},
    )
    asyncio.run(am.create_mission(body, _nonclerk_ctx(), MagicMock()))

    assert "origin_chat_id" not in captured["config"], "no origin for a non-Clerk caller"
    assert "chat_id" not in captured["config"], "caller chat_id still stripped"


def test_import_plan_sets_origin_from_clerk_launching_chat(monkeypatch):
    """/import-plan parity: a Clerk caller's launching chat becomes the server
    origin (stripped-then-set), so a chat-edited plan narrates into that thread."""
    captured = _capture_rest_config(monkeypatch, is_import=True)
    launch_chat = str(uuid.uuid4())

    body = am.PlanImportRequest(
        goal="x",
        plan={"tasks": [{"id": 1}]},
        config={"source": "chat", "chat_id": launch_chat, "keep": "yes"},
    )
    asyncio.run(am.import_mission_plan(body, _clerk_ctx("user_owner"), MagicMock()))

    cfg = captured["config"]
    assert cfg["origin_chat_id"] == launch_chat
    assert "chat_id" not in cfg
    assert cfg.get("keep") == "yes"


def test_import_plan_no_origin_for_non_clerk_caller(monkeypatch):
    """/import-plan guard parity: no origin for a non-Clerk caller."""
    captured = _capture_rest_config(monkeypatch, is_import=True)

    body = am.PlanImportRequest(
        goal="x", plan={"tasks": [{"id": 1}]},
        config={"chat_id": str(uuid.uuid4())},
    )
    asyncio.run(am.import_mission_plan(body, _nonclerk_ctx(), MagicMock()))

    assert "origin_chat_id" not in captured["config"]
    assert "chat_id" not in captured["config"]


# ===========================================================================
# REST-create → delivery, composed (PURE; DB layer mocked). Proves the origin the
# REST path sets is delivered per the layer-2 owner check: OWN chat honoured,
# ANOTHER user's chat → Auto thread. Neither reintroduces a cross-user origin.
# ===========================================================================

def _wire_delivery(monkeypatch, *, owner_int, candidate_chat):
    """Wire the messenger + the RVW-1 independent SessionLocal so a narration
    delivery resolves ``owner_int`` and finds ``candidate_chat`` for its chat_id;
    capture the Auto-thread fallback uids and the chat actually saved into."""
    state = {"foac_uids": [], "saved_chat_id": None}
    auto_chat = types.SimpleNamespace(id=uuid.uuid4(), user_id=owner_int or -1)
    state["auto_chat_id"] = str(auto_chat.id)

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

    narration_db = MagicMock(name="narration_db")
    narration_db.query.return_value.filter.return_value.first.return_value = candidate_chat
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: narration_db)
    return state


def _create_then_narrate(monkeypatch, *, clerk_id, launch_chat, state):
    """Drive the REST create (Clerk caller) to build run.config, then narrate the
    resulting run's terminal state through the REAL coordinator narration path."""
    captured: dict = {}

    class _FakeCoordinator:
        async def create_mission(self, *, db, workspace_id, goal, created_by, config):
            run = types.SimpleNamespace(
                id=uuid.uuid4(),
                workspace_id=workspace_id,
                goal=goal,
                created_by=created_by,
                state=RunState.COMPLETED.value,
                config=config or {},
                plan={"tasks": [{"id": 1}]},
                stop_detail=None,
                stop_reason=None,
            )
            captured["run"] = run
            return run

    monkeypatch.setattr(am, "get_coordinator_service", lambda: _FakeCoordinator())
    monkeypatch.setattr(am, "_run_to_response", lambda run: {"ok": True})

    ctx = types.SimpleNamespace(
        workspace_id=uuid.uuid4(),
        user=types.SimpleNamespace(id=clerk_id, clerk_user_id=clerk_id),
    )
    body = am.MissionCreateRequest(
        goal="Ship it", config={"source": "chat", "chat_id": launch_chat},
    )
    asyncio.run(am.create_mission(body, ctx, MagicMock()))

    run = captured["run"]
    cs._narrate_run_terminal(MagicMock(name="coordinator_db"), run)
    return run


def test_rest_own_chat_narration_targets_that_chat(monkeypatch):
    """POST /api/missions with the creator's OWN launching chat → run.config's
    origin is that chat AND the narration line is delivered INTO it (no Auto
    fallback)."""
    OWNER = 7
    launch_chat = str(uuid.uuid4())
    own = types.SimpleNamespace(id=uuid.UUID(launch_chat), user_id=OWNER)
    state = _wire_delivery(monkeypatch, owner_int=OWNER, candidate_chat=own)

    run = _create_then_narrate(monkeypatch, clerk_id="user_owner", launch_chat=launch_chat, state=state)

    assert run.config["origin_chat_id"] == launch_chat, "REST path set the origin"
    assert state["foac_uids"] == [], "owner's own launching chat is honoured — no Auto fallback"
    assert state["saved_chat_id"] == launch_chat, "narration landed in the launching chat"


def test_rest_other_users_chat_falls_back_to_auto(monkeypatch):
    """POST /api/missions naming ANOTHER user's chat → the layer-2 owner check
    rejects it at delivery; the line lands in the creator's Auto thread, NEVER the
    other user's chat. The origin the REST path set was still owner-checked."""
    OWNER, OTHER = 7, 99
    evil_chat_id = str(uuid.uuid4())
    evil = types.SimpleNamespace(id=uuid.UUID(evil_chat_id), user_id=OTHER)
    state = _wire_delivery(monkeypatch, owner_int=OWNER, candidate_chat=evil)

    run = _create_then_narrate(monkeypatch, clerk_id="user_owner", launch_chat=evil_chat_id, state=state)

    assert run.config["origin_chat_id"] == evil_chat_id, "REST set it, delivery must gate it"
    assert state["foac_uids"] == [OWNER], "must fall back to the OWNER's Auto thread"
    assert state["saved_chat_id"] == state["auto_chat_id"]
    assert state["saved_chat_id"] != evil_chat_id, "must NOT write the other user's chat"


# ===========================================================================
# @integration — real Postgres cross-user round-trip via the REST create path
# (skips cleanly without a DB; CI test.yml is the gate).
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
        pytest.skip(f"RVW-3 REST-origin suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def two_users(new_session):
    """Workspace + owner (with an Auto thread target) + a DIFFERENT user owning a
    private chat in the SAME workspace. Yields ``(ws_id, owner_clerk,
    other_chat_id)``. Committed so a separate narration session sees them."""
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
        {"id": ws_id, "n": "prd227-rvw3"},
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


def test_rest_cross_user_launch_never_lands_in_another_users_chat(two_users, new_session, engine, monkeypatch):
    """End-to-end via the REST path: an authenticated owner POSTs a mission whose
    config.chat_id is ANOTHER user's workspace chat. The REST path server-sets
    origin_chat_id (it cannot know ownership at create time), but the messenger's
    layer-2 owner check delivers to the creator's Auto thread — the other user's
    chat receives NOTHING. Mirrors test_prd227_narration_target_owner's assertion,
    driven from the REST create path (real messenger, RVW-1 independent session)."""
    from core.models.core import Chat, Message

    ws_id, owner_clerk, other_chat_id = two_users

    captured: dict = {}

    class _FakeCoordinator:
        async def create_mission(self, *, db, workspace_id, goal, created_by, config):
            captured["run"] = types.SimpleNamespace(
                id=uuid.uuid4(),
                workspace_id=uuid.UUID(ws_id),
                goal=goal,
                created_by=created_by,  # the authenticated owner clerk id
                state=RunState.COMPLETED.value,
                config=config or {},
                plan={"tasks": [{"id": 1}]},
                stop_detail=None,
                stop_reason=None,
            )
            return captured["run"]

    monkeypatch.setattr(am, "get_coordinator_service", lambda: _FakeCoordinator())
    monkeypatch.setattr(am, "_run_to_response", lambda run: {"ok": True})

    ctx = types.SimpleNamespace(
        workspace_id=uuid.UUID(ws_id),
        user=types.SimpleNamespace(id=owner_clerk, clerk_user_id=owner_clerk),
    )
    body = am.MissionCreateRequest(
        goal="Cross-user REST probe",
        config={"source": "chat", "chat_id": other_chat_id},  # aimed at the other user
    )
    asyncio.run(am.create_mission(body, ctx, MagicMock()))

    run = captured["run"]
    assert run.config["origin_chat_id"] == other_chat_id, "REST set the origin; delivery must gate it"

    cs._narrate_run_terminal(new_session(), run)

    probe = new_session()
    other_msgs = probe.query(Message).filter(Message.chat_id == uuid.UUID(other_chat_id)).count()
    assert other_msgs == 0, "narration must NEVER land in another user's chat"

    auto = (
        probe.query(Chat)
        .filter(Chat.workspace_id == uuid.UUID(ws_id), Chat.kind == "auto")
        .first()
    )
    assert auto is not None, "creator's Auto thread must have been created for the fallback"
    auto_msgs = probe.query(Message).filter(Message.chat_id == auto.id).count()
    assert auto_msgs >= 1, "the narration line must land in the creator's Auto thread"
    probe.close()
