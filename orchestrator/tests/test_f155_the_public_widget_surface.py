"""F155 — what a public widget key reaches.

(h) The widget data plane runs no caller SQL: only the natural-language query,
which the NL2SQL service scopes to the key's workspace, remains.
(d-1) A key's team lock scopes its chat's documents, as it already scopes
/search and /docs; without one, the answering agent's team does. On a widget
turn the lock also scopes every document tool: knowledge and multimodal
search, document reads and listing, the graph, and where an upload lands.
(b)+(e) interim: the chat service marks a widget turn for its duration, and the
predicates every gate shares read the mark, so whatever caller context a tool
call carries, the turn is made for nobody, is never an admin, a super admin or
autonomous, never approves a card by instruction, and is offered no admin tier;
its auto_approve neither starts a mission nor runs a board task's approval, and
it cannot change a mission (approve, resume, reject, pause, cancel, replan or
edit its plan). A mission it starts is stamped widget-born, and its approval is
decided as the widget's even when it is planned later (async_planning).
(c) The widget records the key that starts a conversation; a key reads and
resumes only the conversations it started.
(a), log-only: a chat on a key without an agent lock logs the agent it names,
per key, before a per-key agent allow-list is decided.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

ADMIN_PROBE = "platform_f155_admin_probe"
CARD_PROBE = "platform_f155_card_probe"


def test_the_widget_data_plane_runs_no_caller_sql():
    from api.widgets import data

    assert sorted(route.path for route in data.router.routes) == ["/data/query"]


def test_the_keys_team_lock_scopes_widget_chat(db_session, seed_workspace):
    from api.widgets.chat import _retrieval_team

    agent = db_session.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration, team) "
                                    "VALUES ('Barista', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), 'hq') "
                                    "RETURNING id"), {"w": seed_workspace()}).scalar()
    assert _retrieval_team(db_session, NS(team="franchise-a"), agent) == "franchise-a"
    assert _retrieval_team(db_session, NS(team=None), agent) == "hq"



LOCKED = ("franchise-a", "Franchise-A ")


def _barista(db, ws):
    return db.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration, team) "
                           "VALUES ('Barista', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), 'hq') "
                           "RETURNING id"), {"w": str(ws)}).scalar()


def _locked():
    from core.security.surface import WIDGET, turn_surface

    return turn_surface(WIDGET, ("chat", "documents:read", "documents:write"), LOCKED[1])


def test_the_keys_team_lock_scopes_the_document_and_graph_tools(db_session, seed_workspace):
    from modules.tools.discovery import handlers_documents, handlers_graph

    agent = _barista(db_session, seed_workspace())
    for handlers in (handlers_documents, handlers_graph):
        with _locked():
            assert handlers._resolve_agent_team(db_session, agent) == LOCKED[0]
        assert handlers._resolve_agent_team(db_session, agent) == "hq"


def _document(db, ws, name, teams):
    from core.models import Document

    db.add(Document(workspace_id=ws, filename=name, original_filename=name, file_type="text", file_size=1,
                    file_path=f"/uploads/{name}", content_hash=uuid4().hex, status="completed", team_access=teams))
    db.flush()


def test_a_locked_widget_turn_lists_only_its_teams_documents(db_session, seed_workspace):
    from modules.tools.discovery.handlers_documents import list_documents

    ws = UUID(seed_workspace())
    for name, teams in (("franchise-menu.txt", ["franchise-a"]), ("hq-margins.txt", ["hq"]), ("faq.txt", [])):
        _document(db_session, ws, name, teams)

    def listed():
        reply = asyncio.run(list_documents(db_session, ws, {}))
        return sorted(document["filename"] for document in reply["documents"])

    with _locked():
        assert listed() == ["faq.txt", "franchise-menu.txt"]
    assert listed() == ["faq.txt", "franchise-menu.txt", "hq-margins.txt"]


def test_a_locked_widget_turns_upload_belongs_to_its_team(db_session, seed_workspace, monkeypatch, tmp_path):
    import api.documents as documents
    from modules.tools.discovery.handlers_documents import upload_document

    ws = UUID(seed_workspace())
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(documents, "get_document_manager", lambda w: NS(_process_document=AsyncMock()))
    with _locked():
        reply = asyncio.run(upload_document(db_session, ws, {"filename": "specials.txt", "content": "Soup."}))
    teams = db_session.execute(text("SELECT team_access FROM documents WHERE id = :id"),
                               {"id": reply["document_id"]}).scalar()
    assert teams == [LOCKED[0]]


def test_the_keys_team_lock_scopes_knowledge_search(db_session, seed_workspace):
    from modules.agents.services.agent_platform_tools import AgentPlatformTools

    agent = _barista(db_session, seed_workspace())
    tools = AgentPlatformTools.__new__(AgentPlatformTools)
    tools.db, tools.logger, tools.rag_config = db_session, __import__("logging").getLogger("t"), None
    retrieve = AsyncMock(return_value=NS(chunks=[]))
    tools.rag_service = NS(retrieve_context=retrieve)

    def searched(tool):
        asyncio.run(tools.execute_tool(tool_name=tool, parameters={"query": "margins"}, agent_id=agent))
        return retrieve.call_args.kwargs["team"]

    for tool in ("search_knowledge", "semantic_search"):
        with _locked():
            assert searched(tool) == LOCKED[0]
        assert searched(tool) == "hq"


def test_a_locked_widget_turns_multimodal_search_cannot_name_another_team(monkeypatch):
    from modules.rag.services import multimodal_knowledge_tools as multimodal
    from modules.tools.execution import exec_multimodal

    search = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(multimodal.MultimodalKnowledgeTools, "search_multimodal", search)
    with _locked():
        asyncio.run(exec_multimodal.execute_multimodal_tool(
            NS(db=None), "search_multimodal", {"query": "margins", "team": "hq"}, 9, workspace_id=uuid4()))
    assert search.call_args.kwargs["team"] == LOCKED[0]

# ── (b)+(e) interim: a widget turn is neither admin nor autonomous ─────────

@pytest.fixture
def shop(db_session, seed_workspace, monkeypatch):
    """A workspace, its owner, and two probes: an admin_only action and an
    action a confirmation card holds."""
    from modules.tools.discovery.action_registry import ActionDefinition, get_action_registry

    live = get_action_registry()
    live.get_all()
    empty = {"type": "object", "properties": {}, "required": []}
    monkeypatch.setitem(live._actions, ADMIN_PROBE, ActionDefinition(
        name=ADMIN_PROBE, description="F155 probe", category="t", permission_level="read",
        parameters=empty, admin_only=True))
    monkeypatch.setitem(live._actions, CARD_PROBE, ActionDefinition(
        name=CARD_PROBE, description="F155 probe", category="t", permission_level="write",
        parameters=empty, requires_confirmation=True))
    db = db_session
    ws = UUID(seed_workspace())
    clerk = f"user_{uuid4().hex[:12]}"
    owner = db.execute(text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) RETURNING id"),
                       {"e": f"gerard-{uuid4().hex[:8]}@harbourline.test", "u": f"gerard-{uuid4().hex[:8]}",
                        "c": clerk}).scalar()
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": owner})
    return NS(db=db, ws=ws, owner=owner, owners_turn={"driving_user_id": str(owner), "user_id": clerk,
                                                      "conversation_id": "chat-1"})


def _run(shop, action, caller_context, surface):
    from core.security.surface import turn_surface
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(shop.db, shop.ws)
    handler = AsyncMock(return_value={"success": True, "handler": "ran"})
    executor._handlers[action] = handler
    with turn_surface(surface), \
            patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, {}, caller_context)), handler


def _refused_and_asked(shop, caller_context):
    from core.security.surface import WIDGET

    reply, handler = _run(shop, ADMIN_PROBE, caller_context, WIDGET)
    assert reply.get("permission_denied") is True
    handler.assert_not_called()
    reply, handler = _run(shop, CARD_PROBE, caller_context, WIDGET)
    assert reply.get("requires_confirmation") is True
    handler.assert_not_called()


def _both_run(shop, caller_context):
    for action in (ADMIN_PROBE, CARD_PROBE):
        reply, handler = _run(shop, action, caller_context, None)
        assert reply["success"] is True
        handler.assert_called_once()


def test_a_widget_turn_is_nobodys_whoever_its_context_names(shop):
    _refused_and_asked(shop, shop.owners_turn)
    _both_run(shop, shop.owners_turn)


def test_a_widget_turn_is_not_autonomous_when_the_workspace_is(shop):
    from core.services.auto_autonomy import FULL, set_autonomy_level

    set_autonomy_level(shop.db, shop.ws, FULL)
    visitor = {"conversation_id": "widget-chat-1"}
    _refused_and_asked(shop, visitor)
    _both_run(shop, visitor)


def test_the_shared_predicates_answer_for_nobody_on_a_widget_turn(shop):
    from core.security.driving_user import driver_is_workspace_admin, driving_user_id
    from core.security.surface import WIDGET, turn_surface
    from core.services.auto_autonomy import FULL, is_full_autonomy, set_autonomy_level
    from modules.tools.discovery.platform_executor import _caller_is_super_admin

    set_autonomy_level(shop.db, shop.ws, FULL)
    su = {"system_role": "super_admin"}

    def answers():
        return (driving_user_id(shop.owners_turn), driver_is_workspace_admin(shop.db, shop.ws, shop.owners_turn),
                driver_is_workspace_admin(shop.db, shop.ws, su), _caller_is_super_admin(su),
                is_full_autonomy(shop.db, shop.ws))

    with turn_surface(WIDGET):
        assert answers() == (None, False, False, False, False)
    assert answers() == (shop.owner, True, True, True, True)


def test_a_widget_turn_is_offered_no_admin_tier():
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.tool_router import _resolve_workspace_admin

    with turn_surface(WIDGET):
        assert _resolve_workspace_admin(object(), "ws-1", True, "t-1") is False
    assert _resolve_workspace_admin(object(), "ws-1", True, "t-1") is True


def test_a_turn_closed_from_another_context_leaves_that_contexts_mark():
    """A task the widget turn started inherits the mark; closing the turn from
    that task must not clear it there."""
    import contextvars

    from core.security.surface import WIDGET, turn_surface, widget_turn

    block = turn_surface(WIDGET)
    turn = contextvars.copy_context()
    turn.run(block.__enter__)
    started_by_the_turn = turn.copy()
    started_by_the_turn.run(block.__exit__, None, None, None)
    assert started_by_the_turn.run(widget_turn) is True


@pytest.mark.parametrize("widget_mode", [True, False])
def test_only_a_widget_turn_is_marked_and_only_while_it_runs(widget_mode):
    from consumers.chatbot.service import StreamingChatService
    from core.security.surface import widget_turn

    service = StreamingChatService.__new__(StreamingChatService)
    service.widget_mode = widget_mode
    service.widget_scopes, service.widget_team, service.widget_agent_lock = ("chat",), None, None

    async def _turn(*args, **kwargs):
        yield widget_turn()

    service._stream_response_with_agent_scoped = _turn

    async def _chat():
        chunks = [chunk async for chunk in service.stream_response_with_agent(
            chat_id="chat-1", messages=[], agent_id=9, user_id=1)]
        return chunks, widget_turn()

    assert asyncio.run(_chat()) == ([widget_mode], False)



def test_a_widget_turns_auto_approve_never_starts_a_mission(monkeypatch):
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery import handlers_missions as missions

    monkeypatch.setattr("core.services.approval_policy.load_approval_policy",
                        lambda db, ws: {"policy": "auto_below_budget", "approval_dollar_ceiling": 5.0,
                                        "auto_proceed_after_seconds": None})
    monkeypatch.setattr(missions, "_recent_chat_context", lambda *a, **k: [])
    monkeypatch.setattr("modules.tools.discovery.handlers_watches.auto_create_watch", lambda *a, **k: None)
    run = NS(id=uuid4(), goal="g", state="awaiting_approval", plan={"tasks": [{"title": "a"}]}, config={})

    def create(surface):
        coordinator = NS(create_mission=AsyncMock(return_value=run))
        with turn_surface(surface), patch("services.coordinator_service.CoordinatorService", return_value=coordinator):
            reply = asyncio.run(missions.create_mission(MagicMock(), uuid4(),
                                                        {"goal": "g", "config": {"auto_approve": True}}))
        return reply, coordinator.create_mission.call_args.kwargs["config"]

    reply, config = create(WIDGET)
    assert "auto_approve" not in config and "public widget is no approval" in reply["message"]
    reply, config = create(None)
    assert config["auto_approve"] is True and "not applied" not in reply["message"]


def test_a_widget_turns_auto_approve_never_runs_a_board_approval(db_session, seed_workspace):
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery.handlers_board_tasks import create_board_task

    ws = UUID(seed_workspace())
    published = []

    class _Blog:
        def __init__(self, db, workspace_id):
            pass

        def publish_post(self, post_id):
            published.append(post_id)

    def file(surface):
        params = {"title": "Launch post", "description": "Publish the launch post", "auto_approve": True,
                  "approval_action": {"type": "publish_blog", "post_id": str(uuid4())}}
        with turn_surface(surface), patch("core.services.blog_service.BlogService", _Blog), \
                patch("core.services.notification_service.send_workspace_notification", new=AsyncMock()):
            return asyncio.run(create_board_task(db_session, ws, params))

    reply = file(WIDGET)
    assert (reply["status"], published, reply["auto_approve"]) == (
        "review", [], "not applied: a call from the public widget is no approval")
    reply = file(None)
    assert reply["status"] == "done" and len(published) == 1


EDITS = {"task_edits": [{"task_index": 0, "agent_id": 7}]}


@pytest.mark.parametrize("handler,decision,extra", [("approve_mission", "approve_plan", {}),
                                                    ("resume_mission", "resume_mission", {}),
                                                    ("reject_mission", "reject_plan", {}),
                                                    ("pause_mission", "pause_mission", {}),
                                                    ("cancel_mission", "cancel_mission", {}),
                                                    ("replan_mission", "replan_mission", {}),
                                                    ("update_mission_plan", "update_mission_plan", EDITS)])
def test_a_widget_turn_cannot_change_a_mission(monkeypatch, handler, decision, extra):
    """Under a policy that lets a call approve (auto_below_budget), on a run
    paused at its token budget."""
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery import handlers_missions as missions

    monkeypatch.setattr("core.services.approval_policy.load_approval_policy",
                        lambda db, ws: {"policy": "auto_below_budget", "approval_dollar_ceiling": 5.0,
                                        "auto_proceed_after_seconds": None})
    run = NS(id=uuid4(), state="paused", stop_reason="budget_exhausted", goal="g",
             token_budget_estimate=175_000, tokens_used=429_423, config={})

    def decide(surface):
        coordinator = MagicMock()
        updated = NS(id=run.id, state="running", goal="g", token_budget_estimate=858_846, tokens_used=429_423,
                     config={})
        setattr(coordinator, decision,
                AsyncMock(return_value=updated) if decision == "replan_mission" else MagicMock(return_value=updated))
        with turn_surface(surface), patch.object(missions, "_resolve_run", return_value=(run, None)), \
                patch("services.coordinator_service.CoordinatorService", return_value=coordinator):
            reply = asyncio.run(getattr(missions, handler)(MagicMock(), uuid4(),
                                                           {"mission_id": str(run.id), **extra}))
        return reply, getattr(coordinator, decision)

    reply, called = decide(WIDGET)
    assert reply == {"success": False, "error": missions.WIDGET_CANNOT_CHANGE}
    called.assert_not_called()
    reply, called = decide(None)
    assert reply["success"] is True
    called.assert_called_once()


def test_a_mission_records_the_surface_it_started_from(mock_db):
    from core.security.surface import WIDGET, turn_surface
    from services.coordinator_service import CoordinatorService

    def create(surface, config):
        with turn_surface(surface):
            run = asyncio.run(CoordinatorService().create_mission(db=mock_db, workspace_id=uuid4(), goal="g",
                                                                  created_by="user_test", config=config))
        return run.config

    assert create(WIDGET, {"async_planning": True})["origin_surface"] == WIDGET
    assert "origin_surface" not in create(None, {"async_planning": True, "origin_surface": WIDGET})


@pytest.mark.parametrize("origin,state", [("widget", "awaiting_approval"), (None, "running")])
def test_a_widget_born_mission_planned_later_is_not_autonomous(db_session, seed_workspace, monkeypatch, origin, state):
    """full_auto policy and the full-autonomy dial: a dashboard mission planned
    on the tick auto-runs; a widget-born one waits for the owner."""
    import services.coordinator_service as coordinator
    from core.services.auto_autonomy import FULL, set_autonomy_level
    from modules.coordination.agent_matcher import AgentMatcher
    from modules.coordination.planner import MissionPlanner

    ws = UUID(seed_workspace())
    set_autonomy_level(db_session, ws, FULL)
    monkeypatch.setattr("core.services.approval_policy.load_approval_policy",
                        lambda db, w: {"policy": "full_auto", "approval_dollar_ceiling": None,
                                       "auto_proceed_after_seconds": None})
    monkeypatch.setattr(MissionPlanner, "decompose", AsyncMock(return_value=NS(tasks=[], token_estimate=1000)))
    monkeypatch.setattr(AgentMatcher, "compute_signals_for_tasks", AsyncMock(return_value={}))
    monkeypatch.setattr("services.daily_spend_guard.refuse_new_work", lambda *a, **k: None)
    for name in ("create_mission_board_task", "emit_event"):
        monkeypatch.setattr(coordinator, name, lambda *a, **k: None)
    monkeypatch.setattr(coordinator, "_dispatch_mission_event", AsyncMock())
    states = []
    monkeypatch.setattr(coordinator, "transition_run", lambda **kw: states.append(kw["new_state"].value))
    service = coordinator.CoordinatorService()
    service._persist_decomposition = lambda *a, **k: {}
    service._annotate_match_previews = lambda *a, **k: None
    service._queue_initial_tasks = lambda *a, **k: None
    service._create_mission_field = AsyncMock()
    config = {"async_planning": True, **({"origin_surface": origin} if origin else {})}
    asyncio.run(service._run_planning(db_session, NS(id=uuid4(), workspace_id=ws, goal="g", config=config, plan=None)))
    assert states == [state]

# ── (c): a key reaches only the conversations it started ────────────────────

@pytest.fixture
def site(db_session, seed_workspace):
    """Two widget keys on one workspace, a conversation each started, and a
    dashboard chat, each with one message."""
    from api.widgets.auth import WidgetAuthContext
    from core.models.core import Chat, Message

    db = db_session
    ws = UUID(seed_workspace())
    person = db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                        {"e": f"owner-{uuid4().hex[:8]}@harbourline.test", "u": f"owner-{uuid4().hex[:8]}"}).scalar()
    ours, theirs = uuid4(), uuid4()

    def conversation(key):
        chat = Chat(id=uuid4(), user_id=person, workspace_id=ws, title="t", visibility="private", widget_key_id=key)
        db.add(chat)
        db.flush()
        db.add(Message(chat_id=chat.id, workspace_id=ws, role="user", parts=[{"type": "text", "text": "hello"}]))
        db.flush()
        return str(chat.id)

    return NS(db=db, ws=ws, key_id=ours, key=WidgetAuthContext(workspace_id=ws, api_key_id=ours, permissions=["chat"]),
              ours=conversation(ours), theirs=conversation(theirs), dashboard=conversation(None))


def _history(site, conversation_id):
    from api.widgets.chat import widget_chat_history

    return asyncio.run(widget_chat_history(conversation_id=conversation_id, auth=site.key, db=site.db))


def _send(site, conversation_id=None, agent_id=None, headers=None):
    from api.widgets.chat import WidgetChatRequest, widget_chat

    body = WidgetChatRequest(message="hi", conversation_id=conversation_id, agent_id=agent_id)
    return asyncio.run(widget_chat(body=body, request=NS(headers=headers or {}), auth=site.key, db=site.db))


def test_a_key_reads_only_the_conversations_it_started(site):
    assert [message.content for message in _history(site, site.ours)] == ["hello"]
    for other in (site.theirs, site.dashboard, "not-a-chat"):
        with pytest.raises(HTTPException) as missing:
            _history(site, other)
        assert missing.value.status_code == 404


def test_a_key_resumes_only_the_conversations_it_started(site):
    for other in (site.theirs, site.dashboard):
        with pytest.raises(HTTPException) as missing:
            _send(site, other)
        assert missing.value.status_code == 404


def test_a_conversation_the_widget_starts_records_its_key(site):
    _send(site)
    started = site.db.execute(text("SELECT count(*) FROM chats WHERE workspace_id = CAST(:ws AS uuid) "
                                   "AND widget_key_id = CAST(:key AS uuid)"),
                              {"ws": str(site.ws), "key": str(site.key_id)}).scalar()
    assert started == 2


def test_the_agent_a_keys_visitors_name_is_counted(site, caplog):
    import logging

    agent = _barista(site.db, site.ws)
    with caplog.at_level(logging.INFO, logger="api.widgets.chat"):
        _send(site, agent_id=str(agent))
        _send(site)
    census = [record.getMessage() for record in caplog.records if "AGENT_CENSUS" in record.getMessage()]
    assert len(census) == 1
    assert f"key={site.key_id} named={agent} resolved={agent}" in census[0]


def test_what_a_visitor_sends_cannot_forge_a_log_line(site, caplog):
    import logging

    forged = "\r\n[widget] ADMIN LOGIN ok"
    with caplog.at_level(logging.INFO, logger="api.widgets.chat"), pytest.raises(HTTPException):
        _send(site, agent_id=f"7{forged}", headers={"X-Request-ID": f"r1{forged}", "Origin": f"https://x{forged}"})
    lines = [record.getMessage() for record in caplog.records]
    assert any("REQUEST" in line and "ADMIN LOGIN" in line for line in lines)
    assert not [line for line in lines if "\n" in line or "\r" in line]


def test_widget_auth_logs_what_a_visitor_sends_on_one_line(db_session, caplog):
    import logging

    from api.widgets.auth import widget_auth

    forged = "\r\n[widget] ADMIN LOGIN ok"
    request = NS(headers={"Authorization": f"Bearer ak_pub_x{forged}", "Origin": f"https://x{forged}"}, state=NS())
    with caplog.at_level(logging.WARNING, logger="api.widgets.auth"), pytest.raises(HTTPException):
        asyncio.run(widget_auth(request, db_session))
    lines = [record.getMessage() for record in caplog.records]
    assert any("widget_auth" in line and "ADMIN LOGIN" in line for line in lines)
    assert not [line for line in lines if "\n" in line or "\r" in line]


def test_a_widget_turn_is_marked_with_the_agent_its_key_is_locked_to(site, monkeypatch):
    """Only a locked key's agent brings its own plugins to the turn, so the
    mark carries the lock (resolved to the agent's id); an unlocked key's none."""
    import consumers.chatbot as chatbot

    built, real = [], chatbot.StreamingChatService

    def _recording(*args, **kwargs):
        built.append(kwargs.get("widget_agent_lock"))
        return real(*args, **kwargs)

    monkeypatch.setattr(chatbot, "StreamingChatService", _recording)
    agent = _barista(site.db, site.ws)
    _send(site, agent_id=str(agent))
    site.key.default_agent_id = agent
    _send(site)
    assert built == [None, agent]
