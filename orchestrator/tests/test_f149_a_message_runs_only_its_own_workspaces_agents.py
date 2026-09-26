"""F149 — a message runs only its own workspace's agents and playbooks.

Routing rules, webhook overrides and cached routing decisions carry bare ids.
The router drops a decision whose agent or playbook is not in the message's
workspace; the channel, webhook and trigger executors check again before they
run one; and a routing rule, a channel's default agent and an API key's agent
lock refuse another workspace's agent when they are written.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

NOT_OURS = "default_agent_id is not an agent of this workspace"


def _agent(db, ws, name):
    return db.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
                           "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
                      {"n": name, "w": str(ws), "c": '{"runtime": "api"}'}).scalar()


def _playbook(db, ws):
    from core.models.core import WorkflowTemplate

    playbook = WorkflowTemplate(template_id=f"f149-{uuid4().hex[:8]}", name="Nightly till report",
                                description="Summarise the till.", workspace_id=ws, owner_type="workspace",
                                owner_id=str(ws), created_by="priya@harbourline.test", steps=[{"order": 1}],
                                template_definition={"steps": [], "agents": [], "config": {}, "variables": []})
    db.add(playbook)
    db.flush()
    return playbook.id


@pytest.fixture
def cafes(db_session, seed_workspace):
    db = db_session
    ours, theirs = UUID(seed_workspace()), UUID(seed_workspace())
    return NS(db=db, ours=ours, theirs=theirs, our_agent=_agent(db, ours, "Harbour Barista"),
              their_agent=_agent(db, theirs, "Rival Barista"), their_playbook=_playbook(db, theirs))


def _envelope(cafes, **override):
    from core.models.routing import ChannelSource, RequestEnvelope, RequestUser

    return RequestEnvelope(source=ChannelSource.WEBHOOK, content="what's on the menu today?",
                           user=RequestUser(auth_type="webhook"), workspace_id=cafes.ours, **override)


def test_the_scope_checks_fail_closed(cafes):
    from core.security.workspace_scope import agent_in_workspace, playbook_in_workspace

    assert agent_in_workspace(cafes.db, cafes.our_agent, cafes.ours) is True
    for agent_id, ws in ((cafes.their_agent, cafes.ours), (10**9, cafes.ours), (True, cafes.ours),
                         ("latte", cafes.ours), (cafes.our_agent, None)):
        assert agent_in_workspace(cafes.db, agent_id, ws) is False
    assert playbook_in_workspace(cafes.db, cafes.their_playbook, cafes.theirs) is True
    assert playbook_in_workspace(cafes.db, cafes.their_playbook, cafes.ours) is False


# ── the router ──────────────────────────────────────────────────────────────

def _router(cafes):
    from core.routing.engine import UniversalRouter

    router = UniversalRouter(cafes.db, cache=None)
    router._tier2_5_semantic = AsyncMock(return_value=(None, []))
    router._tier2c_intent_classifier = lambda envelope: None
    router._classify_with_llm = AsyncMock(return_value=None)
    router._store_unrouted_event = lambda *args, **kwargs: None
    return router


def test_the_router_drops_another_workspaces_agent(cafes):
    assert asyncio.run(_router(cafes).route(_envelope(cafes, override_agent_id=cafes.their_agent))) is None


def test_the_router_keeps_its_own_workspaces_agent(cafes):
    decision = asyncio.run(_router(cafes).route(_envelope(cafes, override_agent_id=cafes.our_agent)))
    assert decision.agent_id == cafes.our_agent


@pytest.mark.parametrize("target", ["agent", "playbook"])
def test_a_rule_naming_another_workspaces_target_is_dropped(cafes, target):
    from core.models.routing import RoutingRule

    cafes.db.add(RoutingRule(workspace_id=cafes.ours, source_pattern="webhook", intent_keywords=[], priority=1,
                             is_active=True, target_agent_id=cafes.their_agent if target == "agent" else None,
                             target_workflow_id=cafes.their_playbook if target == "playbook" else None))
    cafes.db.flush()
    assert asyncio.run(_router(cafes).route(_envelope(cafes))) is None


# ── the executors ───────────────────────────────────────────────────────────

class _Session:
    """The test's session, handed to code that opens and closes its own."""

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def close(self):
        pass


def test_a_channel_never_runs_another_workspaces_agent(cafes):
    from channels.base import BaseChannelAdapter
    from core.models.routing import RoutingDecision

    class _Till(BaseChannelAdapter):
        async def start(self): ...
        async def stop(self): ...
        async def test_connection(self): return {"ok": True, "detail": ""}

        async def send_message(self, channel_id, text, **kwargs):
            sent.append(text)
            return True

        def _to_envelope(self, message):
            return _envelope(cafes)

    sent = []
    stale = RoutingDecision(route_type="agent", agent_id=cafes.their_agent, confidence=1.0, reasoning="a stale rule")
    with patch("core.database.database.SessionLocal", return_value=_Session(cafes.db)), \
            patch("core.routing.engine.UniversalRouter.route", AsyncMock(return_value=stale)), \
            patch("modules.agents.factory.agent_factory.AgentFactory") as factory:
        asyncio.run(_Till("c-1", str(cafes.ours), {}).handle_message({"channel_id": "till", "text": "menu?"}))
    factory.assert_not_called()
    assert sent == ["I'm not sure how to handle that request. Please try rephrasing."]


def test_a_webhook_never_runs_another_workspaces_agent(cafes):
    import api.webhooks as webhooks

    with patch.object(webhooks, "get_db", lambda: iter([_Session(cafes.db)])), \
            patch("modules.agents.factory.agent_factory.AgentFactory") as factory:
        reply = asyncio.run(webhooks._execute_agent_sync(agent_id=cafes.their_agent, content="menu?", metadata={},
                                                         workspace_id=cafes.ours))
    assert reply == {"status": "error", "error": "Agent not found"}
    factory.assert_not_called()


def test_a_webhook_never_runs_another_workspaces_playbook(cafes):
    import api.webhooks as webhooks

    with patch("services.playbook_engine.get_playbook_engine") as engine:
        result = asyncio.run(webhooks._dispatch_workflow_async(cafes.their_playbook, _envelope(cafes), cafes.db))
    assert result == "no_recipe_found"
    engine.assert_not_called()


def test_a_trigger_never_runs_another_workspaces_agent(cafes):
    import api.composio as composio

    @contextmanager
    def _session():
        yield cafes.db

    with patch.object(composio, "get_db_session", _session), \
            patch("modules.agents.factory.agent_factory.AgentFactory") as factory:
        asyncio.run(composio._dispatch_agent(agent_id=cafes.their_agent, content="menu?", metadata={},
                                             workspace_id=cafes.ours))
    factory.assert_not_called()


# ── what is written ─────────────────────────────────────────────────────────

def test_a_routing_rule_names_only_this_workspaces_targets(cafes):
    from modules.tools.discovery.handlers_routing import create_routing_rule

    refused = asyncio.run(create_routing_rule(cafes.db, cafes.ours, {"target_agent_id": cafes.their_agent,
                                                                     "source_pattern": "webhook"}))
    assert refused == {"success": False, "error": "target_agent_id is not an agent of this workspace"}
    made = asyncio.run(create_routing_rule(cafes.db, cafes.ours, {"target_agent_id": cafes.our_agent,
                                                                  "source_pattern": "webhook"}))
    assert made["success"] is True


def test_the_rest_rule_refuses_another_workspaces_playbook(cafes):
    import api.routing as routing

    body = routing.CreateRuleRequest(source_pattern="webhook", target_workflow_id=cafes.their_playbook)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(routing.create_rule(body=body, ctx=NS(workspace_id=cafes.ours), db=cafes.db))
    assert (refused.value.status_code, refused.value.detail) == (400, "target_workflow_id is not a playbook of this workspace")


def test_a_channel_is_connected_only_with_its_own_workspaces_agent(cafes):
    from api.channels import connect_channel_for_workspace

    with pytest.raises(ValueError, match=NOT_OURS):
        asyncio.run(connect_channel_for_workspace(cafes.db, str(cafes.ours), "telegram", {"bot_token": "123:abc"},
                                                  default_agent_id=cafes.their_agent))


def _channel(cafes):
    channel_id = str(uuid4())
    cafes.db.execute(text("INSERT INTO channel_connections (id, workspace_id, platform, config) "
                          "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), 'telegram', CAST('{}' AS json))"),
                     {"id": channel_id, "ws": str(cafes.ours)})
    return channel_id


def test_the_tool_refuses_another_workspaces_default_agent(cafes):
    from modules.tools.discovery.handlers_channels import configure_channel

    channel_id = _channel(cafes)
    reply = asyncio.run(configure_channel(cafes.db, cafes.ours, {"channel_id": channel_id,
                                                                 "default_agent_id": cafes.their_agent}))
    assert reply == {"success": False, "error": NOT_OURS}


def test_an_api_key_is_locked_only_to_its_own_workspaces_agent(cafes):
    import api.api_keys as api_keys

    body = api_keys.ApiKeyCreateRequest(name="till", key_type="server", permissions=["chat"],
                                        default_agent_id=cafes.their_agent)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(api_keys.create_api_key(body=body, ctx=NS(workspace_id=cafes.ours), db=cafes.db))
    assert (refused.value.status_code, refused.value.detail) == (400, "default_agent_id is not an agent of this workspace")


def test_the_rest_channel_update_refuses_another_workspaces_default_agent(cafes):
    import api.channels as channels

    with pytest.raises(HTTPException) as refused:
        asyncio.run(channels.update_channel(channel_id=_channel(cafes), payload={"default_agent_id": cafes.their_agent},
                                            ctx=NS(workspace_id=cafes.ours), db=cafes.db))
    assert (refused.value.status_code, refused.value.detail) == (400, NOT_OURS)


@pytest.mark.parametrize("surface", ["routing", "composio"])
def test_a_trigger_subscription_names_only_this_workspaces_targets(cafes, surface):
    if surface == "routing":
        import api.routing as routing

        call = routing.setup_trigger(body=routing.TriggerSetupRequest(agent_id=cafes.their_agent),
                                     ctx=NS(workspace_id=cafes.ours), db=cafes.db)
    else:
        import api.composio as composio

        call = composio.subscribe_to_trigger(
            request=composio.TriggerSubscriptionRequest(trigger_name="JIRA_NEW_ISSUE_TRIGGER", agent_id=cafes.their_agent),
            ctx=NS(workspace_id=cafes.ours), db=cafes.db)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(call)
    assert (refused.value.status_code, refused.value.detail) == (400, "agent_id is not an agent of this workspace")


# ── what serves every workspace ─────────────────────────────────────────────

def test_a_platform_system_agent_serves_every_workspace_and_a_template_none(cafes):
    from core.security.workspace_scope import agent_in_workspace

    def _shared(name, system):
        return cafes.db.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration, "
                                     "is_system_agent) VALUES (:n, 'custom', NULL, 'active', CAST('{}' AS json), :s) "
                                     "RETURNING id"), {"n": name, "s": system}).scalar()

    assert agent_in_workspace(cafes.db, _shared("CTO", True), cafes.ours) is True
    assert agent_in_workspace(cafes.db, _shared("Storefront template", False), cafes.ours) is False


def test_a_code_registered_playbook_serves_every_workspace(cafes):
    from modules.workflows.recipes import get_recipe

    assert get_recipe(workflow_id=9000) is not None
    decision = asyncio.run(_router(cafes).route(_envelope(cafes, override_workflow_id=9000)))
    assert (decision.route_type, decision.workflow_id) == ("workflow", 9000)


def test_a_cached_decision_for_another_workspaces_agent_is_dropped(cafes):
    from core.models.routing import RoutingDecision

    router = _router(cafes)
    router._cache = NS(get=lambda ws, content, source: RoutingDecision(
        route_type="agent", agent_id=cafes.their_agent, confidence=1.0, reasoning="cached"))
    assert asyncio.run(router.route(_envelope(cafes))) is None


def test_a_trigger_never_runs_another_workspaces_playbook(cafes):
    import api.composio as composio

    @contextmanager
    def _session():
        yield cafes.db

    with patch.object(composio, "get_db_session", _session), \
            patch("services.playbook_engine.get_playbook_engine") as engine, \
            patch("api.workflows.execute_workflow_with_progress", AsyncMock()) as standard:
        asyncio.run(composio._dispatch_workflow(workflow_id=cafes.their_playbook, envelope=_envelope(cafes)))
    engine.assert_not_called()
    standard.assert_not_awaited()


# ── what a person picks ─────────────────────────────────────────────────────

def test_a_chosen_agent_is_this_workspaces_own(cafes):
    from api.chat import _explicitly_chosen_agent

    assert _explicitly_chosen_agent(cafes.db, cafes.ours, cafes.our_agent) == cafes.our_agent
    with pytest.raises(HTTPException) as refused:
        _explicitly_chosen_agent(cafes.db, cafes.ours, cafes.their_agent)
    assert refused.value.status_code == 404


def test_a_call_answers_only_with_its_own_workspaces_agent(cafes):
    from api.voice_retell import _named_call_agent

    assert _named_call_agent(cafes.db, cafes.ours, str(cafes.our_agent)) == cafes.our_agent
    assert _named_call_agent(cafes.db, cafes.ours, str(cafes.their_agent)) is None
    assert _named_call_agent(cafes.db, cafes.ours, "latte") is None


# ── the factory (log-only until the logs are clean) ─────────────────────────

@pytest.mark.parametrize("whose", ["theirs", "ours"])
def test_activating_another_workspaces_agent_is_logged_not_refused(cafes, caplog, whose):
    from modules.agents.factory.agent_factory import AgentFactory

    factory = AgentFactory.__new__(AgentFactory)
    factory.db_session, factory.active_agents = cafes.db, {}
    factory.logger = logging.getLogger("f149.factory")
    agent = cafes.their_agent if whose == "theirs" else cafes.our_agent
    with patch("modules.agents.queries.get_agent_with_context", return_value=None) as load, \
            caplog.at_level(logging.WARNING, logger="f149.factory"):
        asyncio.run(factory.activate_agent(agent, workspace_id=cafes.ours))
    assert ("[F149] agent" in caplog.text) is (whose == "theirs")
    load.assert_called_once()
