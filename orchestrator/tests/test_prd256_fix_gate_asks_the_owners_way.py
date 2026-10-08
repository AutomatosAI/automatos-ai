"""P256-FIX-RVW-9 (fix-wave review, MEDIUM): an editor's 'delete <agent>' card names its
agent, and the click deletes that agent.

platform_delete_agent requires confirmation, so for an editor (no instructing-admin pass)
the confirmation gate in PlatformActionExecutor.clear raised its own card with the raw
``agent_name`` before owner_only.platform_ask ran: no agent bound, no FX-008 question. The
click resumed 'agent_name=market', and delete_agent deleted the first agent whose name
contained it. The gate now asks the owner's way (modules/tools/discovery/confirmation_gate),
and a call naming its agent by name alone is never run on a grant: on any lane its ask
binds the agent's id, and the click runs on that id.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

DELETE = "platform_delete_agent"
CLICKER = "user:clicker-7"
EDITOR, OWNER = "editor", "owner"


def _agent(db, ws, name, *, system=False):
    from core.models.core import Agent

    agent = Agent(name=name, agent_type="system" if system else "chatbot", description="", status="active",
                  configuration={}, workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws),
                  is_system_agent=system)
    db.add(agent)
    db.flush()
    return agent.id


@pytest.fixture
def market(db_session, seed_workspace, monkeypatch):
    """A workspace with Auto (the chat's actor) and MARKET-MANAGER; the person typing holds
    the role ``market.role["is"]`` (an editor unless a test says otherwise); the dial off."""
    from modules.tools.discovery import platform_executor as pe
    from modules.tools.execution import tool_grants

    ws = UUID(seed_workspace())
    role = {"is": EDITOR}
    monkeypatch.setattr(pe, "_workspace_role_for_clerk", lambda db, workspace_id, clerk_id: role["is"])
    monkeypatch.setattr(tool_grants, "_notify_approval_pending", lambda *a, **k: None)
    executor = pe.PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: False
    return NS(db=db_session, ws=ws, role=role, executor=executor, auto=_agent(db_session, ws, "Auto", system=True),
              manager=_agent(db_session, ws, "MARKET-MANAGER"))


def _chat():
    return {"driving_user_id": "7", "user_id": "user_typing", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _call(market, params, caller_context):
    """The call as the chat makes it: exec_platform injects the running agent (Auto) as ``_agent_id``."""
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(market.executor.execute(DELETE, {**params, "_agent_id": market.auto}, caller_context))


def _grants(market):
    from core.models.approval_grants import ApprovalGrant

    return market.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == market.ws).all()


def _exists(market, agent_id):
    from core.models.core import Agent

    return market.db.get(Agent, agent_id) is not None


def test_a_name_two_agents_carry_is_refused_naming_both_and_nothing_is_asked(market):
    marketing = _agent(market.db, market.ws, "MARKETING")

    reply = _call(market, {"agent_name": "market"}, _chat())

    assert reply["success"] is False and "requires_confirmation" not in reply
    assert f"{market.manager}:MARKET-MANAGER" in reply["error"] and f"{marketing}:MARKETING" in reply["error"]
    assert _grants(market) == []
    assert _exists(market, market.manager) and _exists(market, marketing)


def test_one_match_the_editors_card_names_the_agent_by_id_and_says_deleted_for_good(market):
    from core.models.approval_grants import ApprovalGrant

    reply = _call(market, {"agent_name": "market"}, _chat())

    assert reply["requires_confirmation"] is True and reply["owner_only"] is True
    assert reply["params"]["agent_id"] == market.manager
    assert f"'MARKET-MANAGER' (agent #{market.manager}) is deleted for good" in reply["question_md"]
    grant = market.db.get(ApprovalGrant, reply["grant_id"])
    assert grant.details["params"]["agent_id"] == market.manager
    assert _exists(market, market.manager)                                      # nothing done yet


def test_the_click_deletes_the_bound_id_even_after_another_market_agent_is_made(market):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    asked = _call(market, {"agent_name": "market"}, _chat())
    marketing = _agent(market.db, market.ws, "MARKETING")                       # made after the card
    grant = market.db.get(ApprovalGrant, asked["grant_id"])
    grant_grant(grant, granted_by=CLICKER)
    market.db.flush()

    resumed = dict(grant.details["params"])                                     # the resume re-sends the stored call
    ran = _call(market, resumed, _chat())

    assert resumed["agent_id"] == market.manager
    assert ran["success"] is True and ran["deleted_agent"] == {"id": market.manager, "name": "MARKET-MANAGER"}
    assert not _exists(market, market.manager) and _exists(market, marketing)
    assert market.db.get(ApprovalGrant, asked["grant_id"]).status == "revoked"  # one click, one run


@pytest.mark.parametrize("lane", ["editor", "owner", "agent-run"])
def test_a_grant_naming_the_agent_by_name_alone_is_never_the_click(market, lane):
    """A yes on 'agent_name=market' (a card raised before the fix) runs nothing: the call is
    bound and asked about again, whoever drives it."""
    from core.models.approval_grants import GrantStatus
    from core.services.approval_grants import grant_grant
    from modules.tools.execution.tool_grants import issue_tool_grant

    market.role["is"] = OWNER if lane == "owner" else EDITOR
    caller_context = None if lane == "agent-run" else _chat()
    named = {"agent_name": "market", "_agent_id": market.auto}
    stale = issue_tool_grant(market.db, market.ws, action=DELETE, params=named, permission_level="destructive",
                             description="Delete an agent", caller_context=caller_context)
    grant_grant(stale, granted_by=CLICKER)
    market.db.flush()

    reply = _call(market, {"agent_name": "market"}, caller_context)

    assert reply["requires_confirmation"] is True and reply["params"]["agent_id"] == market.manager
    assert reply["grant_id"] != stale.id and stale.status == GrantStatus.GRANTED.value
    assert _exists(market, market.manager)


def test_a_card_from_an_agents_run_binds_its_agent_too(market):
    """An agent's own run keeps the plain card (US-004 leaves its lane unchanged), now on the id."""
    reply = _call(market, {"agent_name": "MARKET-MANAGER"}, None)

    assert reply["requires_confirmation"] is True and "owner_only" not in reply
    assert reply["params"]["agent_id"] == market.manager
    assert reply["message"].startswith("This action (destructive) requires confirmation.")


def test_the_owner_only_names_agree():
    """Every action that binds an agent is owner-only, and a call with an id is never 'by name alone'."""
    from modules.tools.discovery.agent_binding import BINDS, names_the_agent_alone
    from modules.tools.discovery.owner_only import OWNER_ONLY_ACTIONS

    assert BINDS <= OWNER_ONLY_ACTIONS
    assert names_the_agent_alone(DELETE, {"agent_name": "market"}) is True
    assert names_the_agent_alone(DELETE, {"agent_name": "market", "agent_id": 4}) is False
    assert names_the_agent_alone(DELETE, {"agent_name": "  "}) is False
    assert names_the_agent_alone("platform_delete_playbook", {"agent_name": "market"}) is False
    assert names_the_agent_alone(DELETE, "agent_name=market") is False


def test_a_gate_that_fails_falls_closed_to_the_ask_and_deletes_nothing(market, monkeypatch):
    from modules.tools.execution import tool_grants

    def boom(*args, **kwargs):
        raise RuntimeError("grant store down")

    monkeypatch.setattr(tool_grants, "consume_tool_grant", boom)
    reply = _call(market, {"agent_id": market.manager}, _chat())

    assert reply["requires_confirmation"] is True and reply["permission_level"] == "unknown"
    assert "Could not verify permissions" in reply["message"]
    assert _exists(market, market.manager)


def test_with_the_dial_on_an_agents_run_never_deletes_the_first_partial_match(market):
    """No card on that lane, so the handler itself reads the name as one agent's."""
    market.executor._full_autonomy = lambda: True
    marketing = _agent(market.db, market.ws, "MARKETING")

    refused = _call(market, {"agent_name": "market"}, None)
    assert refused["success"] is False and f"{marketing}:MARKETING" in refused["error"]
    assert _exists(market, market.manager) and _exists(market, marketing)

    ran = _call(market, {"agent_name": "marketing"}, None)                      # the whole name wins
    assert ran["success"] is True and ran["deleted_agent"]["id"] == marketing
    assert _exists(market, market.manager) and not _exists(market, marketing)
