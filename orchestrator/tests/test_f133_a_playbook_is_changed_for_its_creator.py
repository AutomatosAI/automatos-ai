"""F133 (night 4, B9) — a playbook is changed only for its creator, or for a
workspace owner or admin.

B9: every platform_add_playbook_step by agent 295 (Green Coffee Stock Monitor)
was refused as ``unresolved_owner``. The hierarchy check looked for the
playbook's owner in ``workflow_recipes.created_by_agent_id``, a column that
never existed; the error was swallowed, so every non-system agent was refused
on every playbook, and the old test passed on a sqlite table that invented it.
Gerard's rule (25 Sep): an agent may change a playbook while the call is made
for its creator (``created_by_user_id``) or for a workspace owner/admin, read
fresh from workspace_members, or for the server-side super_admin role. Anything
else is refused, fail-closed, with its real cause and a way to get it done. Both
the creator and the roles come from the server-built caller context, never the
tool call's params. The chat's create paths now record the creator.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from core.models import Agent
from core.models.core import WorkflowTemplate
from core.security import hierarchy_permissions as hp
from core.security.hierarchy_permissions import TARGET_PLAYBOOK, can_actor_modify

HOW = "Ask for it in chat, or ask Auto to make it."


def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


def _member(db, ws, user, role, active=True):
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, :role, :active)"),
               {"ws": str(ws), "user": user, "role": role, "active": active})


def _agent(db, ws, name, *, system=False):
    agent = Agent(name=name, agent_type="worker", description="", status="active", configuration={},
                  model_config=None, workspace_id=ws, created_by="test", owner_type="workspace",
                  owner_id=str(ws), is_system_agent=system)
    db.add(agent)
    db.flush()
    return agent.id


def _playbook(db, ws, creator):
    playbook = WorkflowTemplate(template_id=f"f133-{uuid4().hex[:8]}", name="Monday green-coffee reorder",
                                description="Reorder what is low.", workspace_id=ws, owner_type="workspace",
                                owner_id=str(ws), created_by="gerard@harbourline.test",
                                created_by_user_id=creator, steps=[],
                                template_definition={"steps": [], "agents": [], "config": {}, "variables": []})
    db.add(playbook)
    db.flush()
    return playbook.id


@pytest.fixture
def cafe(db_session, seed_workspace):
    """A workspace with the owner (who made the playbook), an admin, a member and
    a non-system worker agent: B9's Green Coffee Stock Monitor."""
    db = db_session
    ws = UUID(seed_workspace())
    owner, admin, member = _user(db, "gerard"), _user(db, "priya"), _user(db, "sam")
    _member(db, ws, owner, "owner")
    _member(db, ws, admin, "admin")
    _member(db, ws, member, "member")
    return NS(db=db, ws=ws, owner=owner, admin=admin, member=member,
              worker=_agent(db, ws, "Green Coffee Stock Monitor"), playbook=_playbook(db, ws, owner))


def _check(cafe, caller_context, *, actor=None, playbook=None):
    return can_actor_modify(cafe.db, actor_agent_id=actor or cafe.worker, target_type=TARGET_PLAYBOOK,
                            workspace_id=cafe.ws, target_id=playbook or cafe.playbook, change_type="update",
                            source="platform_tool", caller_context=caller_context)


def _for(user, **extra):
    return {"driving_user_id": str(user), "conversation_id": "fbc85cc8", **extra}


# ── allowed ─────────────────────────────────────────────────────────────────

def test_an_agent_changing_a_playbook_for_its_creator_is_allowed(cafe):
    decision = _check(cafe, _for(cafe.owner))
    assert (decision.allowed, decision.reason) == (True, "acts_for_creator")


def test_an_agent_acting_for_a_workspace_admin_is_allowed(cafe):
    decision = _check(cafe, _for(cafe.admin))
    assert (decision.allowed, decision.reason) == (True, "acts_for_workspace_admin")


def test_the_server_side_super_admin_role_counts(cafe):
    decision = _check(cafe, _for(cafe.member, system_role="super_admin"))
    assert (decision.allowed, decision.reason) == (True, "acts_for_workspace_admin")


def test_auto_keeps_its_system_bypass(cafe):
    auto = _agent(cafe.db, cafe.ws, "Auto", system=True)
    decision = _check(cafe, None, actor=auto)
    assert decision.allowed and decision.bypass


# ── refused, with the real cause and a way to get it done ───────────────────

def test_a_member_who_is_not_the_creator_is_refused(cafe):
    decision = _check(cafe, _for(cafe.member))
    assert (decision.allowed, decision.reason, decision.escalation_target) == (
        False, "not_the_creators_agent", "auto")
    assert decision.message == hp._PLAYBOOK_REFUSALS["not_the_creators_agent"]


@pytest.mark.parametrize("context", [None, {}, {"conversation_id": "fbc85cc8"}, {"driving_user_id": "  "}])
def test_an_agent_driving_for_nobody_is_refused_and_told_how(cafe, context):
    decision = _check(cafe, context)
    assert (decision.allowed, decision.reason) == (False, "no_driving_user")
    assert decision.message.endswith(HOW)


def test_an_inactive_admin_is_not_an_admin(cafe):
    former = _user(cafe.db, "former")
    _member(cafe.db, cafe.ws, former, "admin", active=False)
    assert _check(cafe, _for(former)).reason == "not_the_creators_agent"


def test_an_admin_of_another_workspace_is_not_an_admin_here(cafe, seed_workspace):
    elsewhere = _user(cafe.db, "elsewhere")
    _member(cafe.db, UUID(seed_workspace()), elsewhere, "owner")
    assert _check(cafe, _for(elsewhere)).reason == "not_the_creators_agent"


def test_a_playbook_with_no_recorded_creator_needs_an_admin(cafe):
    unowned = _playbook(cafe.db, cafe.ws, None)
    assert _check(cafe, _for(cafe.member), playbook=unowned).reason == "no_creator_recorded"
    assert _check(cafe, _for(cafe.admin), playbook=unowned).allowed


def test_a_playbook_in_another_workspace_is_not_found(cafe, seed_workspace):
    foreign = _playbook(cafe.db, UUID(seed_workspace()), cafe.owner)
    decision = _check(cafe, _for(cafe.owner), playbook=foreign)
    assert (decision.allowed, decision.reason, decision.escalation_target) == (False, "target_not_found", None)


def test_a_failed_creator_lookup_is_refused_with_its_cause(cafe, monkeypatch):
    def _broken(db, target_id, ws):
        raise RuntimeError("relation vanished")

    monkeypatch.setattr(hp, "_playbook_creator", _broken)
    decision = _check(cafe, _for(cafe.owner))
    assert (decision.allowed, decision.reason) == (False, "owner_lookup_failed:RuntimeError")


# ── through the executor: only the server-built context counts ──────────────

SPOOFS = {"_driving_user_id": None, "driving_user_id": None, "system_role": "super_admin",
          "_caller_is_super_admin": True, "_caller_is_admin": True, "workspace_role": "owner"}


def _execute(cafe, action, params, caller_context):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    ran = AsyncMock(return_value={"success": True, "handler": "ran"})
    executor._handlers[action] = ran
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, params, caller_context)), ran


@pytest.mark.parametrize("caller_context", [None, {"conversation_id": "fbc85cc8"}])
def test_a_spoofed_driver_or_role_in_the_tool_call_counts_for_nothing(cafe, caller_context):
    spoofs = {**SPOOFS, "_driving_user_id": cafe.owner, "driving_user_id": str(cafe.owner)}
    params = {"playbook_id": cafe.playbook, "name": "Renamed", "_agent_id": cafe.worker, **spoofs}
    reply, ran = _execute(cafe, "platform_update_playbook", params, caller_context)
    assert (reply["success"], reply["reason"]) == (False, "no_driving_user")
    assert reply["error"].endswith(HOW)
    ran.assert_not_called()


def test_the_executor_passes_the_server_context_and_the_edit_runs(cafe):
    params = {"playbook_id": cafe.playbook, "name": "Renamed", "_agent_id": cafe.worker}
    reply, ran = _execute(cafe, "platform_update_playbook", params, _for(cafe.owner))
    assert reply == {"success": True, "handler": "ran"}
    ran.assert_called_once()


@pytest.mark.parametrize("caller_context, recorded", [("owner", "owner"), (None, None)])
def test_create_playbook_records_the_person_it_is_made_for(cafe, caller_context, recorded):
    """The executor strips a model-supplied _driving_user_id and injects the server's."""
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    context = _for(cafe.owner) if caller_context else None
    params = {"name": "Friday café check-in", "description": "Chase quiet cafés.", "_agent_id": cafe.worker,
              "_driving_user_id": cafe.member}
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        reply = asyncio.run(PlatformActionExecutor(cafe.db, cafe.ws).execute(
            "platform_create_playbook", params, context))
    assert reply["success"] is True
    created = cafe.db.execute(text("SELECT created_by_user_id FROM workflow_recipes WHERE id = :id"),
                              {"id": reply["playbook"]["id"]}).scalar()
    assert created == (cafe.owner if recorded else None)


def test_a_resumed_call_still_knows_who_it_is_made_for(cafe):
    """An ask's grant snapshots the caller; the approval's resume re-dispatches with it."""
    from modules.tools.execution.tool_grants import issue_tool_grant

    grant = issue_tool_grant(cafe.db, cafe.ws, action="platform_delete_playbook",
                             params={"playbook_id": cafe.playbook, "_agent_id": cafe.worker},
                             permission_level="destructive", description="Delete a playbook",
                             caller_context=_for(cafe.owner))
    assert grant is not None
    assert grant.details["caller_context"]["driving_user_id"] == str(cafe.owner)


def test_the_ui_create_records_its_maker():
    """The route and the mission save pass the request's own users.id."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert "created_by_user_id=_creator_pk(db, ctx)," in (root / "api" / "workflow_recipes.py").read_text()
    assert "created_by_user_id=resolve_user_pk(db, ctx)," in (root / "api" / "missions.py").read_text()
