"""F148 — inviting members and changing their roles follow REST's rules.

REST: inviting needs members:invite (an owner or admin); changing a role needs
members:change_role, which only the owner holds (the platform super admin
bypasses both). The chat tools match: invite is admin_only and confirmed,
never invites an owner, and is sent by the person the call is made for;
set_member_role is admin_only and its handler requires the owner (or the super
admin), read from the server-injected driver. A caller-supplied driver or
super-admin flag is stripped.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

REFUSED_OWNER = "Only the workspace owner changes member roles. Ask the owner, or have them do it in chat."


def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


def _member(db, ws, user, role):
    return db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                           "VALUES (CAST(:ws AS uuid), :user, :role, TRUE) RETURNING id"),
                      {"ws": str(ws), "user": user, "role": role}).scalar()


@pytest.fixture
def team(db_session, seed_workspace):
    db = db_session
    ws = UUID(seed_workspace())
    owner, admin, editor = _user(db, "gerard"), _user(db, "priya"), _user(db, "sam")
    _member(db, ws, owner, "owner")
    _member(db, ws, admin, "admin")
    editor_row = _member(db, ws, editor, "editor")
    return NS(db=db, ws=ws, owner=owner, admin=admin, editor=editor, editor_row=editor_row)


def test_the_two_actions_are_gated():
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    invite, role = registry.get("platform_invite_member"), registry.get("platform_set_member_role")
    assert (invite.admin_only, invite.requires_confirmation) == (True, True)
    assert (role.admin_only, role.requires_confirmation) == (True, True)


# ── set_member_role: the owner's alone ──────────────────────────────────────

def _set_role(team, **params):
    from modules.tools.discovery.handlers_members import set_member_role

    return asyncio.run(set_member_role(team.db, team.ws, {"member_id": team.editor_row, "role": "admin", **params}))


def test_an_admin_cannot_change_a_role(team):
    assert _set_role(team, _driving_user_id=team.admin) == {"success": False, "error": REFUSED_OWNER}


def test_no_driver_cannot_change_a_role(team):
    assert _set_role(team) == {"success": False, "error": REFUSED_OWNER}


@pytest.mark.parametrize("who", ["owner", "super_admin"])
def test_the_owner_or_the_super_admin_changes_it(team, who):
    driver = {"_driving_user_id": team.owner} if who == "owner" else {"_driving_super_admin": True}
    reply = _set_role(team, **driver)
    assert (reply["success"], reply["old_role"], reply["new_role"]) == (True, "editor", "admin")


# ── invite_member ───────────────────────────────────────────────────────────

def test_nobody_is_invited_as_the_owner(team):
    from modules.tools.discovery.handlers_members import invite_member

    reply = asyncio.run(invite_member(team.db, team.ws, {"email": "x@example.com", "role": "Owner",
                                                         "_driving_user_id": team.owner}))
    assert reply["success"] is False and "Nobody is invited as the owner" in reply["error"]


def test_the_invitation_is_sent_by_the_person_the_call_is_made_for(team):
    from modules.tools.discovery.handlers_members import invite_member

    sent = AsyncMock(return_value=NS(id=1, email="cafe@example.com", role="editor", expires_at=None))
    with patch("core.workspaces.invitations.invite_member_to_workspace", sent):
        reply = asyncio.run(invite_member(team.db, team.ws, {"email": "cafe@example.com", "role": "editor",
                                                             "_driving_user_id": team.admin}))
    assert reply["success"] is True
    assert sent.await_args.kwargs["inviter_internal_id"] == team.admin


# ── through the executor ────────────────────────────────────────────────────

def _execute(team, action, params, caller_context, *, full_autonomy=False):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(team.db, team.ws)
    executor._full_autonomy = lambda: full_autonomy
    seen = []

    async def _handler(db, ws, handler_params):
        seen.append(handler_params)
        return {"success": True}

    executor._handlers[action] = _handler
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        reply = asyncio.run(executor.execute(action, params, caller_context))
    return reply, seen


@pytest.mark.parametrize("caller_context", ["editor", None])
def test_a_non_admin_or_a_lane_for_nobody_cannot_invite(team, caller_context):
    context = {"driving_user_id": str(team.editor)} if caller_context else None
    reply, seen = _execute(team, "platform_invite_member", {"email": "x@example.com", "role": "admin"}, context)
    assert reply["success"] is False and reply.get("permission_denied") is True
    assert seen == []


def test_an_admins_invitation_waits_for_its_card(team):
    reply, seen = _execute(team, "platform_invite_member", {"email": "x@example.com", "role": "editor"},
                           {"driving_user_id": str(team.admin)})
    assert reply.get("requires_confirmation") is True and seen == []


def test_a_spoofed_driver_or_super_admin_flag_is_replaced_by_the_servers(team):
    spoofed = {"member_id": team.editor_row, "role": "admin", "_driving_user_id": team.owner,
               "_driving_super_admin": True}
    reply, seen = _execute(team, "platform_set_member_role", spoofed, {"driving_user_id": str(team.admin)},
                           full_autonomy=True)
    assert reply["success"] is True
    assert seen[0]["_driving_user_id"] == team.admin and "_driving_super_admin" not in seen[0]
