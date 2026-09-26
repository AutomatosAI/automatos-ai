"""F145 — the executor's admin predicate asks who the call is made for.

``PlatformActionExecutor._caller_is_admin`` (the admin_only gate, and F122's
listing) read a ``workspace_role`` the chat's caller context never carries, so a
workspace owner read as non-admin; and with NO caller context it fell back to
"the workspace has an owner or admin", so a board, mission or heartbeat lane
acting for nobody passed in every workspace that has one (the policy plane is
off by default). No action is admin_only today (PRD-143 Rev 2 moved the tier to
super_admin_only), so nothing was exposed; the first one would have been.

Now: full autonomy (the owner's explicit grant), or a call made for an active
owner/admin (read fresh from workspace_members) or for the server-side
super_admin role (core.security.driving_user). No caller context is not an
admin, unless the policy plane's opt-in agents_inherit_admin says so. A probe
action stands in for the empty admin tier, as F122's test does.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

PROBE = "platform_f145_admin_probe"
REFUSED = f"Action '{PROBE}' requires workspace admin or owner role."


def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


def _member(db, ws, user, role):
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"), {"ws": str(ws), "user": user, "role": role})


@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    """A workspace with an owner, an admin and a member, and an admin_only probe."""
    from modules.tools.discovery.action_registry import ActionDefinition, get_action_registry

    live = get_action_registry()
    live.get_all()
    monkeypatch.setitem(live._actions, PROBE, ActionDefinition(
        name=PROBE, description="F145 probe", category="t", permission_level="read",
        parameters={"type": "object", "properties": {}, "required": []}, admin_only=True))
    db = db_session
    ws = UUID(seed_workspace())
    owner, admin, member = _user(db, "gerard"), _user(db, "priya"), _user(db, "sam")
    _member(db, ws, owner, "owner")
    _member(db, ws, admin, "admin")
    _member(db, ws, member, "member")
    return NS(db=db, ws=ws, owner=owner, admin=admin, member=member)


def _run(cafe, caller_context, params=None, *, full_autonomy=False):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    executor._full_autonomy = lambda: full_autonomy
    handler = AsyncMock(return_value={"success": True, "handler": "ran"})
    executor._handlers[PROBE] = handler
    reply = asyncio.run(executor.execute(PROBE, dict(params or {}), caller_context))
    return reply, handler


def _for(user, **extra):
    return {"driving_user_id": str(user), "conversation_id": "c1", **extra}


def _plane(monkeypatch, *, on, inherit=None, broken=False):
    import modules.policy as policy
    import modules.policy.policy_document as documents

    monkeypatch.setattr(policy, "policy_plane_enabled", lambda: on)

    def _load(db, ws):
        if broken:
            raise RuntimeError("policy row unreadable")
        return NS(agents_inherit_admin=inherit)

    monkeypatch.setattr(documents, "load_policy_document", _load)


# ── made for someone ────────────────────────────────────────────────────────

@pytest.mark.parametrize("who", ["owner", "admin"])
def test_a_call_made_for_an_owner_or_admin_is_an_admins(cafe, who):
    reply, handler = _run(cafe, _for(getattr(cafe, who)))
    assert reply == {"success": True, "handler": "ran"}


def test_the_server_side_super_admin_role_counts(cafe):
    reply, _handler = _run(cafe, _for(cafe.member, system_role="super_admin"))
    assert reply["success"] is True


def test_a_member_is_not_an_admin(cafe):
    reply, handler = _run(cafe, _for(cafe.member))
    assert reply == {"success": False, "permission_denied": True, "required_role": "owner_or_admin", "error": REFUSED}
    handler.assert_not_called()


def test_a_role_without_a_driving_user_is_not_an_admin(cafe):
    """The old predicate trusted these keys; nothing server-side writes them."""
    reply, handler = _run(cafe, {"conversation_id": "c1", "workspace_role": "owner", "system_role": "admin"})
    assert reply["success"] is False
    handler.assert_not_called()


# ── made for nobody ─────────────────────────────────────────────────────────

def test_a_lane_acting_for_nobody_is_not_an_admin_with_the_plane_off(cafe, monkeypatch):
    _plane(monkeypatch, on=False)
    reply, handler = _run(cafe, None)
    assert reply == {"success": False, "permission_denied": True, "required_role": "owner_or_admin", "error": REFUSED}
    handler.assert_not_called()


def test_a_spoofed_role_in_the_tool_call_counts_for_nothing(cafe, monkeypatch):
    _plane(monkeypatch, on=False)
    spoofed = {"workspace_role": "owner", "system_role": "super_admin", "_caller_is_admin": True,
               "_caller_is_super_admin": True, "driving_user_id": str(cafe.owner)}
    reply, handler = _run(cafe, None, spoofed)
    assert reply["success"] is False
    handler.assert_not_called()


@pytest.mark.parametrize("inherit, allowed", [(True, True), (False, False)])
def test_under_the_plane_only_the_opt_in_policy_inherits(cafe, monkeypatch, inherit, allowed):
    _plane(monkeypatch, on=True, inherit=inherit)
    reply, _handler = _run(cafe, None)
    assert reply["success"] is allowed


def test_an_unreadable_policy_is_not_an_admin(cafe, monkeypatch):
    _plane(monkeypatch, on=True, broken=True)
    reply, handler = _run(cafe, None)
    assert reply["success"] is False
    handler.assert_not_called()


def test_full_autonomy_is_the_owners_grant(cafe, monkeypatch):
    _plane(monkeypatch, on=False)
    reply, _handler = _run(cafe, None, full_autonomy=True)
    assert reply["success"] is True
