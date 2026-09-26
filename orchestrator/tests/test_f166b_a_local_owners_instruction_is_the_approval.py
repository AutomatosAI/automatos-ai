"""F166's twin — on the local edition, the owner's own instruction is the approval too.

"The instruction IS the approval" (Gerard, 2026-08-06): a confirmation-gated action
that a workspace owner or admin instructs in an interactive chat turn runs with no
card. The check read only the Clerk id. The local edition has none, so the local
owner's instructions still stopped at a card, the gap F166 closed for mission
attribution. With no Clerk id, the person typing is now the chat's server-threaded
``driving_user_id``: an active owner/admin of the workspace, or the super admin the
local operator is (F145's predicate). Lanes, widget turns, members and the Clerk
path are unchanged.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from core.security.surface import WIDGET, turn_surface

PROBE = "platform_f166b_confirm_probe"


def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


def _member(db, ws, user, role):
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"), {"ws": str(ws), "user": user, "role": role})


@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    """A workspace, its owner, admin and member, the local operator (no membership), and a gated probe."""
    from modules.tools.discovery.action_registry import ActionDefinition, get_action_registry

    live = get_action_registry()
    live.get_all()
    monkeypatch.setitem(live._actions, PROBE, ActionDefinition(
        name=PROBE, description="F166b probe", category="t", permission_level="write",
        parameters={"type": "object", "properties": {}, "required": []}, requires_confirmation=True))
    db = db_session
    ws = UUID(seed_workspace())
    owner, admin, member, operator = _user(db, "gerard"), _user(db, "priya"), _user(db, "sam"), _user(db, "local")
    _member(db, ws, owner, "owner")
    _member(db, ws, admin, "admin")
    _member(db, ws, member, "member")
    return NS(db=db, ws=ws, owner=owner, admin=admin, member=member, operator=operator)


def _run(cafe, caller_context):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    executor._full_autonomy = lambda: False
    handler = AsyncMock(return_value={"success": True, "handler": "ran"})
    executor._handlers[PROBE] = handler
    reply = asyncio.run(executor.execute(PROBE, {}, caller_context))
    return reply, handler


def _chat(user, **extra):
    """What the local chat threads: no Clerk id, the internal id of the person typing."""
    return {"conversation_id": "a7af93bd", "turn_id": "t-1", "driving_user_id": str(user), **extra}


def test_the_local_operators_instruction_is_the_approval(cafe):
    reply, handler = _run(cafe, _chat(cafe.operator, system_role="super_admin"))
    assert handler.await_count == 1 and reply.get("requires_confirmation") is not True


@pytest.mark.parametrize("who", ["owner", "admin"])
def test_an_owners_or_admins_instruction_is_the_approval(cafe, who):
    reply, handler = _run(cafe, _chat(getattr(cafe, who)))
    assert handler.await_count == 1 and reply.get("requires_confirmation") is not True


@pytest.mark.parametrize("context", ["member", "lane", "nobody"])
def test_anyone_else_still_gets_the_card(cafe, context):
    caller = {"member": _chat(cafe.member),
              "lane": {"driving_user_id": str(cafe.owner), "system_role": "super_admin"},  # no conversation
              "nobody": {"conversation_id": "c1"}}[context]
    reply, handler = _run(cafe, caller)
    assert reply.get("requires_confirmation") is True and handler.await_count == 0


def test_a_widget_turn_is_never_the_owners_instruction(cafe):
    with turn_surface(WIDGET, ("chat",), None):
        reply, handler = _run(cafe, _chat(cafe.operator, system_role="super_admin"))
    assert handler.await_count == 0


def test_a_clerk_principal_keeps_the_clerk_path(cafe, monkeypatch):
    """SaaS: a Clerk id is resolved by membership as before; the super admin role
    and a driving_user_id beside it do not widen it."""
    import modules.tools.discovery.platform_executor as pe

    monkeypatch.setattr(pe, "_workspace_role_for_clerk", lambda db, ws, uid: None)
    reply, handler = _run(cafe, _chat(cafe.owner, user_id="user_2abc", system_role="super_admin"))
    assert reply.get("requires_confirmation") is True and handler.await_count == 0


@pytest.mark.parametrize("who", ["operator", "owner"])
def test_on_saas_a_turn_with_no_clerk_id_keeps_the_card(cafe, monkeypatch, who):
    """Review follow-up: only the local edition has no Clerk id by design. On SaaS, a
    turn without one (a failed lookup, a row never linked to Clerk) is not widened,
    super admin or not."""
    from config import config as app_config

    monkeypatch.setattr(app_config, "AUTH_EDITION", "saas")
    reply, handler = _run(cafe, _chat(getattr(cafe, who), system_role="super_admin"))
    assert reply.get("requires_confirmation") is True and handler.await_count == 0
