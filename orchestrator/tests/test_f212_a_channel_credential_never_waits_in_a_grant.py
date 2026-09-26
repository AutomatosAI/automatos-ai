"""F212 — a channel's credentials never wait in a card's approval grant.

A card's approval stores the call's params in approval_grants, in plain text.
platform_connect_channel and platform_configure_channel carry bot tokens and
signing secrets, so neither asks for a card: admin_only is their gate (main's
behaviour plus F151's). The card comes back once grant params are encrypted.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

TOKEN = "123456:f212-bot-token"
CHANNEL_WRITES = ("platform_connect_channel", "platform_configure_channel")


@pytest.mark.parametrize("name", CHANNEL_WRITES)
def test_a_channel_write_is_an_admins_and_asks_for_no_card(name):
    from modules.tools.discovery import get_action_registry

    action = get_action_registry().get(name)
    assert (action.admin_only, action.requires_confirmation) == (True, False)


# ── through the executor ────────────────────────────────────────────────────

def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


@pytest.fixture
def cafe(db_session, seed_workspace):
    db = db_session
    ws = UUID(seed_workspace())
    admin, editor = _user(db, "priya"), _user(db, "sam")
    for user, role in ((admin, "admin"), (editor, "editor")):
        db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                        "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"), {"ws": str(ws), "user": user, "role": role})
    return NS(db=db, ws=ws, admin=admin, editor=editor)


def _run(cafe, action, params, caller_context):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    executor._full_autonomy = lambda: False
    handler = AsyncMock(return_value={"success": True, "handler": "ran"})
    executor._handlers[action] = handler
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, params, caller_context)), handler


def _grants_holding_the_token(cafe):
    from core.models.approval_grants import ApprovalGrant

    return [g for g in cafe.db.query(ApprovalGrant).all() if TOKEN in str(vars(g))]


@pytest.mark.parametrize("name,params", [
    ("platform_connect_channel", {"platform": "telegram", "config": {"bot_token": TOKEN}}),
    ("platform_configure_channel", {"channel_id": str(uuid4()), "config": {"bot_token": TOKEN}}),
])
def test_an_admins_token_goes_to_the_channel_never_to_a_grant(cafe, name, params):
    reply, handler = _run(cafe, name, params, {"driving_user_id": str(cafe.admin)})
    assert reply == {"success": True, "handler": "ran"}
    handler.assert_called_once()
    assert _grants_holding_the_token(cafe) == []


def test_an_editor_still_cannot_change_a_channel(cafe):
    params = {"channel_id": str(uuid4()), "config": {"bot_token": TOKEN}}
    reply, handler = _run(cafe, "platform_configure_channel", params, {"driving_user_id": str(cafe.editor)})
    assert reply["success"] is False and reply.get("permission_denied") is True
    handler.assert_not_called()
