"""F147 — every mutating platform action is gated, and CI runs the audit.

test.yml runs scripts/check_hierarchy_gate.py. The seven mutating actions it
listed are accounted for:

- platform_update_system_setting (a platform-wide setting) is super_admin_only;
- platform_configure_channel is admin_only (F212: with no card, since its grant
  would store the channel's credentials), and the call's config
  merges into the stored one, keeping any key it does not name (as the REST PUT
  keeps trigger_mode);
- platform_update_workspace_settings is admin_only;
- the other four are allow-listed with their reasons.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]


def _audit():
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", ORCH / "scripts" / "check_hierarchy_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_hierarchy_audit_passes():
    assert _audit().main() == 0


def test_ci_runs_the_audit():
    workflow = (ORCH.parent / ".github" / "workflows" / "test.yml").read_text()
    assert "run: python scripts/check_hierarchy_gate.py" in workflow


def test_the_three_open_actions_are_gated():
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    system = registry.get("platform_update_system_setting")
    channel = registry.get("platform_configure_channel")
    workspace = registry.get("platform_update_workspace_settings")
    assert (system.super_admin_only, system.requires_confirmation) == (True, True)
    assert (channel.admin_only, channel.requires_confirmation) == (True, False)
    assert workspace.admin_only is True


# ── through the executor ────────────────────────────────────────────────────

def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


@pytest.fixture
def cafe(db_session, seed_workspace):
    db = db_session
    ws = UUID(seed_workspace())
    owner, member = _user(db, "gerard"), _user(db, "sam")
    for user, role in ((owner, "owner"), (member, "member")):
        db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                        "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"), {"ws": str(ws), "user": user, "role": role})
    return NS(db=db, ws=ws, owner=owner, member=member)


def _run(cafe, action, params, caller_context):
    from unittest.mock import patch

    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    executor._full_autonomy = lambda: False
    handler = AsyncMock(return_value={"success": True, "handler": "ran"})
    executor._handlers[action] = handler
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, params, caller_context)), handler


def _for(user, **extra):
    return {"driving_user_id": str(user), "conversation_id": "c1", **extra}


def test_a_workspace_owner_cannot_change_a_platform_setting(cafe):
    reply, handler = _run(cafe, "platform_update_system_setting",
                          {"category": "llm", "key": "default_model", "value": "x"}, _for(cafe.owner))
    assert reply["success"] is False and reply.get("permission_denied") is True
    assert "super admin" in reply["error"]
    handler.assert_not_called()


def test_a_member_cannot_change_the_workspace_settings_but_an_owner_can(cafe):
    params = {"key": "default_notification_channel", "value": "email"}
    reply, handler = _run(cafe, "platform_update_workspace_settings", params, _for(cafe.member))
    assert reply == {"success": False, "permission_denied": True, "required_role": "owner_or_admin",
                     "error": "Action 'platform_update_workspace_settings' requires workspace admin or owner role."}
    handler.assert_not_called()
    reply, handler = _run(cafe, "platform_update_workspace_settings", params, _for(cafe.owner))
    assert reply == {"success": True, "handler": "ran"}


def test_a_lane_acting_for_nobody_cannot_configure_a_channel(cafe):
    reply, handler = _run(cafe, "platform_configure_channel", {"channel_id": str(uuid4()), "config": {}}, None)
    assert reply["success"] is False
    handler.assert_not_called()


# ── the channel config merges ───────────────────────────────────────────────

STORED = {"bot_token": "123:secret", "trigger_mode": "strict", "chat_id": "c1"}


def _channel(cafe):
    channel_id = str(uuid4())
    cafe.db.execute(text("INSERT INTO channel_connections (id, workspace_id, platform, config) "
                         "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), 'telegram', CAST(:config AS json))"),
                    {"id": channel_id, "ws": str(cafe.ws), "config": json.dumps(STORED)})
    return channel_id


def _configure(cafe, channel_id, config):
    from modules.tools.discovery.handlers_channels import configure_channel

    return asyncio.run(configure_channel(cafe.db, cafe.ws, {"channel_id": channel_id, "config": config}))


def _stored(cafe, channel_id):
    return cafe.db.execute(text("SELECT config FROM channel_connections WHERE id = CAST(:id AS uuid)"),
                           {"id": channel_id}).scalar()


def test_a_config_change_keeps_the_trust_gate_and_the_credentials(cafe):
    channel_id = _channel(cafe)
    assert _configure(cafe, channel_id, {"chat_id": "c2"})["success"] is True
    assert _stored(cafe, channel_id) == {"bot_token": "123:secret", "trigger_mode": "strict", "chat_id": "c2"}


def test_a_named_trigger_mode_is_changed_and_checked(cafe):
    channel_id = _channel(cafe)
    assert _configure(cafe, channel_id, {"trigger_mode": "communication_only"})["success"] is True
    assert _stored(cafe, channel_id)["trigger_mode"] == "communication_only"
    reply = _configure(cafe, channel_id, {"trigger_mode": "everyone"})
    assert reply["success"] is False and "trigger_mode must be one of" in reply["error"]
    assert _stored(cafe, channel_id)["trigger_mode"] == "communication_only"
