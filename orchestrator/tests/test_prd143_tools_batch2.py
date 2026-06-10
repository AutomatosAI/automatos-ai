"""PRD-143 S11 — operator tools batch 2: the administration surface.

Inventory result (AC #1): plugin ENABLE was already covered
(``platform_install_plugin``), as were plugin/skill/model listings and the
workspace-info read. The honest admin-surface gaps were:

  - members              -> platform_list_members / platform_invite_member
                            / platform_set_member_role / platform_remove_member
  - workspace settings   -> platform_update_workspace_settings (fail-closed
                            whitelist: byok_overrides, default_notification_channel)
  - system settings      -> platform_list_system_settings (sensitive values masked)
                            / platform_update_system_setting
  - SDK API keys         -> platform_list_api_keys / platform_create_api_key
                            / platform_revoke_api_key
  - plugin DISABLE       -> platform_uninstall_plugin

Deliberately excluded: BYOK provider-key add/delete (api/user_api_keys.py)
because raw provider secrets must never transit the LLM context; SDK keys are
generated server-side so create is safe (full key returned exactly once,
straight from ApiKeyService like the REST router).

Every tool is OPERATOR tier — the Rev 2 inversion. Safety is gates-and-logs:
destructive/role-changing tools are permission_level='destructive' with
requires_confirmation=True (the executor's destructive backstop rejects any
destructive action without it), everything is workspace-scoped, and the
member-invite flow delegates to api.team.invite_member_to_workspace —
extracted from POST /api/workspaces/{id}/team/invite so router and tool share
one implementation (the S10 connect_channel_for_workspace precedent).

Import idiom mirrors test_prd143_tools_batch1.py (dummy POSTGRES_* on a
closed port + apscheduler stub) — nothing here touches a DB.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    if "apscheduler" in sys.modules:
        return
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


_install_fake_apscheduler()

from modules.tools.discovery.handlers_api_keys import (  # noqa: E402
    create_api_key,
    list_api_keys,
    revoke_api_key,
)
from modules.tools.discovery.handlers_marketplace import uninstall_plugin  # noqa: E402
from modules.tools.discovery.handlers_members import (  # noqa: E402
    invite_member,
    list_members,
    remove_member,
    set_member_role,
)
from modules.tools.discovery.handlers_workspace import (  # noqa: E402
    list_system_settings,
    update_system_setting,
    update_workspace_settings,
)

_WS = UUID("00000000-0000-0000-0000-0000000000bb")
_OTHER_WS = UUID("00000000-0000-0000-0000-0000000000cc")

BATCH2_TOOLS = {
    "platform_list_members": "read",
    "platform_invite_member": "write",
    "platform_set_member_role": "destructive",
    "platform_remove_member": "destructive",
    "platform_update_workspace_settings": "write",
    "platform_list_system_settings": "read",
    "platform_update_system_setting": "write",
    "platform_list_api_keys": "read",
    "platform_create_api_key": "write",
    "platform_revoke_api_key": "destructive",
    "platform_uninstall_plugin": "destructive",
}


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fakes — a sequential ORM session: each db.query(...) pops the next scripted
# result. .first() returns it as-is; .all() wraps a scalar into a list.
# ---------------------------------------------------------------------------

class _CaptureQuery:
    def __init__(self, result=None):
        self._result = result
        self.filters = []

    def filter(self, *args, **kwargs):
        self.filters.extend(args)
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return self._result

    def all(self):
        if self._result is None:
            return []
        return self._result if isinstance(self._result, list) else [self._result]

    def delete(self, *args, **kwargs):
        return self._result if isinstance(self._result, int) else 0


class _SeqDB:
    def __init__(self, results=()):
        self._results = list(results)
        self.queries = []
        self.committed = False
        self.rolledback = False
        self.added = []
        self.deleted = []

    def query(self, *a, **k):
        result = self._results.pop(0) if self._results else None
        q = _CaptureQuery(result)
        self.queries.append(q)
        return q

    def add(self, obj):
        self.added.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolledback = True

    def refresh(self, obj):
        pass

    def flush(self):
        pass


def _bound_values(db):
    """Literal bind values across every captured query's filters."""
    vals = []
    for q in db.queries:
        for f in q.filters:
            right = getattr(f, "right", None)
            v = getattr(right, "value", None)
            if v is not None:
                vals.append(v)
    return vals


def _member(member_id=10, user_id=77, role="editor", ws=_WS):
    return SimpleNamespace(
        id=member_id, user_id=user_id, role=role, workspace_id=ws,
        is_active=True, joined_at=None,
    )


def _user(user_id=77, email="member@example.com", name="Member"):
    return SimpleNamespace(id=user_id, email=email, name=name)


class _FakeWorkspace:
    def __init__(self, settings=None):
        self.id = _WS
        self.settings = settings


# ---------------------------------------------------------------------------
# platform_list_members
# ---------------------------------------------------------------------------

def test_list_members_workspace_scoped():
    db = _SeqDB(results=[[_member()], _user()])
    out = _run(list_members(db, _WS, {}))
    assert out["success"] is True
    assert out["count"] == 1
    assert out["members"][0]["email"] == "member@example.com"
    assert out["members"][0]["role"] == "editor"
    assert out["members"][0]["member_id"] == 10
    assert _WS in _bound_values(db), "members query must filter by workspace_id"


# ---------------------------------------------------------------------------
# platform_invite_member — delegates to the canonical flow extracted from
# POST /api/workspaces/{id}/team/invite into the invitation service layer
# ---------------------------------------------------------------------------

def test_invite_member_delegates_to_canonical_flow(monkeypatch):
    import core.workspaces.invitations as invitations_mod

    captured = {}

    async def _fake_invite(db, workspace_id, email, role, inviter_internal_id, **kwargs):
        captured.update(
            workspace_id=workspace_id, email=email, role=role,
            inviter_internal_id=inviter_internal_id,
        )
        return SimpleNamespace(
            id=5, email=email, role=role, token="secret-token",
            expires_at=None, created_at=None,
        )

    monkeypatch.setattr(invitations_mod, "invite_member_to_workspace", _fake_invite)
    # one scripted query: the workspace owner row (audit/inviter principal)
    db = _SeqDB(results=[_member(member_id=1, user_id=42, role="owner")])
    out = _run(invite_member(db, _WS, {"email": "new@example.com", "role": "editor"}))
    assert out["success"] is True
    assert captured["workspace_id"] == str(_WS), "invite must be scoped to the caller workspace"
    assert captured["email"] == "new@example.com"
    assert captured["role"] == "editor"
    assert captured["inviter_internal_id"] == 42
    assert out["invitation"]["email"] == "new@example.com"
    assert "token" not in out["invitation"], "invite token must never reach the LLM context"


def test_invite_member_invalid_input_fails_closed(monkeypatch):
    import core.workspaces.invitations as invitations_mod

    async def _fake_invite(*a, **k):
        raise ValueError("Invalid role: superuser")

    monkeypatch.setattr(invitations_mod, "invite_member_to_workspace", _fake_invite)
    db = _SeqDB(results=[_member(member_id=1, user_id=42, role="owner")])
    out = _run(invite_member(db, _WS, {"email": "new@example.com", "role": "superuser"}))
    assert out["success"] is False
    assert "Invalid role" in out["error"]


def test_invite_member_no_owner_fails_closed(monkeypatch):
    """No resolvable workspace-owner principal -> refuse, never invite."""
    db = _SeqDB(results=[None])
    out = _run(invite_member(db, _WS, {"email": "new@example.com", "role": "editor"}))
    assert out["success"] is False


def test_invite_member_canonical_helper_exists():
    """The router-extracted helper is the single invite flow (no handler
    reimplementation). It lives in the invitation service layer so the tool
    chain never imports the router's pydantic models."""
    import inspect

    import core.workspaces.invitations as invitations_mod

    helper = invitations_mod.invite_member_to_workspace
    assert inspect.iscoroutinefunction(helper)
    params = list(inspect.signature(helper).parameters)
    for expected in ("db", "workspace_id", "email", "role", "inviter_internal_id"):
        assert expected in params


# ---------------------------------------------------------------------------
# platform_set_member_role
# ---------------------------------------------------------------------------

def test_set_member_role_workspace_scoped_happy_path():
    target = _member(member_id=10, role="editor")
    # queries: member lookup, then owner lookup (audit principal)
    db = _SeqDB(results=[target, _member(member_id=1, user_id=42, role="owner")])
    out = _run(set_member_role(db, _WS, {"member_id": 10, "role": "admin"}))
    assert out["success"] is True
    assert target.role == "admin"
    assert out["old_role"] == "editor"
    assert out["new_role"] == "admin"
    assert db.committed is True
    assert _WS in _bound_values(db), "member lookup must filter by workspace_id"


def test_set_member_role_owner_guard():
    db = _SeqDB(results=[_member(member_id=10, role="owner")])
    out = _run(set_member_role(db, _WS, {"member_id": 10, "role": "editor"}))
    assert out["success"] is False
    assert "owner" in out["error"].lower()
    assert db.committed is False


def test_set_member_role_invalid_role_fails_closed():
    db = _SeqDB(results=[_member(member_id=10, role="editor")])
    out = _run(set_member_role(db, _WS, {"member_id": 10, "role": "superuser"}))
    assert out["success"] is False
    assert db.committed is False


def test_set_member_role_tenant_isolation():
    """A workspace-A principal cannot touch workspace-B members: the lookup is
    workspace-filtered, so the cross-tenant row is simply never found."""
    db = _SeqDB(results=[None])
    out = _run(set_member_role(db, _WS, {"member_id": 999, "role": "admin"}))
    assert out["success"] is False
    assert "not found" in out["error"].lower()
    assert db.committed is False
    assert _WS in _bound_values(db)


# ---------------------------------------------------------------------------
# platform_remove_member
# ---------------------------------------------------------------------------

def test_remove_member_workspace_scoped_happy_path():
    target = _member(member_id=10, role="editor")
    db = _SeqDB(results=[target, _member(member_id=1, user_id=42, role="owner")])
    out = _run(remove_member(db, _WS, {"member_id": 10}))
    assert out["success"] is True
    assert target.is_active is False
    assert db.committed is True
    assert _WS in _bound_values(db)


def test_remove_member_owner_guard():
    db = _SeqDB(results=[_member(member_id=10, role="owner")])
    out = _run(remove_member(db, _WS, {"member_id": 10}))
    assert out["success"] is False
    assert "owner" in out["error"].lower()
    assert db.committed is False


def test_remove_member_tenant_isolation():
    db = _SeqDB(results=[None])
    out = _run(remove_member(db, _WS, {"member_id": 999}))
    assert out["success"] is False
    assert "not found" in out["error"].lower()
    assert db.committed is False
    assert _WS in _bound_values(db)


# ---------------------------------------------------------------------------
# platform_update_workspace_settings — fail-closed key whitelist
# ---------------------------------------------------------------------------

def test_update_workspace_settings_byok_happy_path():
    ws = _FakeWorkspace(settings={"byok_overrides": {"openai": False}})
    db = _SeqDB(results=[ws])
    out = _run(update_workspace_settings(db, _WS, {
        "key": "byok_overrides",
        "value": {"openai": True, "not_a_provider": True},
    }))
    assert out["success"] is True
    # merged, provider-whitelisted (same semantics as PUT /current/byok-preferences)
    assert ws.settings["byok_overrides"]["openai"] is True
    assert "not_a_provider" not in ws.settings["byok_overrides"]
    assert db.committed is True
    assert _WS in _bound_values(db)


def test_update_workspace_settings_notification_channel():
    ws = _FakeWorkspace(settings={})
    db = _SeqDB(results=[ws])
    out = _run(update_workspace_settings(db, _WS, {
        "key": "default_notification_channel",
        "value": "telegram",
    }))
    assert out["success"] is True
    assert ws.settings["default_notification_channel"] == "telegram"
    assert db.committed is True


def test_update_workspace_settings_invalid_channel_fails_closed():
    db = _SeqDB(results=[_FakeWorkspace(settings={})])
    out = _run(update_workspace_settings(db, _WS, {
        "key": "default_notification_channel",
        "value": "carrier-pigeon",
    }))
    assert out["success"] is False
    assert db.committed is False


def test_update_workspace_settings_rejects_non_whitelisted_key():
    """integrations/orchestrator/autonomy slices are NOT writable here —
    fail-closed, exactly like the S10 widget-config whitelist."""
    db = _SeqDB(results=[_FakeWorkspace(settings={})])
    out = _run(update_workspace_settings(db, _WS, {
        "key": "integrations",
        "value": {"slack_token": "xoxb-steal-me"},
    }))
    assert out["success"] is False
    assert "byok_overrides" in out["error"]
    assert db.committed is False


# ---------------------------------------------------------------------------
# platform_list_system_settings / platform_update_system_setting
# ---------------------------------------------------------------------------

def _setting(category="llm", key="model", value="gpt-x", sensitive=False, sid=1):
    return SimpleNamespace(
        id=sid, category=category, key=key, value=value,
        is_sensitive=sensitive, is_required=False, description=None,
        updated_at=None,
    )


def test_list_system_settings_masks_sensitive():
    rows = [
        _setting(sid=1, key="model", value="gpt-x", sensitive=False),
        _setting(sid=2, key="api_key", value="sk-super-secret", sensitive=True),
    ]
    db = _SeqDB(results=[rows])
    out = _run(list_system_settings(db, _WS, {}))
    assert out["success"] is True
    assert out["count"] == 2
    by_key = {s["key"]: s for s in out["settings"]}
    assert by_key["model"]["value"] == "gpt-x"
    assert by_key["api_key"]["value"] == "****", "sensitive values must never reach the LLM context"


def test_update_system_setting_by_category_key():
    row = _setting(category="llm", key="model", value="gpt-x")
    db = _SeqDB(results=[row])
    out = _run(update_system_setting(db, _WS, {
        "category": "llm", "key": "model", "value": "claude-opus-4-7",
    }))
    assert out["success"] is True
    assert row.value == "claude-opus-4-7"
    assert db.committed is True


def test_update_system_setting_not_found_fails_closed():
    db = _SeqDB(results=[None])
    out = _run(update_system_setting(db, _WS, {
        "category": "llm", "key": "ghost", "value": "x",
    }))
    assert out["success"] is False
    assert "not found" in out["error"].lower()
    assert db.committed is False


# ---------------------------------------------------------------------------
# SDK API keys — delegate to ApiKeyService (the router's service layer)
# ---------------------------------------------------------------------------

def test_list_api_keys_workspace_scoped(monkeypatch):
    from core.services.api_key_service import ApiKeyService

    captured = {}

    def _fake_list(db, workspace_id):
        captured["workspace_id"] = workspace_id
        return [{"id": "k-1", "name": "ci", "key_prefix": "ak_live_12…", "key_type": "server",
                 "permissions": ["chat"], "is_active": True}]

    monkeypatch.setattr(ApiKeyService, "list_api_keys", _fake_list)
    out = _run(list_api_keys(MagicMock(), _WS, {}))
    assert out["success"] is True
    assert out["count"] == 1
    assert out["keys"][0]["key_prefix"] == "ak_live_12…"
    assert captured["workspace_id"] == _WS, "list must be scoped to the caller workspace"


def test_create_api_key_happy_path(monkeypatch):
    from core.services.api_key_service import ApiKeyService

    captured = {}

    def _fake_create(db, workspace_id, name, key_type, permissions, **kwargs):
        captured.update(workspace_id=workspace_id, name=name, key_type=key_type,
                        permissions=permissions)
        return {"id": "k-9", "name": name, "key": "ak_live_full-key-once",
                "key_type": key_type, "permissions": permissions}

    monkeypatch.setattr(ApiKeyService, "create_api_key", _fake_create)
    out = _run(create_api_key(MagicMock(), _WS, {
        "name": "ci key", "key_type": "server", "permissions": ["chat"],
    }))
    assert out["success"] is True
    assert out["key"]["key"] == "ak_live_full-key-once"
    assert captured["workspace_id"] == _WS, "create must be scoped to the caller workspace"
    assert captured["key_type"] == "server"


def test_create_api_key_public_requires_domains():
    out = _run(create_api_key(MagicMock(), _WS, {
        "name": "widget key", "key_type": "public", "permissions": ["chat"],
    }))
    assert out["success"] is False
    assert "allowed_domains" in out["error"]


def test_create_api_key_invalid_permission_fails_closed():
    out = _run(create_api_key(MagicMock(), _WS, {
        "name": "bad key", "key_type": "server", "permissions": ["root:everything"],
    }))
    assert out["success"] is False
    assert "permission" in out["error"].lower()


def test_revoke_api_key_workspace_scoped(monkeypatch):
    from core.services.api_key_service import ApiKeyService

    captured = {}

    def _fake_revoke(db, key_id, workspace_id):
        captured.update(key_id=key_id, workspace_id=workspace_id)
        return True

    monkeypatch.setattr(ApiKeyService, "revoke_api_key", _fake_revoke)
    out = _run(revoke_api_key(MagicMock(), _WS, {"key_id": str(uuid4())}))
    assert out["success"] is True
    assert captured["workspace_id"] == _WS, "revoke must be scoped to the caller workspace"


def test_revoke_api_key_tenant_isolation(monkeypatch):
    """ApiKeyService scopes by workspace_id — a workspace-B key is never found
    from workspace A, so revoke fails closed."""
    from core.services.api_key_service import ApiKeyService

    monkeypatch.setattr(ApiKeyService, "revoke_api_key", lambda db, key_id, workspace_id: False)
    out = _run(revoke_api_key(MagicMock(), _WS, {"key_id": str(uuid4())}))
    assert out["success"] is False
    assert "not found" in out["error"].lower()


# ---------------------------------------------------------------------------
# platform_uninstall_plugin
# ---------------------------------------------------------------------------

def test_uninstall_plugin_workspace_scoped_happy_path():
    plugin_id = uuid4()
    junction = SimpleNamespace(workspace_id=_WS, plugin_id=plugin_id)
    plugin = SimpleNamespace(id=plugin_id, name="shop-sync", slug="shop-sync", enable_count=3)
    # queries: junction lookup, workspace agent ids, assignment delete, plugin row
    db = _SeqDB(results=[junction, [SimpleNamespace(id=7)], 2, plugin])
    out = _run(uninstall_plugin(db, _WS, {"plugin_id": str(plugin_id)}))
    assert out["success"] is True
    assert out["agents_unassigned"] == 2
    assert junction in db.deleted
    assert plugin.enable_count == 2
    assert db.committed is True
    assert _WS in _bound_values(db), "junction lookup must filter by workspace_id"


def test_uninstall_plugin_not_enabled_fails_closed():
    db = _SeqDB(results=[None])
    out = _run(uninstall_plugin(db, _WS, {"plugin_id": str(uuid4())}))
    assert out["success"] is False
    assert "not enabled" in out["error"].lower()
    assert db.committed is False


def test_uninstall_plugin_invalid_id_fails_closed():
    db = _SeqDB()
    out = _run(uninstall_plugin(db, _WS, {"plugin_id": "not-a-uuid"}))
    assert out["success"] is False


# ---------------------------------------------------------------------------
# Registry tier + executor wiring (AC: test_batch2_tools_operator_tier_and_permission_levels)
# ---------------------------------------------------------------------------

def test_batch2_tools_operator_tier_and_permission_levels():
    from modules.tools.discovery.action_registry import ActionRegistry

    registry = ActionRegistry()
    actions = {a.name: a for a in registry.get_all()}
    for name, level in BATCH2_TOOLS.items():
        assert name in actions, f"{name} missing from registry"
        action = actions[name]
        assert action.super_admin_only is False, (
            f"{name} must be operator tier (Rev 2 inversion — admin surface is "
            "deliberately open, gated by logs not exclusion)"
        )
        assert action.admin_only is False, f"{name} must not be admin-gated (post-S4 catalogue)"
        assert action.workspace_scoped is True, f"{name} must be workspace-scoped"
        assert action.permission_level == level, (
            f"{name}: expected permission_level={level!r}, got {action.permission_level!r}"
        )
        if level == "destructive":
            assert action.requires_confirmation is True, (
                f"{name} is destructive and must carry requires_confirmation=True "
                "(the executor's destructive backstop rejects it otherwise)"
            )


def test_batch2_handlers_wired_in_executor():
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(MagicMock(), uuid4())
    for name in BATCH2_TOOLS:
        assert name in executor._handlers, f"{name} has no executor handler"
