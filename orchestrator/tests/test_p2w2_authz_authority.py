"""PRD-195 S1 (P2-14) — one authorization authority: ``modules/policy/roles.py``.

The five role vocabularies collapse onto one module, with one semantic per
question:

- **empty permissions = deny** on every key plane — service
  (``ApiKeyService.check_permissions``), widget (``api/widgets/auth.py``
  ``require_permission``) and board (``_sdk_key_has_scope``) — with an explicit
  ``"*"`` honoured as a deliberate full grant (F042 closed, unconditionally);
- **super_admin ⊇ admin** on every admin gate, with no flag in the path
  (F043 closed; G2 decoupled the authZ legs from ``AUTOMATOS_POLICY_PLANE``);
- **workspace roles** ``owner ⊇ admin ⊇ editor ⊇ viewer`` live in the same
  module (matrix absorbed from ``core/workspaces/permissions.py``), with the
  canonical G1 strings — ``missions:*`` / ``playbooks:*``, never the legacy
  ``workflows:*``.

Pure tests: no DB, no network. Source-grep guards prove the legacy permissive
branches are DELETED, not just bypassed — if a later refactor moves the
checkers, repoint these greps (do not delete them).
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# Blessed fake-POSTGRES preamble (see tests/test_prd143_boundary_sweep.py):
# api/widgets/auth.py pulls the config chain; nothing here touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# ---------------------------------------------------------------------------
# The authority: workspace hierarchy + wildcards (G1 vocabulary)
# ---------------------------------------------------------------------------

def test_workspace_hierarchy_and_wildcards():
    from modules.policy.roles import workspace_has_permission

    # owner ⊇ admin ⊇ editor ⊇ viewer on the agents resource
    assert workspace_has_permission("owner", "agents:create") is True    # via agents:*
    assert workspace_has_permission("admin", "agents:delete") is True    # via agents:*
    assert workspace_has_permission("editor", "agents:create") is True
    assert workspace_has_permission("editor", "agents:update") is True
    assert workspace_has_permission("editor", "agents:delete") is False  # no delete below admin
    assert workspace_has_permission("viewer", "agents:read") is True
    assert workspace_has_permission("viewer", "agents:create") is False

    # S3/S4 spec: an editor creates, updates AND runs Missions/Playbooks —
    # but does not delete, manage members, or administer the workspace.
    assert workspace_has_permission("editor", "missions:execute") is True
    assert workspace_has_permission("editor", "playbooks:execute") is True
    assert workspace_has_permission("editor", "missions:delete") is False
    assert workspace_has_permission("editor", "members:invite") is False
    assert workspace_has_permission("editor", "workspace:manage") is False

    # S6 spec: settings/integrations/BYOK are workspace:manage — owner AND admin.
    assert workspace_has_permission("owner", "workspace:manage") is True
    assert workspace_has_permission("admin", "workspace:manage") is True
    # …but only the owner changes roles / deletes / bills the workspace.
    assert workspace_has_permission("admin", "members:change_role") is False
    assert workspace_has_permission("admin", "workspace:delete") is False

    # Viewer is a viewer: reads only.
    for perm in ("missions:read", "playbooks:read", "documents:read", "knowledge:read"):
        assert workspace_has_permission("viewer", perm) is True
    for perm in ("documents:create", "knowledge:update", "missions:execute"):
        assert workspace_has_permission("viewer", perm) is False

    # Unknown / missing roles satisfy nothing (fail-closed).
    assert workspace_has_permission("member", "agents:read") is False
    assert workspace_has_permission(None, "agents:read") is False
    assert workspace_has_permission("owner", "") is False


def test_matrix_speaks_canonical_vocabulary_only():
    """G1: the legacy ``workflows:*`` strings are renamed, not aliased —
    ``missions:*`` + ``playbooks:*`` are present, ``workflows:*`` is gone."""
    from modules.policy.roles import ROLE_PERMISSIONS

    all_perms = {p for perms in ROLE_PERMISSIONS.values() for p in perms}
    assert not {p for p in all_perms if p.startswith("workflows:")}, (
        "legacy workflows:* strings survived the G1 rename"
    )
    assert "missions:*" in all_perms and "playbooks:*" in all_perms
    resources = {p.split(":", 1)[0] for p in all_perms}
    assert resources == {
        "workspace", "members", "agents", "missions", "playbooks",
        "documents", "knowledge", "audit",
    }, f"unexpected permission resources: {sorted(resources)}"


# ---------------------------------------------------------------------------
# Empty = deny on ALL THREE key planes (F042, unconditional)
# ---------------------------------------------------------------------------

def test_empty_permissions_deny_all_planes():
    # 1) Service plane — ApiKeyService.check_permissions
    from core.services.api_key_service import ApiKeyService

    assert ApiKeyService.check_permissions(SimpleNamespace(permissions=[]), "chat") is False
    assert ApiKeyService.check_permissions(SimpleNamespace(permissions=None), "chat") is False
    assert ApiKeyService.check_permissions(SimpleNamespace(permissions=["chat"]), "chat") is True
    assert ApiKeyService.check_permissions(SimpleNamespace(permissions=["blog"]), "chat") is False
    # explicit "*" is a deliberate grant — honoured
    assert ApiKeyService.check_permissions(SimpleNamespace(permissions=["*"]), "chat") is True

    # 2) Widget plane — require_permission's dependency, driven directly
    from fastapi import HTTPException

    from api.widgets.auth import WidgetAuthContext, require_permission

    def _widget_ctx(perms):
        import uuid

        return WidgetAuthContext(
            workspace_id=uuid.uuid4(), api_key_id=uuid.uuid4(), permissions=perms
        )

    check = require_permission("chat")
    with pytest.raises(HTTPException) as ei:
        asyncio.run(check(auth=_widget_ctx([])))
    assert ei.value.status_code == 403
    with pytest.raises(HTTPException):
        asyncio.run(check(auth=_widget_ctx(None)))
    ctx = _widget_ctx(["chat"])
    assert asyncio.run(check(auth=ctx)) is ctx
    star = _widget_ctx(["*"])
    assert asyncio.run(check(auth=star)) is star

    # 3) Board plane — regression-pin: already deny-on-empty, stays that way
    from core.auth.hybrid import _sdk_key_has_scope
    from core.auth.scopes import TASKS_READ

    assert _sdk_key_has_scope([], TASKS_READ) is False
    assert _sdk_key_has_scope(None, TASKS_READ) is False
    assert _sdk_key_has_scope([TASKS_READ], TASKS_READ) is True


# ---------------------------------------------------------------------------
# super_admin passes admin gates with NO flag set (F043, unconditional)
# ---------------------------------------------------------------------------

def test_super_admin_passes_admin_gates(monkeypatch):
    import config as _config_mod
    from core.auth.roles import caller_is_admin

    # Default deployment: plane OFF — the historical F043 repro.
    monkeypatch.setattr(_config_mod.config, "POLICY_PLANE_ENABLED", False)
    monkeypatch.setattr(_config_mod.config, "POLICY_PLANE_MODE", "off", raising=False)

    assert caller_is_admin(SimpleNamespace(system_role="super_admin")) is True
    assert caller_is_admin(SimpleNamespace(system_role="admin")) is True
    assert caller_is_admin(SimpleNamespace(system_role="user")) is False
    assert caller_is_admin(SimpleNamespace(system_role="service")) is False
    assert caller_is_admin(None) is False


def test_admin_prompts_helper_delegates_to_authority():
    """The stray ``system_role in (...)`` checker in api/admin_prompts.py is
    collapsed onto caller_is_admin — super-admin passes, member refuses."""
    from fastapi import HTTPException

    from api.admin_prompts import _assert_admin

    _assert_admin(SimpleNamespace(auth_type="clerk", user=SimpleNamespace(system_role="super_admin")))
    _assert_admin(SimpleNamespace(auth_type="api_key", user=None))  # service lane unchanged
    with pytest.raises(HTTPException):
        _assert_admin(SimpleNamespace(auth_type="clerk", user=SimpleNamespace(system_role="user")))
    assert "caller_is_admin" in inspect.getsource(_assert_admin)


# ---------------------------------------------------------------------------
# One authority — the losing implementations are DELETED, not dormant
# ---------------------------------------------------------------------------

def test_one_has_permission_authority():
    """Source-grep: no surviving independent permission-membership fork.

    Repoint (don't delete) if a refactor moves these files — the guard proves
    the widget legacy arm, the service-plane empty-true and the plane-OFF
    admin branch cannot silently return.
    """
    widget_src = (_ORCH / "api" / "widgets" / "auth.py").read_text(encoding="utf-8")
    assert "policy_plane_enabled" not in widget_src, (
        "widget require_permission must not consult the governance flag"
    )
    assert "empty list = unrestricted" not in widget_src

    from core.services.api_key_service import ApiKeyService

    svc_src = inspect.getsource(ApiKeyService.check_permissions)
    assert "return True" not in svc_src, "check_permissions grew a permissive arm back"
    assert "has_permission" in svc_src, "check_permissions must delegate to the authority"

    roles_src = (_ORCH / "core" / "auth" / "roles.py").read_text(encoding="utf-8")
    assert "policy_plane_enabled" not in roles_src, (
        "caller_is_admin must be flag-independent (G2)"
    )
    assert 'role == ADMIN_ROLE' not in roles_src, "legacy exact-match admin check survived"

    # The matrix lives in the authority now — core/workspaces/permissions.py
    # (until S2 deletes it) must not carry its own copy.
    legacy = _ORCH / "core" / "workspaces" / "permissions.py"
    if legacy.exists():
        legacy_src = legacy.read_text(encoding="utf-8")
        assert '"workspace:manage"' not in legacy_src, (
            "a second ROLE_PERMISSIONS matrix survived outside modules/policy/roles.py"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
