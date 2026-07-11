"""PRD-195 S2 (P2-14) — the workspace-permission gate: auth-lane contract.

Pure tests (fake ctx + MagicMock session — the ``test_p2w0_cockpit_reach.py``
idiom) for ``core/auth/workspace_permission.py``. The lane table in the module
docstring is the spec; every lane is pinned here:

- clerk member: viewer denied on writes, editor allowed (matrix-driven);
- owner fallback (legacy workspaces without a member row) passes;
- non-member clerk user refused;
- ``super_admin`` bypasses — the ONLY bypass (G3 narrowed: plain ``admin``,
  including the env-API-key principal, is refused without membership);
- ``anonymous`` (REQUIRE_AUTH=false / AUTH_EDITION=local) acts as owner of its
  resolved workspace — the local edition keeps working with zero Clerk env
  (PRD-175 regression);
- ``sdk_key`` never satisfies workspace gates;
- workspace_id path-param mismatch refuses non-super-admins;
- the FastAPI dependency 403s with the permission named, and every produced
  dependency carries the sweep marker.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# Blessed fake-POSTGRES preamble — the import below pulls the config chain via
# core.auth.hybrid; nothing in this file touches a DB (CI's real POSTGRES_*
# makes these setdefaults no-ops there).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.auth.workspace_permission import (  # noqa: E402
    PERMISSION_MARKER_ATTR,
    require_workspace_permission,
    resolve_workspace_role,
    workspace_permission_granted,
)


def _ctx(
    *,
    system_role="user",
    clerk_user_id=None,
    auth_type="clerk",
    workspace_id=None,
    user=...,
):
    if user is ...:
        user = SimpleNamespace(
            id="u", email=None, role="user",
            system_role=system_role, clerk_user_id=clerk_user_id,
        )
    return SimpleNamespace(
        user=user,
        workspace_id=workspace_id or uuid.uuid4(),
        auth_type=auth_type,
    )


def _db(member_role=None, owner_row=None):
    """MagicMock session: first execute → member lookup, second → owner fallback."""
    db = MagicMock()
    member = MagicMock()
    member.fetchone.return_value = (member_role,) if member_role else None
    owner = MagicMock()
    owner.fetchone.return_value = owner_row
    db.execute.side_effect = [member, owner]
    return db


# ---------------------------------------------------------------------------
# clerk lane — the matrix decides
# ---------------------------------------------------------------------------

def test_viewer_denied_editor_allowed_writes():
    viewer_ctx = _ctx(clerk_user_id="clerk_v")
    assert workspace_permission_granted(_db("viewer"), viewer_ctx, "agents:create") is False
    assert workspace_permission_granted(_db("viewer"), viewer_ctx, "documents:read") is True

    editor_ctx = _ctx(clerk_user_id="clerk_e")
    assert workspace_permission_granted(_db("editor"), editor_ctx, "agents:create") is True
    assert workspace_permission_granted(_db("editor"), editor_ctx, "missions:execute") is True
    assert workspace_permission_granted(_db("editor"), editor_ctx, "documents:delete") is False
    assert workspace_permission_granted(_db("editor"), editor_ctx, "workspace:manage") is False

    admin_ctx = _ctx(clerk_user_id="clerk_a")
    assert workspace_permission_granted(_db("admin"), admin_ctx, "documents:delete") is True
    assert workspace_permission_granted(_db("admin"), admin_ctx, "workspace:manage") is True


def test_owner_fallback_and_missing_member_denied():
    # Legacy workspace: no member row, but workspaces.owner_id matches → owner.
    legacy_owner = _ctx(clerk_user_id="clerk_o")
    assert workspace_permission_granted(_db(None, owner_row=(1,)), legacy_owner, "workspace:manage") is True

    # No member row, not the owner → refused.
    stranger = _ctx(clerk_user_id="clerk_s")
    assert workspace_permission_granted(_db(None, owner_row=None), stranger, "agents:read") is False


def test_resolve_workspace_role_lanes():
    assert resolve_workspace_role(_db("editor"), _ctx(clerk_user_id="c")) == "editor"
    assert resolve_workspace_role(_db(None, owner_row=(1,)), _ctx(clerk_user_id="c")) == "owner"
    assert resolve_workspace_role(_db("editor"), _ctx(clerk_user_id=None)) is None
    assert resolve_workspace_role(None, _ctx(clerk_user_id="c")) is None
    no_ws = _ctx(clerk_user_id="c")
    no_ws.workspace_id = None
    assert resolve_workspace_role(_db("editor"), no_ws) is None


# ---------------------------------------------------------------------------
# super_admin — the only bypass (G3)
# ---------------------------------------------------------------------------

def test_super_admin_bypasses_without_touching_db():
    db = MagicMock()
    db.execute.side_effect = AssertionError("super-admin must short-circuit before any DB read")
    ctx = _ctx(system_role="super_admin", clerk_user_id="clerk_su")
    assert workspace_permission_granted(db, ctx, "workspace:delete") is True


def test_system_admin_no_longer_bypasses():
    """G3 narrowed the old decorator's (admin, super_admin) bypass: a plain
    system-admin without membership — including the env-API-key principal,
    hybrid.py mints system_role='admin' — is refused on workspace gates."""
    clerk_admin = _ctx(system_role="admin", clerk_user_id="clerk_staff")
    assert workspace_permission_granted(_db(None, owner_row=None), clerk_admin, "agents:create") is False

    env_key = _ctx(system_role="admin", clerk_user_id=None, auth_type="api_key")
    assert workspace_permission_granted(_db(), env_key, "agents:create") is False

    # …but a system-admin who IS a member works through their membership.
    member_admin = _ctx(system_role="admin", clerk_user_id="clerk_staff")
    assert workspace_permission_granted(_db("editor"), member_admin, "agents:create") is True


# ---------------------------------------------------------------------------
# anonymous / local edition — PRD-175 regression
# ---------------------------------------------------------------------------

def test_local_edition_anonymous_writes_still_work():
    """AUTH_EDITION=local / REQUIRE_AUTH=false mints an anonymous ctx with no
    clerk identity — the trusted single-user posture owns its workspace and
    must not 403 on every write."""
    db = MagicMock()
    db.execute.side_effect = AssertionError("anonymous lane must not query membership")
    anon = _ctx(auth_type="anonymous", user=SimpleNamespace(
        id=None, email=None, role="user", system_role="user", clerk_user_id=None,
    ))
    assert workspace_permission_granted(db, anon, "agents:create") is True
    assert workspace_permission_granted(db, anon, "workspace:manage") is True


# ---------------------------------------------------------------------------
# sdk_key — never a workspace member
# ---------------------------------------------------------------------------

def test_sdk_key_never_satisfies_workspace_gates():
    db = MagicMock()
    db.execute.side_effect = AssertionError("sdk_key lane must not query membership")
    sdk = _ctx(auth_type="sdk_key", user=SimpleNamespace(
        id="sdk:1", email=None, role="service", system_role="service", clerk_user_id=None,
    ))
    for perm in ("agents:read", "documents:read", "agents:create", "workspace:manage"):
        assert workspace_permission_granted(db, sdk, perm) is False


def test_missing_user_denied():
    ctx = SimpleNamespace(user=None, workspace_id=uuid.uuid4(), auth_type="clerk")
    assert workspace_permission_granted(MagicMock(), ctx, "agents:read") is False


# ---------------------------------------------------------------------------
# The FastAPI dependency — 403 naming the permission, marker for the sweep
# ---------------------------------------------------------------------------

def _request(path_params=None):
    return SimpleNamespace(path_params=path_params or {})


def test_dependency_403_names_permission_and_returns_ctx():
    dep = require_workspace_permission("agents:create")
    assert getattr(dep, PERMISSION_MARKER_ATTR) == "agents:create"

    viewer = _ctx(clerk_user_id="clerk_v")
    with pytest.raises(HTTPException) as ei:
        asyncio.run(dep(_request(), ctx=viewer, db=_db("viewer")))
    assert ei.value.status_code == 403
    assert "agents:create" in ei.value.detail

    editor = _ctx(clerk_user_id="clerk_e")
    assert asyncio.run(dep(_request(), ctx=editor, db=_db("editor"))) is editor


def test_dependency_workspace_path_param_mismatch_refuses():
    """A workspace_id PATH param addressed to another tenant refuses — the
    resolved role belongs to ctx.workspace_id, not the path's."""
    dep = require_workspace_permission("members:invite")
    ws = uuid.uuid4()
    other = uuid.uuid4()

    owner = _ctx(clerk_user_id="clerk_o", workspace_id=ws)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(dep(_request({"workspace_id": str(other)}), ctx=owner, db=_db("owner")))
    assert ei.value.status_code == 403

    # Matching path param passes through to the normal role check.
    owner2 = _ctx(clerk_user_id="clerk_o", workspace_id=ws)
    assert asyncio.run(
        dep(_request({"workspace_id": str(ws)}), ctx=owner2, db=_db("owner"))
    ) is owner2

    # The super-admin operator lane may cross workspaces (PRD-143 posture).
    su = _ctx(system_role="super_admin", clerk_user_id="clerk_su", workspace_id=ws)
    assert asyncio.run(
        dep(_request({"workspace_id": str(other)}), ctx=su, db=MagicMock())
    ) is su


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
