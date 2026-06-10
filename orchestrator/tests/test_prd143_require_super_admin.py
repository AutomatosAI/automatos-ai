"""PRD-143 S5 — the one canonical ``require_super_admin`` FastAPI dependency.

``core/auth/super_admin.py`` must 403 every principal except a literal
``system_role == 'super_admin'``, fail-closed on missing user/role.
``core/auth/hybrid.py`` (the 657-call-site shared auth, PRD-09 precedent)
stays untouched — the new dependency only composes
``get_request_context_hybrid``.

Principal shapes mirror the verified sources (PRD-143 §4, 2026-06-09):
  - API key     → UserContext(role='admin',   system_role='admin')    (hybrid.py:783)
  - SDK/service → UserContext(role='service', system_role='service')  (hybrid.py:616)
"""
from __future__ import annotations

import inspect
import os
import uuid

import pytest

# Dummy POSTGRES_* satisfies the config chain at import (blessed pattern,
# see test_prd143_su_executor_gate.py) — the port points at nothing so any
# fail-soft connect refuses instantly. CI exports real vars (setdefault no-ops).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import HTTPException

from core.auth.dependencies import RequestContext, UserContext
from core.auth.super_admin import require_super_admin

pytestmark = pytest.mark.asyncio

_WS = uuid.uuid4()


def _ctx(user, auth_type: str = "clerk") -> RequestContext:
    return RequestContext(workspace_id=_WS, user=user, auth_type=auth_type)


async def _assert_403(ctx: RequestContext) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_super_admin(ctx)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Super admin only"


async def test_403_for_member():
    await _assert_403(_ctx(UserContext(id="u-member", role="member", system_role="user")))


async def test_403_for_workspace_admin():
    # Workspace-level admin/owner WITHOUT the system role: refused.
    await _assert_403(_ctx(UserContext(id="u-ws-admin", role="admin", system_role="user")))
    await _assert_403(_ctx(UserContext(id="u-ws-owner", role="owner", system_role="user")))


async def test_403_for_api_key_admin():
    # hybrid.py:783 — API-key principals carry system_role='admin'.
    await _assert_403(
        _ctx(
            UserContext(id="api_key", email=None, role="admin", system_role="admin"),
            auth_type="api_key",
        )
    )


async def test_403_for_service_principal():
    # hybrid.py:616 — SDK keys carry system_role='service'.
    await _assert_403(
        _ctx(
            UserContext(id="sdk:rec-1", role="service", system_role="service"),
            auth_type="sdk_key",
        )
    )


async def test_403_when_user_or_role_missing():
    await _assert_403(_ctx(None))
    await _assert_403(_ctx(UserContext(id="u-x", role="admin", system_role=None)))
    await _assert_403(_ctx(object()))  # principal with no system_role attribute at all


async def test_returns_ctx_for_super_admin():
    ctx = _ctx(UserContext(id="u-gerard", role="admin", system_role="super_admin"))
    assert await require_super_admin(ctx) is ctx


async def test_dependency_composes_hybrid_auth():
    # S6/S7 routers rely on this exact wiring: Depends(get_request_context_hybrid).
    from core.auth.hybrid import get_request_context_hybrid

    default = inspect.signature(require_super_admin).parameters["ctx"].default
    assert getattr(default, "dependency", None) is get_request_context_hybrid
