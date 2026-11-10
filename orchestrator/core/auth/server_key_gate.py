"""A route a member with a permission, or a product's server key with a scope, may call
(PRD-274 Wave 1, §5.2).

``require_workspace_permission`` refuses every SDK key by design: a key is never a workspace
member. PRD-274 opens a named set of routes to a product's backend, each with the scope it needs.
This gate sits on those routes only; the shared ``get_request_context_hybrid`` is untouched, so no
other route starts accepting keys (``core/auth/hybrid.py`` says why that matters).

* **A server key** (``ak_srv_``) must list the route's scope. Its workspace is the key's, and an
  ``X-Workspace-ID`` header may only match it (``hybrid._resolve_sdk_key_context``). An empty
  permission list grants nothing.
* **Its per-key limits** apply next: requests and tokens a minute, under the platform's ceiling
  (``services.key_limits``). Over either is a 429.
* **A publishable key** (``ak_pub_``) is refused: it lives in a browser, so anyone can read it.
* **Anyone else** goes through ``require_workspace_permission(permission)`` exactly as before.
* **With the policy plane on,** these routes are exempt from slowapi's per-IP limit and limit in
  here instead: a server key by its own limits, anyone else by the same per-IP limit, and failed
  keys per IP before any lookup (``core.auth.server_key_ip_limit``).

The dependency carries the permission marker, so the authz sweep classifies the route as gated, and
the key scope marker, so a test can list which routes keys reach.
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from core.auth.bearer_registry import bearer_token
from core.auth.dependencies import RequestContext
from core.auth.hybrid import _sdk_key_context_in_own_session, get_request_context_hybrid
from core.auth.server_key_ip_limit import limit_this_caller, note_failed_key, refuse_after_failed_keys
from core.auth.workspace_permission import PERMISSION_MARKER_ATTR, require_workspace_permission
from core.database.database import get_db

SERVER_KEY_PREFIX = "ak_srv_"
PUBLISHABLE_KEY_PREFIX = "ak_pub_"
KEY_SCOPE_MARKER_ATTR = "_server_key_scope"
SDK_KEY_AUTH_TYPE = "sdk_key"
API_KEY_FILER = "api_key"  # created_by_type of a ticket a server key filed
PUBLISHABLE_KEY_REFUSED = "A publishable key can't call this route: use a server key, kept on your own server"


def require_workspace_permission_or_server_key(permission: str, key_scope: str) -> Callable:
    """Dependency factory: a member whose role grants ``permission``, or a server key listing
    ``key_scope``. Answers the caller's :class:`RequestContext`."""
    member_gate = require_workspace_permission(permission)

    async def _dep(request: Request, db: Any = Depends(get_db)) -> RequestContext:
        """A publishable key is refused; a server key goes through its scope and limits; anyone
        else meets the per-IP limit, then the member gate."""
        token = bearer_token(request)
        if token and token.startswith(PUBLISHABLE_KEY_PREFIX):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=PUBLISHABLE_KEY_REFUSED)
        if token and token.startswith(SERVER_KEY_PREFIX):
            return await run_in_threadpool(_server_key_context, request, key_scope)
        await run_in_threadpool(limit_this_caller, request)
        ctx = await get_request_context_hybrid(request)
        return await run_in_threadpool(member_gate, request, ctx=ctx, db=db)

    setattr(_dep, PERMISSION_MARKER_ATTR, permission)
    setattr(_dep, KEY_SCOPE_MARKER_ATTR, key_scope)
    _dep.__name__ = f"require_{permission.replace(':', '_')}_or_server_key_{key_scope.replace(':', '_')}"
    return _dep


def request_context_or_server_key(key_scope: str) -> Callable:
    """For a read route any signed-in caller may use: a server key listing ``key_scope``, or
    ``get_request_context_hybrid`` exactly as before. A publishable key is refused."""

    async def _dep(request: Request) -> RequestContext:
        """A publishable key is refused; a server key goes through its scope and limits; anyone
        else meets the per-IP limit, then signs in as before."""
        token = bearer_token(request)
        if token and token.startswith(PUBLISHABLE_KEY_PREFIX):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=PUBLISHABLE_KEY_REFUSED)
        if token and token.startswith(SERVER_KEY_PREFIX):
            return await run_in_threadpool(_server_key_context, request, key_scope)
        await run_in_threadpool(limit_this_caller, request)
        return await get_request_context_hybrid(request)

    setattr(_dep, KEY_SCOPE_MARKER_ATTR, key_scope)
    _dep.__name__ = f"request_context_or_server_key_{key_scope.replace(':', '_')}"
    return _dep


def _server_key_context(request: Request, key_scope: str) -> RequestContext:
    """The key's context once its scope and its per-key limits allow the request (threadpool)."""
    from services.key_limits import enforce_on_own_session

    refuse_after_failed_keys(request)
    try:
        ctx = _sdk_key_context_in_own_session(request, key_scope)
    except HTTPException as refused:
        if refused.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN):
            note_failed_key(request)
        raise
    return enforce_on_own_session(ctx)


def is_server_key(ctx: Any) -> bool:
    """Whether the caller is an SDK key (on these routes, always a server key)."""
    return getattr(ctx, "auth_type", None) == SDK_KEY_AUTH_TYPE
