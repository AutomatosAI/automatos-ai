"""PRD-266 §4.3/§4.11 — the bearer-prefix registry: which lane a bearer token belongs to.

A bearer's prefix names the lane that issued it. ``BEARER_RESOLVERS`` maps each registered
prefix to the resolver that turns such a bearer into a ``RequestContext``, and
``resolve_registered_bearers`` is the ONE seam ``core/auth/hybrid.py`` uses: a decorator on
``get_request_context_hybrid`` that hands a registered bearer to its resolver BEFORE the
Clerk lane, and calls the wrapped function, unchanged, for everything else (a Clerk JWT, an
API key, no credential at all, a bearer no lane claims).

* ``amd_`` — a paired phone or browser (PRD-266): ``core.auth.device.resolve_device_context``.
* PRD-257's runner and workload-identity tokens add their own entry here later, never another
  branch in ``hybrid.py``.

A resolver is ``async (request, token) -> RequestContext``; it answers the request itself
(an ``HTTPException`` for a refused bearer) and never falls through to the Clerk lane, so a
registered token is never shown to Clerk.

``functools.wraps`` keeps FastAPI's view of the wrapped function: ``inspect.signature``
follows ``__wrapped__`` to ``(request: Request)``, so every
``Depends(get_request_context_hybrid)`` and every
``app.dependency_overrides[get_request_context_hybrid]`` keep working.
"""
from __future__ import annotations

import functools
from types import MappingProxyType
from typing import Awaitable, Callable, Mapping, Optional

from fastapi import Request

from core.auth.dependencies import RequestContext
from core.auth.device import resolve_device_context

BEARER_SCHEME = "bearer "
# A paired phone's or browser's device token (services/mobile/pairing.py mints it).
DEVICE_TOKEN_PREFIX = "amd_"

BearerResolver = Callable[[Request, str], Awaitable[RequestContext]]
ContextDependency = Callable[[Request], Awaitable[RequestContext]]

BEARER_RESOLVERS: Mapping[str, BearerResolver] = MappingProxyType({
    DEVICE_TOKEN_PREFIX: resolve_device_context,
})


def bearer_token(request: Request) -> Optional[str]:
    """The token of an ``Authorization: Bearer <token>`` header, or ``None``."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    if not auth.lower().startswith(BEARER_SCHEME):
        return None
    return auth.split(" ", 1)[1].strip()


def resolver_for(token: Optional[str]) -> Optional[BearerResolver]:
    """The resolver registered for this token's prefix, or ``None`` (not a registered lane)."""
    if not isinstance(token, str) or not token:
        return None
    for prefix, resolver in BEARER_RESOLVERS.items():
        if token.startswith(prefix):
            return resolver
    return None


def resolve_registered_bearers(wrapped: ContextDependency) -> ContextDependency:
    """Decorate a request-context dependency: registered bearers first, then ``wrapped``."""

    @functools.wraps(wrapped)
    async def resolve(request: Request) -> RequestContext:
        token = bearer_token(request)
        resolver = resolver_for(token)
        if resolver is None:
            return await wrapped(request)
        return await resolver(request, token)

    return resolve


__all__ = [
    "BEARER_RESOLVERS",
    "DEVICE_TOKEN_PREFIX",
    "bearer_token",
    "resolve_registered_bearers",
    "resolver_for",
]
