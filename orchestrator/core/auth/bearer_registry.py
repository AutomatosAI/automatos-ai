"""PRD-266 §4.3/§4.11 — the bearer-prefix registry: which lane a bearer token belongs to.

A bearer's prefix names the lane that issued it. ``BEARER_RESOLVERS`` maps each registered
prefix to the resolver that turns such a bearer into a ``RequestContext``, and
``resolve_registered_bearers`` is the ONE seam ``core/auth/hybrid.py`` uses: a decorator on
``get_request_context_hybrid`` that hands a registered bearer to its resolver BEFORE the
Clerk lane, and calls the wrapped function, unchanged, for everything else (a Clerk JWT, an
API key, no credential at all, a bearer no lane claims).

The one exception is a request with NO credential at all on a local install that runs
``LOCAL_REMOTE_ACCESS=paired`` (PRD-266 §4.3): hybrid would make it the anonymous operator,
so it is ``401 {"code": "device_pairing_required", "message": …}`` here instead and never
reaches hybrid. "No credential" is hybrid's own test, read from here: no bearer token
(``bearer_token``) and no API key (``api_key``: ``X-API-Key`` or ``Authorization: ApiKey``).
Every lane with a credential of its own is unchanged (K7): the API key, the CLI host and
session tokens and signed webhooks (their own dependencies), SDK keys (a bearer), and the
public routes (``/health``, ``POST /api/v1/mobile/pair``), which never ask for a context.
Whether the lane is open is ``services.mobile.remote_access``'s decision, imported at call
time so this module, which hybrid imports, never imports the services package at import
time.

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

from fastapi import HTTPException, Request, status

from config import config
from core.auth.dependencies import RequestContext
from core.auth.device import resolve_device_context

BEARER_SCHEME = "bearer "
API_KEY_SCHEME = "apikey "
DEVICE_PAIRING_REQUIRED = "device_pairing_required"
DEVICE_PAIRING_REQUIRED_MESSAGE = (
    "This install only answers paired devices. Pair this browser or phone with a one-time "
    "code from Settings → Phones & devices on a device already paired, or from the server console."
)
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


def api_key(request: Request) -> Optional[str]:
    """The key of an ``X-API-Key`` header or an ``Authorization: ApiKey <key>`` header, or ``None``."""
    key = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    if auth.lower().startswith(API_KEY_SCHEME):
        return auth.split(" ", 1)[1].strip()
    return None


def anonymous_lane_closed() -> bool:
    """True on a local install whose remote-access mode closes the anonymous lane (``paired``)."""
    from services.mobile import remote_access

    return not remote_access.anonymous_lane_open(config.AUTH_EDITION, config.LOCAL_REMOTE_ACCESS)


def _pairing_required() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": DEVICE_PAIRING_REQUIRED, "message": DEVICE_PAIRING_REQUIRED_MESSAGE},
    )


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
        if resolver is not None:
            return await resolver(request, token)
        if not token and not api_key(request) and anonymous_lane_closed():
            raise _pairing_required()
        return await wrapped(request)

    return resolve


__all__ = [
    "BEARER_RESOLVERS",
    "DEVICE_PAIRING_REQUIRED",
    "DEVICE_TOKEN_PREFIX",
    "anonymous_lane_closed",
    "api_key",
    "bearer_token",
    "resolve_registered_bearers",
    "resolver_for",
]
