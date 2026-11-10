"""The per-IP limit on the routes a server key reaches (PRD-274 Wave 1, §5.2).

With the policy plane on, slowapi limits every route to ``APP_RATE_LIMIT_PER_MINUTE`` per client IP
(``main.py``). A product's backend sends all its users' requests from one address, so on the routes
a server key reaches, that limit would cut in long before the key's own ceiling
(``services.key_limits``): PRD-274 asks for "a per-key ceiling instead of the per-IP one". Those
routes are exempted from slowapi (``exempt_server_key_routes``) and limit inside their gate, once the
caller is known:

* **A server key that authenticates** meets its per-key limits, not the IP's.
* **Anyone else** (members, the anonymous local lane) meets the same per-IP limit as slowapi's, for
  that route.
* **A key that fails to authenticate** is counted against its IP; over the limit, the next attempt
  is refused before the key is looked up. A made-up ``ak_srv_`` prefix never skips the IP limit.

With the plane off, slowapi isn't running and neither is any of this, as before.
"""
from __future__ import annotations

from typing import Any, Iterable, List

from fastapi import HTTPException, Request, status

from core.security.ip_bucket import ip_rate_bucket

WINDOW_SECONDS = 60
LIMITER_LABEL = "server-key route IP limiter"
CALLER_BUCKET = "ip:{route}"
FAILED_KEY_BUCKET = "ip:failed-server-key"
TOO_MANY = "Too many requests from this address; try again shortly"
TOO_MANY_FAILED_KEYS = "Too many requests with an invalid key from this address; try again shortly"


def plane_on() -> bool:
    """Whether the policy plane is on (any stage but off), when slowapi limits every route."""
    from config import config

    return bool(getattr(config, "POLICY_PLANE_ENABLED", False))


def per_minute() -> int:
    """The per-IP limit a minute, the same number slowapi applies."""
    from config import config

    return int(config.APP_RATE_LIMIT_PER_MINUTE)


def _store(store: Any) -> Any:
    """The given window (tests), else the shared Redis sliding window."""
    from api.widgets.rate_limit import RateLimitStore

    return store or RateLimitStore()


def _bucket(request: Request, prefix: str) -> str:
    """The caller's address as a bucket: one IPv4 address or one IPv6 /64."""
    from api.credentials import get_client_ip

    return ip_rate_bucket(get_client_ip(request), prefix)


def _route(request: Request) -> str:
    """The matched route's endpoint, as slowapi names it, else the path."""
    endpoint = request.scope.get("endpoint")
    return f"{endpoint.__module__}.{endpoint.__name__}" if endpoint else request.url.path


def _refused(detail: str, retry_after: int) -> HTTPException:
    """A 429 that says when to retry."""
    return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail,
                         headers={"Retry-After": str(max(1, int(retry_after)))})


def limit_this_caller(request: Request, store: Any = None) -> None:
    """A caller that isn't a server key: the per-IP limit slowapi would apply to this route."""
    if not plane_on():
        return
    allowed, _limit, _remaining, reset = _store(store).check(
        _bucket(request, CALLER_BUCKET.format(route=_route(request))), per_minute(), WINDOW_SECONDS,
        label=LIMITER_LABEL)
    if not allowed:
        raise _refused(TOO_MANY, reset)


def refuse_after_failed_keys(request: Request, store: Any = None) -> None:
    """Before a key is looked up: refused when this address has sent too many keys that failed."""
    if plane_on() and _store(store).recent(_bucket(request, FAILED_KEY_BUCKET), WINDOW_SECONDS) >= per_minute():
        raise _refused(TOO_MANY_FAILED_KEYS, WINDOW_SECONDS)


def note_failed_key(request: Request, store: Any = None) -> None:
    """A key that failed to authenticate counts against its address."""
    if plane_on():
        _store(store).check(_bucket(request, FAILED_KEY_BUCKET), per_minute(), WINDOW_SECONDS, quiet=True,
                            label=LIMITER_LABEL)


def exempt_server_key_routes(limiter: Any, routes: Iterable[Any]) -> List[str]:
    """Exempts from slowapi every route whose gate carries the server-key marker; their names."""
    from fastapi.routing import APIRoute, iter_route_contexts

    from core.auth.server_key_gate import KEY_SCOPE_MARKER_ATTR

    exempted = []
    for route in iter_route_contexts(list(routes)):
        if not isinstance(route.original_route, APIRoute):
            continue
        if any(getattr(dep.call, KEY_SCOPE_MARKER_ATTR, None) for dep in route.dependant.dependencies):
            limiter.exempt(route.endpoint)
            exempted.append(f"{route.endpoint.__module__}.{route.endpoint.__name__}")
    return exempted
