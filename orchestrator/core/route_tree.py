"""FastAPI's routes as the app serves them, included routers flattened (PRD-262 US-008).

Since FastAPI 0.137 ``include_router`` keeps each included router as ONE node of
``app.routes``, with no ``path`` and no ``endpoint``, instead of copying its routes in.
A routed request's ``scope["route"]`` is the route as its router declared it, without
the prefixes it was included under. ``fastapi.routing.iter_route_contexts`` walks the
tree and gives every served route with its full path, methods, endpoint and inherited
dependencies. Two readers of the old flat list go through this module:

* ``matching_endpoint``: slowapi's middleware looks a request's endpoint up before
  routing (``core.security.rate_limit_middleware``);
* ``route_template``: main.py's API stats key each request by its path template.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from fastapi.routing import RouteContext, iter_route_contexts
from starlette.routing import BaseRoute, Match
from starlette.types import Scope

# One FastAPI app per process; a few more only in tests, which build their own.
_TEMPLATE_CACHE_APPS = 4


def matching_endpoint(routes: Sequence[BaseRoute], scope: Scope) -> Optional[Callable[..., Any]]:
    """The endpoint of the last served route that fully matches ``scope``, or None.

    The last match, not the first: slowapi's own lookup over the old flat list took
    the last one too, so the limits it applies stay the same.
    """
    endpoint = None
    for route in iter_route_contexts(routes):
        match, _ = route.matches(scope)
        if match == Match.FULL and route.endpoint is not None:
            endpoint = route.endpoint
    return endpoint


@lru_cache(maxsize=_TEMPLATE_CACHE_APPS)
def _served_contexts(app: Any) -> Dict[int, Tuple[RouteContext, ...]]:
    """Every served route of ``app``, grouped by the declared route it serves."""
    grouped: Dict[int, Tuple[RouteContext, ...]] = {}
    for context in iter_route_contexts(app.routes):
        key = id(context.original_route)
        grouped[key] = (*grouped.get(key, ()), context)
    return grouped


def route_template(scope: Scope) -> Optional[str]:
    """The path template ``scope`` was routed to, include prefixes and all, or None.

    A route included under two prefixes has two templates; the one that matches the
    request's path wins. A route added after the first lookup falls back to the path
    its router declared.
    """
    route = scope.get("route")
    if route is None:
        return None
    app = scope.get("app")
    contexts = _served_contexts(app).get(id(route), ()) if app is not None else ()
    for context in contexts:
        match, _ = context.matches(scope)
        if match != Match.NONE:
            return context.path
    return getattr(route, "path", None)
