"""slowapi's middleware, with an endpoint lookup that sees included routers (PRD-262 US-008).

slowapi finds a request's endpoint by walking ``app.routes`` for a full match that
has an ``endpoint``. Since FastAPI 0.137 an included router is one node of that list
with no ``endpoint``, so the stock lookup finds nothing for a route of an included
router (nearly every route of this API), and slowapi then exempts the request: the
default limits main.py sets under the policy plane (F040) would apply to almost
nothing. This subclass keeps slowapi 0.1.10's dispatch line for line and looks the
endpoint up with ``core.route_tree.matching_endpoint`` instead.
"""
from __future__ import annotations

from slowapi.middleware import SlowAPIMiddleware, _should_exempt, sync_check_limits
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from core.route_tree import matching_endpoint


class RouteTreeSlowAPIMiddleware(SlowAPIMiddleware):
    """``SlowAPIMiddleware`` whose endpoint lookup walks FastAPI's route tree."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        app = request.app
        limiter = app.state.limiter
        if not limiter.enabled:
            return await call_next(request)

        handler = matching_endpoint(app.routes, request.scope)
        if _should_exempt(limiter, handler):
            return await call_next(request)

        error_response, should_inject_headers = sync_check_limits(limiter, request, handler, app)
        if error_response is not None:
            return error_response

        response = await call_next(request)
        if should_inject_headers:
            response = limiter._inject_headers(response, request.state.view_rate_limit)
        return response
