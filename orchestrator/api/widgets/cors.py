"""
Widget CORS Middleware (ASGI-native)
=====================================

Dynamic CORS for the storefront-widget + dashboard-Sites API surfaces.
Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) so
that ``StreamingResponse`` / SSE connections are **not buffered**.

Path coverage (PRD-008-A):
- ``/api/widgets/*``  — storefront widget calls (any storefront origin)
- ``/api/sites/*``    — Automatos dashboard Sites CRUD (admin operations)

Both surfaces share the ``WIDGET_ORIGIN_ALLOWLIST`` env var. When empty,
all origins are allowed (development default). The actual security
boundary on /api/sites is the JWT in cookies, not CORS — CORS just lets
the dashboard browser issue the request in the first place.
"""

import logging
from starlette.types import ASGIApp, Receive, Scope, Send
from config import config

logger = logging.getLogger(__name__)

# Path prefixes this middleware governs. First-match short-circuits the
# rest of the ASGI stack to inject CORS headers; non-matching paths flow
# through to FastAPI's default CORSMiddleware (which has its own allowlist).
COVERED_PATH_PREFIXES: tuple[str, ...] = (
    "/api/widgets",
    "/api/sites",
)

# Explicit origin allowlist — comma-separated in env var.
# Only these origins may make credentialed requests to widget endpoints.
_RAW_ALLOWLIST = config.WIDGET_ORIGIN_ALLOWLIST or ""
WIDGET_ORIGIN_ALLOWLIST: set[str] = {
    o.strip().rstrip("/") for o in _RAW_ALLOWLIST.split(",") if o.strip()
}


def _origin_allowed(origin: str) -> bool:
    """Return True if *origin* is in the configured allowlist.

    When the allowlist is empty (not configured), ALL origins are allowed
    for backwards-compatibility during development.  In production the
    env var should always be set.
    """
    if not WIDGET_ORIGIN_ALLOWLIST:
        return True
    return origin.rstrip("/") in WIDGET_ORIGIN_ALLOWLIST


def _path_is_covered(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in COVERED_PATH_PREFIXES)


class WidgetCORSMiddleware:
    """Add CORS headers to widget + Sites API paths without buffering responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if not _path_is_covered(path):
            await self.app(scope, receive, send)
            return

        # Extract Origin header from raw ASGI headers
        headers = dict(scope.get("headers", []))
        origin = (headers.get(b"origin") or b"").decode("latin-1")

        # Handle OPTIONS preflight
        if scope["method"] == "OPTIONS":
            if not origin or not _origin_allowed(origin):
                response_headers = [
                    (b"content-type", b"text/plain"),
                    (b"vary", b"Origin"),
                ]
                status = 400 if not origin else 403
                await send({"type": "http.response.start", "status": status, "headers": response_headers})
                await send({"type": "http.response.body", "body": b""})
                return

            response_headers = [
                (b"access-control-allow-origin", origin.encode("latin-1")),
                (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                (b"access-control-allow-headers", b"Authorization, Content-Type, X-Workspace-ID"),
                (b"access-control-max-age", b"86400"),
                (b"access-control-allow-credentials", b"true"),
            ]
            await send({"type": "http.response.start", "status": 200, "headers": response_headers})
            await send({"type": "http.response.body", "body": b""})
            return

        # For actual requests, inject CORS headers only for allowed origins
        allowed = _origin_allowed(origin) if origin else False

        async def send_with_cors(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"vary", b"Origin"))
                if origin and allowed:
                    headers.append((b"access-control-allow-origin", origin.encode("latin-1")))
                    headers.append((b"access-control-allow-credentials", b"true"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_cors)
