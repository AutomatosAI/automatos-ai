"""
Widget CORS Middleware (ASGI-native)
=====================================

Dynamic CORS for ``/api/widgets/*`` routes.  Implemented as a pure ASGI
middleware (not ``BaseHTTPMiddleware``) so that ``StreamingResponse`` /
SSE connections are **not buffered**.
"""

import logging
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class WidgetCORSMiddleware:
    """Add CORS headers to /api/widgets/* without buffering responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if not path.startswith("/api/widgets"):
            await self.app(scope, receive, send)
            return

        # Extract Origin header from raw ASGI headers
        headers = dict(scope.get("headers", []))
        origin = (headers.get(b"origin") or b"").decode("latin-1")

        # Handle OPTIONS preflight
        if scope["method"] == "OPTIONS":
            if not origin:
                response_headers = [
                    (b"content-type", b"text/plain"),
                ]
                await send({"type": "http.response.start", "status": 400, "headers": response_headers})
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

        # For actual requests, inject CORS headers into the response
        async def send_with_cors(message: dict) -> None:
            if message["type"] == "http.response.start" and origin:
                headers = list(message.get("headers", []))
                headers.append((b"access-control-allow-origin", origin.encode("latin-1")))
                headers.append((b"access-control-allow-credentials", b"true"))
                headers.append((b"vary", b"Origin"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_cors)
