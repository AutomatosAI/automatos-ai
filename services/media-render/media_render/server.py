"""The media-render HTTP service (aiohttp, as the workspace worker is).

US-101 ships /health and the X-Internal-Token gate, copied from the worker
(services/workspace-worker/main.py): /health is always open, and every other
path needs the token when one is configured. Boot refuses production without
one (boot.token_problems). US-102 adds /render and /tts behind the gate.
"""

from __future__ import annotations

import hmac
import logging

from aiohttp import web

from .config import Settings
from .versions import read_versions

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"
PUBLIC_PATHS = frozenset({"/health"})


def _token_matches(supplied: str, expected: str) -> bool:
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def create_app(settings: Settings) -> web.Application:
    @web.middleware
    async def internal_token(request: web.Request, handler):
        if request.path in PUBLIC_PATHS:
            return await handler(request)
        if settings.internal_token and not _token_matches(
            request.headers.get(TOKEN_HEADER, ""), settings.internal_token
        ):
            return web.json_response({"error": "Unauthorized"}, status=401)
        return await handler(request)

    versions = read_versions(settings.versions_path)

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy", "service": "media-render", "versions": versions})

    app = web.Application(middlewares=[internal_token])
    app.router.add_get("/health", health)
    return app


def serve(settings: Settings) -> None:
    logger.info("media-render listening on %s:%d", settings.bind_host, settings.port)
    web.run_app(
        create_app(settings),
        host=settings.bind_host,
        port=settings.port,
        access_log=None,
        print=None,
    )
