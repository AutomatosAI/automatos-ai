"""The HTTP gate copied from the worker: /health is open; every other path needs
X-Internal-Token when one is configured (US-101; US-102 adds the routes).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from aiohttp.test_utils import TestClient, TestServer

from media_render.config import load_settings
from media_render.server import TOKEN_HEADER, create_app

TOKEN = "t0ken-for-tests"


def _get(settings, path, headers=None):
    async def go():
        async with TestClient(TestServer(create_app(settings))) as client:
            response = await client.get(path, headers=headers or {})
            return response.status, await response.text()

    return asyncio.run(go())


def test_health_is_open_without_a_token():
    status, text = _get(replace(load_settings(), internal_token=TOKEN), "/health")
    assert status == 200
    body = json.loads(text)
    assert body["status"] == "healthy" and body["service"] == "media-render"
    assert body["versions"]["hyperframes"] == "0.8.62"


def test_other_paths_need_the_token():
    settings = replace(load_settings(), internal_token=TOKEN)
    assert _get(settings, "/render")[0] == 401
    assert _get(settings, "/render", {TOKEN_HEADER: "wrong"})[0] == 401
    # The right token passes the gate; the route itself arrives in US-102.
    assert _get(settings, "/render", {TOKEN_HEADER: TOKEN})[0] == 404
