"""PRD-262 US-053: the Kokoro model loads in the background once the service listens.

/health answers while the model is still loading, a line that arrives during the
load waits for it instead of loading the model again, and the job log says when
the load finished (media-render.yml waits for that line before its first POST).
The load itself is a stand-in here; test_kokoro.py speaks with the real model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from functools import lru_cache
from typing import List

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from helpers import TOKEN
from media_render import kokoro_tts, server
from media_render.kokoro_tts import SpokenLine
from media_render.server import TOKEN_HEADER, create_app, warm_voice_in_background
from media_render.tts import Speaker
from test_server import StubPipeline, fake_synthesize, settle

AUTH = {TOKEN_HEADER: TOKEN}
LOADED = "kokoro: the voice model loaded in the background"


class BlockingLoad:
    """A model load that holds until released, counting its calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.release = threading.Event()

    def __call__(self, settings) -> float:
        self.calls += 1
        assert self.release.wait(timeout=30), "the test never released the load"
        return 1.25


def _client(settings, speaker: Speaker) -> TestClient:
    app = create_app(settings, pipeline=StubPipeline(), speaker=speaker, library={})
    return TestClient(TestServer(app))


def test_health_answers_while_the_model_loads_and_the_log_says_when_it_finished(settings, caplog):
    async def go():
        load = BlockingLoad()
        try:
            async with _client(settings, Speaker(settings, synthesize=fake_synthesize, load=load)) as client:
                with caplog.at_level(logging.INFO, logger="media_render.server"):
                    task = warm_voice_in_background(client.app)
                    await settle(lambda: load.calls == 1)
                    health = await asyncio.wait_for(client.get("/health"), timeout=5)
                    assert health.status == 200, "/health waited for the model"
                    assert not task.done()
                    load.release.set()
                    await asyncio.wait_for(task, timeout=10)
                assert any(f"{LOADED} in 1.25 s" in record.getMessage() for record in caplog.records)
        finally:
            load.release.set()

    asyncio.run(go())


def test_a_line_spoken_during_the_load_waits_for_it_and_the_model_loads_once(settings):
    async def go():
        load = BlockingLoad()
        spoken: List[str] = []

        def synthesize(text, output, settings, **options) -> SpokenLine:
            assert load.release.is_set(), "a line was spoken before the model finished loading"
            spoken.append(text)
            return fake_synthesize(text, output, settings, **options)

        try:
            async with _client(settings, Speaker(settings, synthesize=synthesize, load=load)) as client:
                task = warm_voice_in_background(client.app)
                await settle(lambda: load.calls == 1)
                request = {"lines": [{"id": "l01", "text": "Hello."}]}
                tts = asyncio.ensure_future(client.post("/tts", data=json.dumps(request), headers=AUTH))
                await asyncio.sleep(0.2)
                assert not tts.done() and spoken == []
                load.release.set()
                response = await asyncio.wait_for(tts, timeout=10)
                assert response.status == 200 and spoken == ["Hello."]
                await asyncio.wait_for(task, timeout=10)
                assert load.calls == 1
        finally:
            load.release.set()

    asyncio.run(go())


def test_a_load_that_fails_is_logged_and_the_service_keeps_serving(settings, caplog):
    async def go():
        def broken(settings) -> float:
            raise RuntimeError("no model at that path")

        async with _client(settings, Speaker(settings, synthesize=fake_synthesize, load=broken)) as client:
            with caplog.at_level(logging.INFO, logger="media_render.server"):
                await asyncio.wait_for(warm_voice_in_background(client.app), timeout=10)
            assert any("did not load in the background" in record.getMessage() for record in caplog.records)
            assert not any(LOADED in record.getMessage() for record in caplog.records)
            assert (await client.get("/health")).status == 200

    asyncio.run(go())


def test_serve_starts_the_load_once_every_site_is_listening(settings, monkeypatch):
    """web.run_app calls its ``print`` after its sites start: serve hangs the load on it."""
    captured = {}
    warmed = []
    app = web.Application()
    monkeypatch.setattr(server, "create_app", lambda received: app)
    monkeypatch.setattr(server.web, "run_app", lambda target, **options: captured.update(app=target, **options))
    monkeypatch.setattr(server, "warm_voice_in_background", warmed.append)
    server.serve(settings)
    assert captured["app"] is app and (captured["host"], captured["port"]) == (settings.bind_host, settings.port)
    assert warmed == [], "the load started before the listener was up"
    captured["print"]("======== Running on http://0.0.0.0:8090 ========")
    assert warmed == [app]


def test_the_load_reads_the_configured_files_and_a_second_load_finds_the_model_loaded(settings, monkeypatch):
    built = []
    monkeypatch.setattr(kokoro_tts, "_model", lru_cache(maxsize=1)(lambda *paths: built.append(paths)))
    assert kokoro_tts.load_model(settings) >= 0
    assert kokoro_tts.load_model(settings) >= 0
    assert built == [(settings.kokoro_model_path, settings.kokoro_voices_path, settings.espeak_data_path)]
