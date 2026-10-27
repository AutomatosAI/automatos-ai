"""The HTTP API with a stand-in model and encoder: the bearer gate, the refusals and their codes,
the streamed answer, the concurrency limit and the warm-up. test_speech.py speaks for real.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import replace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from helpers import AUTH, TOKEN, FakeEncoder, FakeEngine, catalogue
from voice_tts import server
from voice_tts.server import create_app, token_matches, warm_in_background

SECRET_TEXT = "The quarterly numbers for Project Nightingale are confidential."


def _client(settings, tmp_path, engine=None) -> TestClient:
    app = create_app(settings, engine=engine or FakeEngine(), catalogue=catalogue(tmp_path), encoder=FakeEncoder.start)
    return TestClient(TestServer(app))


def _run(coro):
    return asyncio.run(coro)


async def _settle(condition, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "the condition never held"
        await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- the bearer gate


@pytest.mark.parametrize("header, ok", [
    (f"Bearer {TOKEN}", True),
    (f"Bearer {TOKEN}x", False),
    (f"bearer {TOKEN}", False),
    (TOKEN, False),
    ("Bearer ", False),
    ("", False),
])
def test_the_bearer_token_is_compared_exactly(header, ok):
    assert token_matches(header, TOKEN) is ok


def test_the_compare_is_constant_time(monkeypatch):
    calls = []
    real = server.hmac.compare_digest
    monkeypatch.setattr(server.hmac, "compare_digest", lambda a, b: calls.append((a, b)) or real(a, b))

    token_matches(f"Bearer {TOKEN}", TOKEN)

    assert calls == [(TOKEN.encode(), TOKEN.encode())]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
def test_speak_without_the_token_is_a_401(settings, tmp_path, headers):
    async def go():
        async with _client(settings, tmp_path) as client:
            response = await client.post("/speak", json={"text": "Hi there."}, headers=headers)
            assert response.status == 401
            assert (await response.json())["code"] == "unauthorized"

    _run(go())


def test_health_ready_and_voices_need_no_token(settings, tmp_path):
    async def go():
        async with _client(settings, tmp_path) as client:
            assert (await (await client.get("/healthz")).json()) == {"ready": True}
            assert (await client.get("/readyz")).status == 200
            voices = await (await client.get("/voices")).json()
            assert voices["default"] == "af_heart" and voices["voices"][0]["id"] == "af_heart"

    _run(go())


# --------------------------------------------------------------------------- the answer


def test_speak_streams_mpeg_with_the_voice_and_no_store(settings, tmp_path):
    engine = FakeEngine()

    async def go():
        async with _client(settings, tmp_path, engine) as client:
            body = {"text": "Your board has three tasks waiting. Two are due today!", "voice": "bm_george",
                    "speed": 1.5}
            response = await client.post("/speak", json=body, headers=AUTH)
            assert response.status == 200
            assert response.headers["Content-Type"] == "audio/mpeg"
            assert response.headers["X-Voice-Id"] == "bm_george"
            assert response.headers["Cache-Control"] == "no-store"
            audio = await response.read()
            assert audio.startswith(b"\xff\xf3")

    _run(go())
    assert [(text, voice, speed, lang) for text, voice, speed, lang in engine.spoken] == [
        ("Your board has three tasks waiting.", "bm_george", 1.2, "en-gb"),
        ("Two are due today!", "bm_george", 1.2, "en-gb"),
    ]
    encoder = FakeEncoder.instances[-1]
    assert encoder.closed
    pause = bytes(2 * int(settings.sentence_pause_seconds * 24000))
    assert encoder.fed == [b"\x01\x00" * 50, pause, b"\x01\x00" * 50]


def test_a_sentence_with_nothing_to_say_is_skipped(settings, tmp_path):
    engine = FakeEngine(refuse=("###",))

    async def go():
        async with _client(settings, tmp_path, engine) as client:
            body = {"text": "#### #### #### ####. Here is the summary."}
            response = await client.post("/speak", json=body, headers=AUTH)
            assert response.status == 200
            await response.read()

    _run(go())
    assert [spoken[0] for spoken in engine.spoken] == ["Here is the summary."]


def test_a_text_with_nothing_to_say_at_all_is_a_422_not_a_broken_stream(settings, tmp_path):
    async def go():
        async with _client(settings, tmp_path, FakeEngine(refuse=("#",))) as client:
            response = await client.post("/speak", json={"text": "#### ####"}, headers=AUTH)
            assert response.status == 422
            assert (await response.json())["code"] == "tts_bad_request"

    _run(go())


@pytest.mark.parametrize("body, status, code", [
    ({"text": ""}, 422, "tts_bad_request"),
    ({"text": "Hi", "voice": "af_nobody"}, 422, "tts_bad_request"),
    ({"text": "Hi", "speed": "fast"}, 422, "tts_bad_request"),
    ({"text": "x" * 501}, 413, "tts_too_long"),
])
def test_a_body_it_cannot_speak_is_refused_with_its_code(settings, tmp_path, body, status, code):
    async def go():
        async with _client(settings, tmp_path) as client:
            response = await client.post("/speak", json=body, headers=AUTH)
            assert response.status == status
            assert (await response.json())["code"] == code

    _run(go())


def test_a_body_that_is_not_json_is_a_422(settings, tmp_path):
    async def go():
        async with _client(settings, tmp_path) as client:
            response = await client.post("/speak", data=b"not json", headers=AUTH)
            assert response.status == 422

    _run(go())


def test_the_text_is_never_logged(settings, tmp_path, caplog):
    async def go():
        async with _client(settings, tmp_path) as client:
            response = await client.post("/speak", json={"text": SECRET_TEXT}, headers=AUTH)
            await response.read()

    with caplog.at_level(logging.DEBUG):
        _run(go())
    assert any("speak ok: voice=af_heart chars=" in record.getMessage() for record in caplog.records)
    assert "Nightingale" not in caplog.text


# --------------------------------------------------------------------------- warming and busy


def test_while_the_model_loads_speak_is_a_503_tts_warming(settings, tmp_path):
    async def go():
        async with _client(settings, tmp_path, FakeEngine(loaded=False)) as client:
            assert (await (await client.get("/healthz")).json()) == {"ready": False}
            assert (await client.get("/readyz")).status == 503
            response = await client.post("/speak", json={"text": "Hi there."}, headers=AUTH)
            assert response.status == 503
            assert (await response.json())["code"] == "tts_warming"

    _run(go())


def test_the_model_loads_in_the_background_and_then_speak_answers(settings, tmp_path):
    engine = FakeEngine(loaded=False)

    async def go():
        async with _client(settings, tmp_path, engine) as client:
            await warm_in_background(client.app)
            assert (await (await client.get("/healthz")).json()) == {"ready": True}
            response = await client.post("/speak", json={"text": "Hi there."}, headers=AUTH)
            assert response.status == 200

    _run(go())


def test_past_the_concurrency_limit_speak_is_a_429_tts_busy_at_once(settings, tmp_path):
    engine = FakeEngine()
    engine.hold = threading.Event()
    one_at_a_time = replace(settings, max_concurrent=1)

    async def go():
        async with _client(one_at_a_time, tmp_path, engine) as client:
            first = asyncio.ensure_future(client.post("/speak", json={"text": "The first answer."}, headers=AUTH))
            await _settle(engine.started.is_set)
            second = await asyncio.wait_for(
                client.post("/speak", json={"text": "The second answer."}, headers=AUTH), timeout=5)
            assert second.status == 429
            assert (await second.json())["code"] == "tts_busy"
            engine.hold.set()
            answered = await first
            assert answered.status == 200
            await answered.read()
            assert client.app[server.STATE].active == 0
            third = await client.post("/speak", json={"text": "The third answer."}, headers=AUTH)
            assert third.status == 200

    try:
        _run(go())
    finally:
        engine.hold.set()


def test_an_open_service_speaks_without_a_token_in_development(settings, tmp_path):
    async def go():
        async with _client(replace(settings, token=""), tmp_path) as client:
            response = await client.post("/speak", data=json.dumps({"text": "Hi there."}))
            assert response.status == 200

    _run(go())
