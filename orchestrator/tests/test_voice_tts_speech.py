"""Auto's server voice — ``POST /api/voice/speech`` and ``GET /api/voice/voices`` (api/voice_speech.py).

The phone and the web quick-ask fall back to the device voice on any answer that is not audio,
so the contract is the refusals as much as the audio:

* a member who may chat (``agents:execute``, as ``POST /api/chat``) gets voice-tts's MP3 streamed
  through, chunk by chunk, with ``Cache-Control: no-store`` and ``X-Voice-Id``; a viewer gets 403;
* every refusal is ``{"code", "detail"}``: 503 ``tts_off`` without ``VOICE_TTS_URL`` (voice-tts is
  never called), 422 ``tts_bad_request``, 413 ``tts_too_long``, 429 ``tts_busy`` (voice-tts busy, or
  the workspace's ``VOICE_TTS_RATE_PER_MINUTE`` spent) and 503 ``tts_unavailable`` (voice-tts down,
  warming, refusing the token, or too slow);
* voice-tts gets ``{text, voice, speed}`` with the bearer token, the speed clamped and the default
  voice filled in; one usage line per answer names the workspace, the voice, the characters and the
  latency, never the text; the permission check's database connection is given back first;
* the voice list is voice-tts's, cached, and checked before it is passed on;
* discovery lists ``tts`` only while ``VOICE_TTS_URL`` is set, and the routes are mounted and in the
  route manifest.

Through the real router with the request context and the session overridden and voice-tts behind
httpx's MockTransport: no database, no Redis (a fake), no network.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.voice_speech as vs
import core.auth.workspace_permission as permission
from api.widgets.rate_limit import RateLimitStore
from config import config
from modules.voice import speech, speech_client
from services.mobile import discovery

ORCH = Path(__file__).resolve().parents[1]
SPEECH = "/api/voice/speech"
VOICES = "/api/voice/voices"
URL = "http://voice-tts.test:8100"
TOKEN = "core-voice-test-token"
WS = uuid.UUID("7b1f4d2e-3c5e-4f9a-8b2c-7e1ce0a0a0a1")
OTHER_WS = uuid.UUID("7b1f4d2e-3c5e-4f9a-8b2c-7e1ce0a0a0a2")
MP3_CHUNKS = (b"\xff\xf3\x64\xc4" + b"\x00" * 60, b"\xff\xf3\x64\xc4" + b"\x01" * 60)
SECRET_TEXT = "The quarterly numbers for Project Nightingale are confidential."
VOICE_LIST = {
    "default": "af_heart",
    "voices": [
        {"id": "af_heart", "name": "Heart", "accent": "en-US", "gender": "female"},
        {"id": "bm_george", "name": "George", "accent": "en-GB", "gender": "male"},
    ],
}


# --------------------------------------------------------------------------- fakes


class _Body(httpx.AsyncByteStream):
    """A response body that really streams (a ``content=`` body is read at construction)."""

    def __init__(self, chunks, between=None):
        self._chunks = chunks
        self._between = between

    async def __aiter__(self):
        for index, chunk in enumerate(self._chunks):
            if index and self._between is not None:
                await self._between()
            yield chunk


class _Voice:
    """voice-tts behind a MockTransport: records each request, answers with ``answer(request)``."""

    def __init__(self, answer=None):
        self.requests = []
        self.answer = answer or (lambda request: httpx.Response(
            200, headers={"Content-Type": "audio/mpeg", "X-Voice-Id": json.loads(request.content)["voice"]},
            stream=_Body(MP3_CHUNKS)))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.answer(request)


class _FakeRedis:
    """The sorted-set commands RateLimitStore uses, on a dict."""

    def __init__(self):
        self.z = {}

    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, redis):
        self._redis = redis
        self._ops = []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self._ops.append((name, args))
            return self
        return queue

    def execute(self):
        return [self._run(name, *args) for name, args in self._ops]

    def _run(self, name, key, *args):
        marks = self._redis.z.setdefault(key, {})
        if name == "zremrangebyscore":
            return 0
        if name == "zcard":
            return len(marks)
        if name == "zrange":
            return sorted(marks.items(), key=lambda kv: kv[1])[:1]
        if name == "zadd":
            marks.update(args[0])
            return 1
        return True


@pytest.fixture
def voice(monkeypatch) -> _Voice:
    fake = _Voice()
    monkeypatch.setattr(config, "VOICE_TTS_URL", URL)
    monkeypatch.setattr(config, "VOICE_TTS_TOKEN", TOKEN)
    monkeypatch.setattr(config, "VOICE_TTS_DEFAULT_VOICE", "af_heart")
    monkeypatch.setattr(config, "VOICE_TTS_RATE_PER_MINUTE", 60)
    monkeypatch.setattr(speech_client, "_shared", None)
    monkeypatch.setattr(speech_client, "_voices", None)
    real = speech_client.new_http_client

    def mocked() -> httpx.AsyncClient:
        made = real()
        return httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), timeout=made.timeout,
                                 headers=made.headers)

    monkeypatch.setattr(speech_client, "new_http_client", mocked)
    monkeypatch.setattr(vs, "_rate_store", RateLimitStore(redis_factory=lambda: _FakeRedis()))
    return fake


@pytest.fixture
def role(monkeypatch):
    """The caller's workspace role; an editor unless a test says otherwise."""
    holder = {"role": "editor"}
    monkeypatch.setattr(permission, "resolve_workspace_role", lambda db, ctx: holder["role"])
    return holder


def _app(workspace=WS, db=None) -> TestClient:
    app = FastAPI()
    app.include_router(vs.router)
    ctx = NS(workspace_id=workspace, user=NS(id=7, system_role="user", clerk_user_id="user_c1"), auth_type="clerk")
    app.dependency_overrides[vs.get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[vs.get_db] = lambda: db if db is not None else MagicMock()
    return TestClient(app)


def _speak(body, workspace=WS, db=None):
    return _app(workspace, db).post(SPEECH, json=body)


# --------------------------------------------------------------------------- the audio


def test_a_member_who_may_chat_gets_voice_tts_mp3_streamed_through(voice, role):
    response = _speak({"text": "  Your board has three tasks waiting.  ", "voice": "bm_george", "speed": 3})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-voice-id"] == "bm_george"
    assert response.content == b"".join(MP3_CHUNKS)
    sent = voice.requests[0]
    assert str(sent.url) == f"{URL}/speak"
    assert sent.headers["authorization"] == f"Bearer {TOKEN}"
    assert json.loads(sent.content) == {"text": "Your board has three tasks waiting.", "voice": "bm_george", "speed": 1.2}


@pytest.mark.parametrize("given, sent", [(None, 1.0), (0.1, 0.8), (1, 1.0), (1.05, 1.05)])
def test_null_voice_is_the_default_and_speed_is_clamped(voice, role, given, sent):
    response = _speak({"text": "Hello there.", "voice": None, "speed": given})

    assert response.status_code == 200
    assert json.loads(voice.requests[0].content)["voice"] == "af_heart"
    assert json.loads(voice.requests[0].content)["speed"] == sent
    assert response.headers["x-voice-id"] == "af_heart"


def test_a_field_the_server_does_not_know_is_ignored(voice, role):
    assert _speak({"text": "Hello there.", "lang": "en-us"}).status_code == 200


def test_the_relay_passes_each_chunk_on_before_the_next_exists():
    """Streamed, not buffered: the second chunk is only produced once the first was passed on."""
    async def go():
        passed_on = asyncio.Event()

        async def wait_for_the_first():
            await asyncio.wait_for(passed_on.wait(), timeout=5)

        response = httpx.Response(200, stream=_Body(MP3_CHUNKS, between=wait_for_the_first))
        stream = speech_client.SpeechStream(voice="af_heart", response=response, started=time.monotonic())
        spec = speech.SpeechRequest(text="Hi there.", voice="af_heart", speed=1.0)
        received = []
        async for chunk in speech_client.relay(stream, spec, str(WS)):
            received.append(chunk)
            passed_on.set()
        return received

    assert asyncio.run(go()) == list(MP3_CHUNKS)


def test_one_usage_line_per_answer_and_never_the_text(voice, role, caplog):
    with caplog.at_level(logging.DEBUG):
        response = _speak({"text": SECRET_TEXT, "voice": "bm_george"})

    assert response.status_code == 200
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("voice_tts_speech ")]
    assert len(lines) == 1
    assert f"workspace={WS} voice=bm_george chars={len(SECRET_TEXT)} first_byte_ms=" in lines[0]
    assert "total_ms=" in lines[0] and f"bytes={sum(map(len, MP3_CHUNKS))} outcome=ok" in lines[0]
    assert "Nightingale" not in caplog.text


def test_the_permission_checks_connection_is_given_back_before_the_stream(voice, role):
    db = MagicMock()

    assert _speak({"text": "Hello there."}, db=db).status_code == 200
    db.close.assert_called()


def test_the_client_sends_the_token_and_gives_up_on_a_connection_fast(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TTS_TOKEN", TOKEN)
    monkeypatch.setattr(config, "VOICE_TTS_TIMEOUT_SECONDS", 15.0)
    monkeypatch.setattr(config, "VOICE_TTS_CONNECT_TIMEOUT_SECONDS", 2.0)

    client = speech_client.new_http_client()

    assert client.headers["authorization"] == f"Bearer {TOKEN}"
    assert (client.timeout.connect, client.timeout.pool, client.timeout.read) == (2.0, 2.0, 15.0)
    monkeypatch.setattr(config, "VOICE_TTS_TOKEN", "")
    assert "authorization" not in speech_client.new_http_client().headers


# --------------------------------------------------------------------------- refusals


def test_without_voice_tts_url_it_is_503_tts_off_and_voice_tts_is_never_called(voice, role, monkeypatch):
    monkeypatch.setattr(config, "VOICE_TTS_URL", "")

    response = _speak({"text": "Hello there."})

    assert response.status_code == 503
    assert response.json() == {"code": "tts_off", "detail": response.json()["detail"]}
    assert "VOICE_TTS_URL" in response.json()["detail"]
    assert voice.requests == []


@pytest.mark.parametrize("body, status, code", [
    ({"text": ""}, 422, "tts_bad_request"),
    ({"text": "   "}, 422, "tts_bad_request"),
    ({"text": 42}, 422, "tts_bad_request"),
    ({"voice": "af_heart"}, 422, "tts_bad_request"),
    ({"text": "Hi", "voice": "Heart"}, 422, "tts_bad_request"),
    ({"text": "Hi", "voice": 7}, 422, "tts_bad_request"),
    ({"text": "Hi", "speed": "fast"}, 422, "tts_bad_request"),
    ({"text": "Hi", "speed": True}, 422, "tts_bad_request"),
    (["Hi"], 422, "tts_bad_request"),
    ({"text": "x" * 501}, 413, "tts_too_long"),
])
def test_a_body_it_cannot_speak_is_refused_before_voice_tts_is_asked(voice, role, body, status, code):
    response = _speak(body)

    assert response.status_code == status
    assert response.json()["code"] == code and response.json()["detail"]
    assert voice.requests == []


def test_text_of_500_characters_after_trimming_is_spoken(voice, role):
    assert _speak({"text": "  " + "a" * 500 + "  "}).status_code == 200


@pytest.mark.parametrize("raw", [b"not json", b'{"text": "Hi", "speed": NaN}'])
def test_a_body_that_is_not_json_or_has_no_finite_speed_is_422(voice, role, raw):
    response = _app().post(SPEECH, content=raw, headers={"Content-Type": "application/json"})

    assert response.status_code == 422 and response.json()["code"] == "tts_bad_request"


def test_a_body_over_16_kib_is_413(voice, role):
    response = _app().post(SPEECH, content=json.dumps({"text": "a" * 20000}).encode(),
                           headers={"Content-Type": "application/json"})

    assert response.status_code == 413 and response.json()["code"] == "tts_too_long"


@pytest.mark.parametrize("status, upstream, code", [
    (413, {"code": "tts_too_long", "detail": "too long"}, "tts_too_long"),
    (422, {"code": "tts_bad_request", "detail": "voice 'af_nobody' is not one of this server's voices"},
     "tts_bad_request"),
    (429, {"code": "tts_busy", "detail": "busy"}, "tts_busy"),
    (503, {"code": "tts_warming", "detail": "loading"}, "tts_unavailable"),
    (401, {"code": "unauthorized", "detail": "send the token"}, "tts_unavailable"),
    (500, None, "tts_unavailable"),
])
def test_voice_tts_refusals_become_this_servers_codes(voice, role, status, upstream, code):
    voice.answer = lambda request: httpx.Response(status, json=upstream) if upstream else httpx.Response(status)

    response = _speak({"text": "Hello there.", "voice": "af_nobody"})

    assert response.status_code == (503 if code == "tts_unavailable" else status)
    assert response.json()["code"] == code
    if status == 422:
        assert "af_nobody" in response.json()["detail"]


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow"),
                                   httpx.ReadTimeout("slow")])
def test_voice_tts_down_or_too_slow_is_503_tts_unavailable(voice, role, error):
    def fail(request):
        raise error

    voice.answer = fail

    response = _speak({"text": "Hello there."})

    assert response.status_code == 503 and response.json()["code"] == "tts_unavailable"


def test_a_viewer_may_not_have_speech(voice, role):
    role["role"] = "viewer"

    assert _speak({"text": "Hello there."}).status_code == 403
    assert _app().get(VOICES).status_code == 403
    assert voice.requests == []


def test_both_routes_take_the_hybrid_context_and_the_chat_permission():
    from core.auth.hybrid import get_request_context_hybrid
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR

    routes = {route.path: route for route in vs.router.routes}
    assert set(routes) == {SPEECH, VOICES}
    for route in routes.values():
        markers = [getattr(dep.call, PERMISSION_MARKER_ATTR, None) for dep in route.dependant.dependencies]
        assert "agents:execute" in markers, route.path
        nested = [sub.call for dep in route.dependant.dependencies for sub in dep.dependencies]
        assert get_request_context_hybrid in nested, route.path


# --------------------------------------------------------------------------- the rate limit


def test_past_the_workspaces_sentences_a_minute_it_is_429_tts_busy(voice, role, monkeypatch):
    monkeypatch.setattr(config, "VOICE_TTS_RATE_PER_MINUTE", 2)
    redis = _FakeRedis()
    monkeypatch.setattr(vs, "_rate_store", RateLimitStore(redis_factory=lambda: redis))

    statuses = [_speak({"text": "Hello there."}).status_code for _ in range(2)]
    refused = _speak({"text": "Hello there."})

    assert statuses == [200, 200]
    assert refused.status_code == 429 and refused.json()["code"] == "tts_busy"
    assert int(refused.headers["retry-after"]) > 0
    assert len(voice.requests) == 2, "the refused sentence never reached voice-tts"
    assert _speak({"text": "Hello there."}, workspace=OTHER_WS).status_code == 200, "one bucket per workspace"


def test_a_rate_of_zero_is_no_limit(voice, role, monkeypatch):
    monkeypatch.setattr(config, "VOICE_TTS_RATE_PER_MINUTE", 0)
    store = MagicMock()
    monkeypatch.setattr(vs, "_rate_store", store)

    assert _speak({"text": "Hello there."}).status_code == 200
    store.check.assert_not_called()


# --------------------------------------------------------------------------- the voices


def test_the_voices_are_voice_tts_list_cached(voice, role):
    voice.answer = lambda request: httpx.Response(200, json=VOICE_LIST)

    first = _app().get(VOICES)
    second = _app().get(VOICES)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == VOICE_LIST
    assert [str(r.url) for r in voice.requests] == [f"{URL}/voices"], "the second answer came from the copy"


def test_an_expired_copy_or_another_url_is_fetched_again(voice, role, monkeypatch):
    voice.answer = lambda request: httpx.Response(200, json=VOICE_LIST)
    monkeypatch.setattr(speech_client, "VOICES_CACHE_SECONDS", 0)

    _app().get(VOICES)
    _app().get(VOICES)
    monkeypatch.setattr(speech_client, "VOICES_CACHE_SECONDS", 300)
    monkeypatch.setattr(config, "VOICE_TTS_URL", "http://voice-tts.elsewhere:8100")
    _app().get(VOICES)

    assert len(voice.requests) == 3


@pytest.mark.parametrize("answer", [
    lambda request: httpx.Response(200, json={"voices": "nope"}),
    lambda request: httpx.Response(200, content=b"not json"),
    lambda request: httpx.Response(500),
])
def test_a_voice_list_that_is_not_the_contracts_is_503_and_not_cached(voice, role, answer):
    voice.answer = answer

    response = _app().get(VOICES)

    assert response.status_code == 503 and response.json()["code"] == "tts_unavailable"
    assert speech_client._voices is None


def test_the_voices_without_voice_tts_url_are_503_tts_off(voice, role, monkeypatch):
    monkeypatch.setattr(config, "VOICE_TTS_URL", "")

    response = _app().get(VOICES)

    assert response.status_code == 503 and response.json()["code"] == "tts_off"
    assert voice.requests == []


# --------------------------------------------------------------------------- discovery and the manifests


@pytest.mark.parametrize("edition", ["saas", "local"])
def test_discovery_lists_tts_only_while_voice_tts_url_is_set(monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, "VOICE_TTS_URL", "")
    assert "tts" not in discovery.offered_capabilities()

    monkeypatch.setattr(config, "VOICE_TTS_URL", URL)
    assert "tts" in discovery.offered_capabilities()
    assert discovery.CAPABILITY_ROUTES["tts"] == ("POST", SPEECH)
    assert discovery.offered_capabilities() == sorted(discovery.CAPABILITY_ROUTES)


def test_the_router_is_mounted_and_both_routes_are_in_the_route_manifest():
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert RouterSpec("api.voice_speech") in MANIFEST_ROUTERS
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": "POST", "path": SPEECH} in manifest["routes"]
    assert {"method": "GET", "path": VOICES} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


def test_the_settings_are_on_the_config_surface():
    names = set(json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"])

    assert {"VOICE_TTS_URL", "VOICE_TTS_TOKEN", "VOICE_TTS_TIMEOUT_SECONDS", "VOICE_TTS_CONNECT_TIMEOUT_SECONDS",
            "VOICE_TTS_DEFAULT_VOICE", "VOICE_TTS_RATE_PER_MINUTE"} <= names
