"""PRD-258 US-002 — each call stores Retell's own latency.

* the pure subset (``call_events.vendor_latency_subset``): a full block keeps
  p50/p90/num of the five stages, missing stages are left out, junk values
  are dropped, an empty or malformed block is None;
* the signed lifecycle webhook (``api/voice_retell.retell_events_webhook``):
  a ``call_analyzed`` event carrying ``latency`` fills ``vendor_latency``; one
  without makes exactly one get-call (``httpx.MockTransport``, byte-stream
  body); a failing get-call still answers 2xx; a replay writes the same value
  and a later event without latency never overwrites it.

No Retell, no network, no database: the session is a fake holding one row.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

_SECRET = "whsec_" + "test"  # concat: never a literal secret-shaped token
_API_KEY = "key_" + "test"
_CALL_ID = "call_latency1"

# Retell's latency block shape: more stats than we keep, per stage.
_FULL_LATENCY = {
    "e2e": {"p50": 1450.5, "p90": 2100, "p95": 2300, "max": 2600, "min": 900, "num": 12,
            "values": [900, 2600]},
    "asr": {"p50": 210, "p90": 340, "num": 12},
    "llm": {"p50": 980, "p90": 1500, "num": 12},
    "llm_websocket_network_rtt": {"p50": 40, "p90": 75, "num": 30},
    "tts": {"p50": 260, "p90": 410, "num": 12},
    "knowledge_base": {"p50": 5, "p90": 9, "num": 1},
}
_FULL_SUBSET = {
    "e2e": {"p50": 1450.5, "p90": 2100, "num": 12},
    "asr": {"p50": 210, "p90": 340, "num": 12},
    "llm": {"p50": 980, "p90": 1500, "num": 12},
    "llm_websocket_network_rtt": {"p50": 40, "p90": 75, "num": 30},
    "tts": {"p50": 260, "p90": 410, "num": 12},
}


# ---------------------------------------------------------------------------
# The pure subset
# ---------------------------------------------------------------------------

def test_subset_keeps_p50_p90_num_of_the_five_stages():
    from modules.voice.call_events import VENDOR_LATENCY_STAGES, vendor_latency_subset

    assert VENDOR_LATENCY_STAGES == ("e2e", "asr", "llm", "llm_websocket_network_rtt", "tts")
    assert vendor_latency_subset(_FULL_LATENCY) == _FULL_SUBSET


def test_subset_leaves_out_missing_stages_and_stats():
    from modules.voice.call_events import vendor_latency_subset

    out = vendor_latency_subset({"e2e": {"p50": 1200, "num": 3}, "tts": {"min": 100}})
    assert out == {"e2e": {"p50": 1200, "num": 3}}


def test_subset_drops_junk_values():
    from modules.voice.call_events import vendor_latency_subset

    junk = {
        "e2e": {"p50": "1200", "p90": True, "num": None},
        "asr": {"p50": float("nan"), "p90": float("inf"), "num": 4},
        "llm": [1, 2, 3],
        "tts": "fast",
    }
    assert vendor_latency_subset(junk) == {"asr": {"num": 4}}


@pytest.mark.parametrize("block", [None, {}, [], "latency", 7, {"e2e": {}}, {"e2e": {"p50": "x"}}])
def test_subset_of_an_empty_or_malformed_block_is_none(block):
    from modules.voice.call_events import vendor_latency_subset

    assert vendor_latency_subset(block) is None


# ---------------------------------------------------------------------------
# The signed webhook
# ---------------------------------------------------------------------------

class _FakeRequest:
    def __init__(self, body: bytes, signature: str):
        self._body = body
        self.headers = {"x-retell-signature": signature}

    async def body(self):
        return self._body


def _signed(payload: dict) -> _FakeRequest:
    """Retell's scheme, the way test_prd207_voice_live signs events: v={ts},d={hmac(body+ts)}."""
    body = json.dumps(payload).encode()
    ts = str(int(time.time() * 1000))
    digest = hmac.new(_SECRET.encode(), body + ts.encode(), hashlib.sha256).hexdigest()
    return _FakeRequest(body, f"v={ts},d={digest}")


def _analyzed(latency=None) -> dict:
    call = {"call_id": _CALL_ID, "call_status": "ended"}
    if latency is not None:
        call["latency"] = latency
    return {"event": "call_analyzed", "call": call}


@pytest.fixture
def webhook(monkeypatch):
    """The route with a fake session holding one minted row and a mocked Retell.

    ``env.get_call`` decides each get-call answer; ``env.requests`` records them.
    """
    import api.voice_retell as vr
    from modules.voice import retell_api
    from modules.voice.live_settings import RetellCredentials

    row = SimpleNamespace(
        call_id=_CALL_ID, status="ended", vendor_latency=None,
        # call_ended folds the lifecycle columns too (_apply_ended reads them), and charges the
        # call's workspace (PRD-265 D-6; none here, so nothing is charged)
        started_at=None, ended_at=None, duration_seconds=None, disconnect_reason=None, workspace_id=None,
    )
    env = SimpleNamespace(row=row, requests=[], get_call=None, vr=vr)

    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials(_API_KEY, _SECRET, "agent")
    )

    @contextmanager
    def fake_session():
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row
        yield db

    monkeypatch.setattr(vr, "get_db_session", fake_session)

    def handler(request: httpx.Request) -> httpx.Response:
        env.requests.append(request)
        return env.get_call(request)

    real_client = httpx.AsyncClient

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(retell_api.httpx, "AsyncClient", client)
    return env


def _json_response(status: int, data) -> httpx.Response:
    # A byte stream, not content=: a content= response is pre-read.
    return httpx.Response(status, stream=httpx.ByteStream(json.dumps(data).encode()))


def _post(env, payload: dict) -> dict:
    return asyncio.run(env.vr.retell_events_webhook(_signed(payload)))


def test_call_analyzed_with_latency_fills_vendor_latency_without_a_get_call(webhook):
    webhook.get_call = lambda request: pytest.fail("no get-call when the event carries latency")

    assert _post(webhook, _analyzed(_FULL_LATENCY)) == {"ok": True}
    assert webhook.row.vendor_latency == _FULL_SUBSET
    assert webhook.requests == []


def test_call_analyzed_without_latency_makes_exactly_one_get_call(webhook):
    from modules.voice.retell_api import RETELL_GET_CALL_URL

    webhook.get_call = lambda request: _json_response(
        200, {"call_id": _CALL_ID, "latency": _FULL_LATENCY}
    )

    assert _post(webhook, _analyzed()) == {"ok": True}
    assert len(webhook.requests) == 1
    request = webhook.requests[0]
    assert request.method == "GET"
    assert str(request.url) == f"{RETELL_GET_CALL_URL}/{_CALL_ID}"
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    assert webhook.row.vendor_latency == _FULL_SUBSET


@pytest.mark.parametrize("failure", ["http_500", "unreachable", "not_json"])
def test_a_failing_get_call_still_answers_2xx(webhook, caplog, failure):
    def get_call(request):
        if failure == "unreachable":
            raise httpx.ConnectError("refused", request=request)
        if failure == "not_json":
            return httpx.Response(200, stream=httpx.ByteStream(b"<html>oops</html>"))
        return _json_response(500, {"error": "boom"})

    webhook.get_call = get_call
    with caplog.at_level(logging.WARNING, logger="modules.voice.call_events"):
        assert _post(webhook, _analyzed()) == {"ok": True}

    assert len(webhook.requests) == 1
    assert webhook.row.vendor_latency is None
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "voice_live_get_call_failed" in logged and _CALL_ID in logged
    assert _API_KEY not in logged


def test_a_replay_writes_the_same_value_and_nothing_overwrites_it_with_none(webhook):
    webhook.get_call = lambda request: _json_response(500, {"error": "boom"})

    _post(webhook, _analyzed(_FULL_LATENCY))
    first = dict(webhook.row.vendor_latency)
    _post(webhook, _analyzed(_FULL_LATENCY))  # Retell retries: a replay
    assert webhook.row.vendor_latency == first == _FULL_SUBSET

    # A later event with no latency (the get-call fails too) keeps what is stored.
    _post(webhook, _analyzed())
    _post(webhook, {"event": "call_ended", "call": {"call_id": _CALL_ID}})
    assert webhook.row.vendor_latency == _FULL_SUBSET


def test_call_ended_carrying_latency_stores_it(webhook):
    webhook.get_call = lambda request: pytest.fail("call_ended never fetches")
    payload = {"event": "call_ended", "call": {"call_id": _CALL_ID, "latency": _FULL_LATENCY}}

    assert _post(webhook, payload) == {"ok": True}
    assert webhook.row.vendor_latency == _FULL_SUBSET
    assert webhook.requests == []


def test_a_bad_signature_never_reaches_retell(webhook):
    from fastapi import HTTPException

    webhook.get_call = lambda request: pytest.fail("HMAC first: no get-call on a refused event")
    body = json.dumps(_analyzed()).encode()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(webhook.vr.retell_events_webhook(_FakeRequest(body, "v=1,d=deadbeef")))
    assert exc.value.status_code == 401
    assert webhook.requests == []
