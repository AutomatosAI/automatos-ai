"""PRD-203 V·S4 — Retell streaming transport: the streaming contract + auth.

Pure, no vendor / pod / GPU. Proves the property the blocking self-hosted path
never had: first audio is requested BEFORE the full agent stream completes.
Plus the inbound-webhook auth + payload parsing (routing) units.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac

import pytest

from modules.voice.providers.retell import (
    INTERACTION_RESPONSE_REQUIRED,
    extract_agent_text,
    parse_llm_request,
    retell_response_frames,
    verify_webhook_signature,
)


# ---------------------------------------------------------------------------
# THE streaming contract — first audio before the full agent stream completes
# ---------------------------------------------------------------------------


def test_first_audio_before_full_agent_stream():
    """A Retell content frame is emitted while the agent is still generating —
    the exact property the collect-everything-then-speak path lacked."""

    async def scenario():
        events: list[str] = []

        async def agent_stream():
            yield '0:"Hello "'
            events.append("agent_yield_1")
            yield '0:"there"'
            events.append("agent_yield_2")
            yield 'd:{"finishReason":"stop"}'  # non-text terminal line
            events.append("agent_done")

        frames = []
        async for frame in retell_response_frames(7, agent_stream()):
            frames.append(frame)
            events.append(f"frame_{len(frames)}")

        # First audio frame emitted BEFORE the agent stream finished producing.
        assert events.index("frame_1") < events.index("agent_done")
        # It even precedes the agent's 2nd chunk — streaming, not buffering.
        assert events.index("frame_1") < events.index("agent_yield_2")

        content_frames = [f for f in frames if not f["content_complete"]]
        assert [f["content"] for f in content_frames] == ["Hello ", "there"]
        assert frames[0]["response_id"] == 7
        # A terminal complete frame closes the turn; the non-text 'd:' line
        # produced no content frame.
        assert frames[-1] == {"response_id": 7, "content": "", "content_complete": True}

    asyncio.run(scenario())


def test_empty_agent_stream_still_closes_turn():
    async def scenario():
        async def empty():
            if False:  # pragma: no cover
                yield ""

        frames = [f async for f in retell_response_frames(1, empty())]
        assert frames == [{"response_id": 1, "content": "", "content_complete": True}]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "chunk,expected",
    [
        ('0:"hi"', "hi"),
        ('0:"multi word text"', "multi word text"),
        ('2:[{"tool":"x"}]', ""),      # data line → no text
        ('d:{"finishReason":"stop"}', ""),  # finish line → no text
        ("garbage", ""),
        ('0:not-json', ""),
    ],
)
def test_extract_agent_text(chunk, expected):
    assert extract_agent_text(chunk) == expected


# ---------------------------------------------------------------------------
# Webhook auth + routing (parse)
# ---------------------------------------------------------------------------


def test_parse_llm_request_extracts_turn_and_context():
    payload = {
        "interaction_type": "response_required",
        "response_id": 5,
        "transcript": [
            {"role": "agent", "content": "Hi, how can I help?"},
            {"role": "user", "content": "What is my order status?"},
        ],
        "call": {
            "call_id": "call_abc",
            "retell_llm_dynamic_variables": {"workspace_id": "ws-1", "agent_id": "42"},
        },
    }
    req = parse_llm_request(payload)
    assert req.interaction_type == INTERACTION_RESPONSE_REQUIRED
    assert req.response_id == 5
    assert req.user_text == "What is my order status?"  # the LAST user turn
    assert req.workspace_id == "ws-1"
    assert req.agent_id == "42"
    assert req.call_id == "call_abc"


def test_parse_llm_request_tolerates_missing_context():
    req = parse_llm_request({"interaction_type": "reminder_required", "response_id": 0})
    assert req.user_text == ""
    assert req.workspace_id is None
    assert req.agent_id is None
    assert req.call_id is None


def test_verify_webhook_signature_valid_and_failclosed():
    """Retell's REAL scheme (their SDK): v={ts_ms},d=HMAC(body+ts).hexdigest()
    with a ±5-minute window. The PRD-203 plain HMAC-of-body compare could
    never match a genuine Retell webhook — first live events all 401'd."""
    secret = "shhh-secret"
    body = b'{"event":"call_started"}'
    ts = 1_700_000_000_000
    digest = hmac.new(secret.encode(), body + str(ts).encode(), hashlib.sha256).hexdigest()
    good = f"v={ts},d={digest}"

    assert verify_webhook_signature(secret, good, body, now_ms=ts) is True
    # within the freshness window (either direction)
    assert verify_webhook_signature(secret, good, body, now_ms=ts + 4 * 60 * 1000) is True
    assert verify_webhook_signature(secret, good, body, now_ms=ts - 4 * 60 * 1000) is True
    # Fail-closed conditions:
    assert verify_webhook_signature(secret, good, body, now_ms=ts + 6 * 60 * 1000) is False  # replay
    assert verify_webhook_signature(secret, digest, body, now_ms=ts) is False  # bare digest, old shape
    assert verify_webhook_signature(secret, f"v={ts},d=deadbeef", body, now_ms=ts) is False
    assert verify_webhook_signature(secret, None, body, now_ms=ts) is False
    assert verify_webhook_signature("", good, body, now_ms=ts) is False
    assert verify_webhook_signature(secret, good, b"tampered", now_ms=ts) is False
