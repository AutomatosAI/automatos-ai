"""PRD-203 V·S5 + V·S6 — voice correctness/cost + telemetry (pure/mocked).

V·S5: the full reply reaches TTS (the ~500-char cap is gone) and a turn returns
      exactly one audio delivery path (S3-backed audio_url, no inline base64).
V·S6: one voice_turns row is persisted from the already-logged latencies.

All boundaries mocked — voice client, STT/TTS, audio upload, DB session, and the
agent bridge — so nothing live runs.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


def _run_voice_chat(monkeypatch, *, response_text: str, response_format: str = "both"):
    """Drive api.chat_voice.voice_chat with every boundary mocked.

    Returns (synthesize_mock, parsed_response_body).
    """
    from api import chat_voice as cv
    from modules.voice.client import SynthesisResult, TranscriptionResult

    monkeypatch.setattr(cv._voice_client, "health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        cv._voice_client,
        "transcribe",
        AsyncMock(
            return_value=TranscriptionResult(
                text="hello there", language="en", duration_ms=12.0, audio_duration_ms=None
            )
        ),
    )
    synth = AsyncMock(
        return_value=SynthesisResult(
            audio=b"AUDIOBYTES", format="mp3", duration_ms=34.0, audio_duration_ms=None
        )
    )
    monkeypatch.setattr(cv._voice_client, "synthesize", synth)
    monkeypatch.setattr(cv, "validate_audio", AsyncMock(return_value=b"rawaudio"))
    # Bridge to the agent loop → a text reply, no specific agent (skips profile lookup).
    monkeypatch.setattr(
        cv, "_collect_streaming_response", AsyncMock(return_value=(response_text, None))
    )
    monkeypatch.setattr(cv, "upload_voice_audio", MagicMock(return_value="workspaces/x/voice/y.mp3"))
    # V·S6 writer is exercised in its own test; keep it inert here.
    monkeypatch.setattr(cv, "record_voice_turn", MagicMock())

    # get_user_id is a local import inside voice_chat.
    fake_chat = types.ModuleType("api.chat")
    fake_chat.get_user_id = lambda db: 1
    monkeypatch.setitem(sys.modules, "api.chat", fake_chat)

    audio = SimpleNamespace(filename="a.webm")
    ctx = SimpleNamespace(workspace_id=uuid4())
    resp = asyncio.run(
        cv.voice_chat(
            audio=audio,
            conversation_id="c1",
            agent_id=None,
            response_format=response_format,
            language=None,
            voice=None,
            db=MagicMock(),
            ctx=ctx,
        )
    )
    return synth, json.loads(resp.body)


# ---------------------------------------------------------------------------
# V·S5 — full reply reaches TTS (no ~500-char cap)
# ---------------------------------------------------------------------------


def test_full_reply_reaches_tts(monkeypatch):
    long_reply = "A" * 1200  # well past the old 500-char cap, no sentence break
    synth, _ = _run_voice_chat(monkeypatch, response_text=long_reply)

    synth.assert_awaited_once()
    sent = synth.await_args.kwargs["text"]
    assert sent == long_reply, "TTS did not receive the full reply"
    assert len(sent) == 1200, "reply was truncated before synthesis"


# ---------------------------------------------------------------------------
# V·S5 — one audio delivery path (no inline base64 alongside the S3 url)
# ---------------------------------------------------------------------------


def test_single_audio_delivery_path(monkeypatch):
    _, body = _run_voice_chat(monkeypatch, response_text="a short spoken reply")

    # The redundant inline base64 return is gone — one mechanism only.
    assert "audio_base64" not in body
    # ...and it is the S3-backed url served by GET /api/chat/voice/audio/{id}.
    assert body["audio_url"] == f"/api/chat/voice/audio/{body['message_id']}"


def test_text_only_format_delivers_no_audio(monkeypatch):
    _, body = _run_voice_chat(monkeypatch, response_text="text only", response_format="text")
    assert body["audio_url"] is None
    assert "audio_base64" not in body


# ---------------------------------------------------------------------------
# V·S6 — telemetry compute + persistence
# ---------------------------------------------------------------------------


def test_voice_turn_fields_pure():
    from modules.voice.telemetry import voice_turn_fields

    fields = voice_turn_fields(
        stt_latency_ms=12.4,
        tts_latency_ms=34.6,
        total_ms=101.4,
        transcript="hi",
        response_text="x" * 600,
        truncated=False,
        audio_delivered=True,
    )
    assert fields["stt_latency_ms"] == 12
    assert fields["tts_latency_ms"] == 35
    assert fields["total_ms"] == 101
    assert fields["transcript_len"] == 2
    assert fields["response_len"] == 600
    assert fields["truncated"] is False
    assert fields["audio_delivered"] is True


def test_record_voice_turn_persists_with_latencies(monkeypatch):
    import core.database.database as dbmod
    from modules.voice import telemetry as tel

    added: list = []

    class _FakeCtx:
        def __enter__(self):
            db = MagicMock()
            db.add.side_effect = lambda row: added.append(row)
            return db

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(dbmod, "get_db_session", lambda: _FakeCtx())

    ws = uuid4()
    tel.record_voice_turn(
        workspace_id=str(ws),
        conversation_id="c1",
        message_id="m1",
        stt_latency_ms=12.0,
        tts_latency_ms=34.0,
        total_ms=100.0,
        transcript="hello",
        response_text="world!",
        truncated=False,
        audio_delivered=True,
    )

    assert len(added) == 1, "exactly one voice_turns row must be written"
    row = added[0]
    assert row.stt_latency_ms == 12
    assert row.tts_latency_ms == 34
    assert row.total_ms == 100
    assert row.transcript_len == 5
    assert row.response_len == 6
    assert row.audio_delivered is True
    assert str(row.workspace_id) == str(ws)
    assert row.conversation_id == "c1"


def test_record_voice_turn_never_raises(monkeypatch):
    """A persist failure must be swallowed — telemetry never fails a voice turn."""
    import core.database.database as dbmod
    from modules.voice import telemetry as tel

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(dbmod, "get_db_session", _boom)
    # Must not raise.
    tel.record_voice_turn(
        workspace_id=str(uuid4()),
        conversation_id="c1",
        message_id="m1",
        stt_latency_ms=1.0,
        tts_latency_ms=2.0,
        total_ms=3.0,
        transcript="a",
        response_text="b",
        truncated=False,
        audio_delivered=False,
    )
