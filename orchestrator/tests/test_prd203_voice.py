"""PRD-203 V·S6 — voice telemetry (pure/mocked).

One voice_turns row is persisted from the already-logged latencies. The writer
is shared by the Retell live-voice lane (api/voice_retell.py); the old
voice-message lane (api/chat_voice.py) was decommissioned with the
self-hosted voice-service stack.

All boundaries mocked — DB session — so nothing live runs.
"""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4


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
