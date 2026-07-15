"""
PRD-203 V·S6: Voice turn telemetry writer
==========================================

Fire-and-forget persistence of one ``voice_turns`` row per voice chat turn,
built from the latency fields the path ALREADY logs. A persist failure is
logged and swallowed — telemetry must never fail a voice turn (mirrors the
``modules/tools/execution/telemetry.py`` and PRD-185 tracer posture).

Split into a pure ``voice_turn_fields`` (trivially unit-testable, no DB) and a
guarded ``record_voice_turn`` (opens its own short-lived session so it never
disturbs the request transaction).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def voice_turn_fields(
    *,
    stt_latency_ms: float,
    tts_latency_ms: float,
    total_ms: float,
    transcript: str,
    response_text: str,
    truncated: bool,
    audio_delivered: bool,
) -> Dict[str, Any]:
    """Pure: turn the raw per-turn measurements into typed ``voice_turns`` columns.

    Latencies are rounded to whole milliseconds; only text *lengths* are kept
    (never the transcript/response content — privacy). No I/O.
    """
    return {
        "stt_latency_ms": int(round(stt_latency_ms or 0)),
        "tts_latency_ms": int(round(tts_latency_ms or 0)),
        "total_ms": int(round(total_ms or 0)),
        "transcript_len": len(transcript or ""),
        "response_len": len(response_text or ""),
        "truncated": bool(truncated),
        "audio_delivered": bool(audio_delivered),
    }


def record_voice_turn(
    *,
    workspace_id: Any,
    conversation_id: Optional[str],
    message_id: Optional[str],
    stt_latency_ms: float,
    tts_latency_ms: float,
    total_ms: float,
    transcript: str,
    response_text: str,
    truncated: bool,
    audio_delivered: bool,
) -> None:
    """Persist one voice turn. Never raises — a telemetry fault is not fatal.

    Opens its own ``get_db_session`` (which commits on clean exit) so it stays
    off the request transaction.
    """
    try:
        from core.database.database import get_db_session
        from core.models.voice_turns import VoiceTurn

        fields = voice_turn_fields(
            stt_latency_ms=stt_latency_ms,
            tts_latency_ms=tts_latency_ms,
            total_ms=total_ms,
            transcript=transcript,
            response_text=response_text,
            truncated=truncated,
            audio_delivered=audio_delivered,
        )
        with get_db_session() as db:
            db.add(
                VoiceTurn(
                    workspace_id=workspace_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    **fields,
                )
            )
    except Exception:  # noqa: BLE001 — fire-and-forget; never fail the voice turn
        logger.debug("voice_turn telemetry persist failed", exc_info=True)
