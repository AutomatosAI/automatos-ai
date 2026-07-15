"""
PRD-203 V·S6: Voice turn telemetry
===================================

Append-only per-turn record for the voice chat path. Before this, voice wrote
NO voice-specific telemetry — only ephemeral log lines carried
``stt_latency_ms`` / ``tts_latency_ms`` — so "is voice any good?" was
unanswerable from data (``voice_profiles`` present, ``voice_turns`` absent from
the census).

One row per ``POST /api/chat/voice`` turn makes §7's voice metrics real:
p50/p95 latency (decomposed into STT vs TTS vs total), truncation-rate
(→ 0 after V·S5's cap deletion), and whether audio was delivered.

Reuses the platform's established single-table fire-and-forget telemetry
PATTERN (``WidgetEventLog`` PRD-008-A / ``ToolExecutionLog`` PRD-139) — a
purpose-built table, NOT a parallel tracing plane. ``llm_usage`` is
token/cost/billing-shaped (no STT/TTS/truncation columns) and the PRD-185
tracer emits Langfuse spans (not a queryable latency distribution), so neither
fits a voice turn; this table does.

Written fire-and-forget by ``modules/voice/telemetry.py`` — a persist failure
must never fail the voice turn.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


class VoiceTurn(Base):
    __tablename__ = "voice_turns"
    __table_args__ = (
        Index("idx_voice_turns_workspace_created", "workspace_id", "created_at"),
        {"extend_existing": True},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Workspace-scoped — the metric grain the dashboard rolls up by.
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False)

    # Chat + message identifiers (opaque strings; no FK — a turn is telemetry,
    # not a chat-lifecycle row, and a deleted chat should not cascade-drop history).
    conversation_id = Column(String(64), nullable=True)
    message_id = Column(String(64), nullable=True)

    # Decomposed latencies (milliseconds) — the whole point of the table.
    stt_latency_ms = Column(Integer, nullable=False, server_default="0")
    tts_latency_ms = Column(Integer, nullable=False, server_default="0")
    total_ms = Column(Integer, nullable=False, server_default="0")

    # Content shape (lengths only — never the transcript/response text: privacy).
    transcript_len = Column(Integer, nullable=False, server_default="0")
    response_len = Column(Integer, nullable=False, server_default="0")

    # truncation_fired → 0 after V·S5 removed the ~500-char cap; kept so the
    # truncation-rate metric is provably zero rather than absent.
    truncated = Column(Boolean, nullable=False, server_default="false")
    # Whether an audio reply was actually produced + delivered this turn.
    audio_delivered = Column(Boolean, nullable=False, server_default="false")

    created_at = Column(DateTime, nullable=False, server_default=func.now())
