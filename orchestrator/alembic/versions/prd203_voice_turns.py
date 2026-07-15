"""PRD-203 V·S6: voice_turns telemetry table

Append-only per-turn record for the voice chat path — decomposed STT/TTS/total
latency, content lengths, truncation + audio-delivered flags. Makes voice
measurable (p50/p95, truncation-rate) where it wrote no telemetry before.

Revision ID: prd203_voice_turns
Revises: prd196_audit_logs_ws_created_idx
Create Date: 2026-07-14
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID

revision = "prd203_voice_turns"
down_revision = "prd196_audit_logs_ws_created_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "voice_turns",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=True),
        sa.Column("message_id", sa.String(64), nullable=True),
        sa.Column("stt_latency_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tts_latency_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("transcript_len", sa.Integer, nullable=False, server_default="0"),
        sa.Column("response_len", sa.Integer, nullable=False, server_default="0"),
        sa.Column("truncated", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("audio_delivered", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_voice_turns_workspace_created",
        "voice_turns",
        ["workspace_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_voice_turns_workspace_created", table_name="voice_turns")
    op.drop_table("voice_turns")
