"""
PRD-207 S3: Voice call lifecycle + metering
============================================

One row per live voice CALL — the per-call grain ``voice_turns`` (per-turn
telemetry, PRD-203 V·S6) deliberately does not carry. This table is:

* the **mint registry** — the row is BORN at ``POST /api/voice/web-call``
  (status ``minted``) so the ``call_id → workspace/user/chat`` mapping exists
  BEFORE any Retell event arrives. S2's webhook trust boundary, S3's
  attribution and S9's BYOK secret-resolution all key off it;
* the **minute meter** — Retell lifecycle events idempotently stamp
  ``started_at/ended_at/duration_seconds`` and the monthly rollup powers the
  S4 cap gate and the S7 settings meter;
* the **thread registry for a call** — ``chat_id`` is the mint-proven bound
  thread; ``fallback_chat_id`` remembers the per-call chat the webhook
  creates when binding is absent or fails validation, so a multi-turn call
  keeps ONE thread (the merged-unarmed webhook created a new chat every
  TURN — ``get_chat("retell:{call_id}")`` can never parse as a UUID).

No FKs by design (the ``voice_turns`` discipline): a deleted chat/user must
not cascade-drop the billing/lifecycle history; the S2 trust check re-reads
``chats`` live anyway. ``workspace_id`` is nullable ONLY for orphan rows —
lifecycle events for a ``call_id`` we never minted are stored loud, never
silently dropped.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

# Lifecycle states. minted → started → ended is the happy path; ``failed``
# marks a call that ended without ever starting (dial/connect failure).
VOICE_CALL_STATUSES = ("minted", "started", "ended", "failed")


class VoiceCall(Base):
    __tablename__ = "voice_calls"
    __table_args__ = (
        Index("idx_voice_calls_workspace_minted", "workspace_id", "minted_at"),
        Index("uq_voice_calls_call_id", "call_id", unique=True),
        {"extend_existing": True},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Retell's call id — the join key for webhooks and voice_turns linkage.
    call_id = Column(String(128), nullable=False)
    provider = Column(String(20), nullable=False, server_default="retell")

    # NULL only on orphan rows (an event for a call_id we never minted).
    workspace_id = Column(PGUUID(as_uuid=True), nullable=True)
    # INTEGER users.id (the #513 discipline — never a Clerk string).
    user_id = Column(Integer, nullable=True)
    # The mint-proven bound thread (str(chats.id)). The S2 webhook writes into
    # this chat ONLY after cross-validating the dynamic vars against this row.
    chat_id = Column(String(64), nullable=True)
    # The per-call chat the webhook created when binding was absent/refused —
    # remembered here so every later turn of the call reuses it.
    fallback_chat_id = Column(String(64), nullable=True)

    status = Column(String(16), nullable=False, server_default="minted")

    minted_at = Column(DateTime, nullable=False, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    # Stamped from Retell's end event timestamps; the meter's unit of truth.
    duration_seconds = Column(Integer, nullable=True)
    disconnect_reason = Column(String(64), nullable=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
