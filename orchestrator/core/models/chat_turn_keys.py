"""A product's server key's chat turn under an ``Idempotency-Key`` (PRD-274 Wave 1, §5.2).

One row per (key, Idempotency-Key): a hash of the request that started the turn, the turn's state
(running, done or failed), the chat it answered in and the id of its saved answer once it has ended.
The row keeps no words of the turn: a repeat reads the answer from the saved message by its id
(``api.chat_turn_keys``). ``chat_id`` and ``answer_message_id`` have no foreign key, so deleting a chat
(an end user's erase, say) is never held up by a row; a repeat whose answer is gone runs as new.

``alembic/versions/chat_turn_keys_20261116.py`` creates the same table and unique index.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

RUNNING = "running"
DONE = "done"
FAILED = "failed"
STATES = (RUNNING, DONE, FAILED)
STATE_LENGTH = 16
KEY_LENGTH = 255  # the longest Idempotency-Key the header check allows (api.agent_execute.IDEMPOTENCY_KEY_MAX)
HASH_LENGTH = 64  # a sha256, in hex
UNIQUE_INDEX = "uq_chat_turn_keys_key"


class ChatTurnKey(Base):
    """One server key's chat turn under one ``Idempotency-Key``: of two racing requests, the unique
    index lets one run."""

    __tablename__ = "chat_turn_keys"
    __table_args__ = (Index(UNIQUE_INDEX, "api_key_id", "idempotency_key", unique=True),)

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    api_key_id = Column(PGUUID(as_uuid=True), ForeignKey("sdk_api_keys.id", ondelete="CASCADE"), nullable=False)
    idempotency_key = Column(String(KEY_LENGTH), nullable=False)
    request_hash = Column(String(HASH_LENGTH), nullable=False)
    state = Column(String(STATE_LENGTH), nullable=False, default=RUNNING, server_default=text(f"'{RUNNING}'"))
    chat_id = Column(PGUUID(as_uuid=True), nullable=True)
    answer_message_id = Column(PGUUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
