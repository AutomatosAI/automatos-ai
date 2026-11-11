"""A product's end user in a workspace: the person a server key remembers things about (PRD-274 Wave 3).

§8-17: an end user is ``(workspace, end_user_id)``, the id the product's server key asserts, and
their facts live in ``mem:{ws}:end_user:{id}`` inside the workspace's tenant boundary. This row is
the product's consent record for them (the product is responsible for consent, §8-6) and how long
their memories are kept. Consent and retention hang off the key that recorded them until Wave 5's
product record. ``api/end_user_memory.py`` reads and writes it; erasing the end user deletes it.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

END_USER_ID_MAX = 128


class EndUser(Base):
    """One end user of a workspace, with the consent a product recorded for them."""

    __tablename__ = "end_users"
    __table_args__ = (Index("uq_end_users_workspace_end_user", "workspace_id", "end_user_id", unique=True),)

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    end_user_id = Column(String(END_USER_ID_MAX), nullable=False)
    api_key_id = Column(PGUUID(as_uuid=True), ForeignKey("sdk_api_keys.id", ondelete="SET NULL"), nullable=True)
    consent_recorded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    retention_days = Column(Integer, nullable=True)  # None: kept until erased
    # Set when an erase starts, before anything is deleted: from then on their consent no longer counts
    # (no new turn, store or write), and an erase that failed part-way is finished by the next one.
    erasing_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


def consented(db: Any, workspace_id: Any, end_user_id: str) -> bool:
    """The end user's consent is on record in the workspace, and no erase of them has started."""
    return db.query(EndUser.id).filter(EndUser.workspace_id == workspace_id, EndUser.end_user_id == end_user_id,
                                       EndUser.erasing_at.is_(None)).first() is not None
