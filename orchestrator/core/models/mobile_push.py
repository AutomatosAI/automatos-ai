"""PRD-266 §4.4, §4.5 — push tickets and digest posts: what the push wave remembers.

* ``MobilePushTicket`` — Expo answers every accepted message with a ticket id; only its
  receipt, read later by id, says whether the phone took it. The row keeps the ticket
  until the receipts job reads it (a day at most): the device it went to (a
  ``DeviceNotRegistered`` receipt turns that device's push off), its workspace and its
  push type. Deleting the device or the workspace deletes its tickets.
* ``MobileDigestPost`` — one row per workspace, person and local date on which the daily
  digest was posted. The primary key is the claim (``INSERT … ON CONFLICT DO NOTHING``),
  so ticks, workers and replicas never post one person two digests on one day.

Why two tables and not an extended one: mobile_devices is one row per device and neither
Expo's tickets (many per device, kept a day) nor a person's daily digest (one per person
per local date, unique across workers and replicas) fits a column there; notifications
has no slot for either. The Alembic revision ``prd266_w2_push`` declares the same names.
"""
from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

# The three push types (PRD-266 D22): the only kinds of message a phone is ever sent.
PUSH_TYPES = ("needs_answer", "review", "auto_message")

TICKET_ID_LENGTH = 64  # Expo's ticket ids are UUIDs (36 characters)
PUSH_TYPE_LENGTH = 16

PUSH_TYPE_CHECK = "push_type IN ({})".format(", ".join(f"'{t}'" for t in PUSH_TYPES))


class MobilePushTicket(Base):
    """One Expo ticket waiting for its receipt: the device, workspace and push type it was for."""

    __tablename__ = "mobile_push_tickets"
    __table_args__ = (
        CheckConstraint(PUSH_TYPE_CHECK, name="ck_mobile_push_tickets_push_type"),
        Index("ix_mobile_push_tickets_created_at", "created_at"),
        {"extend_existing": True},
    )

    ticket_id = Column(String(TICKET_ID_LENGTH), primary_key=True)
    device_id = Column(
        PGUUID(as_uuid=True), ForeignKey("mobile_devices.id", ondelete="CASCADE"), nullable=False,
    )
    workspace_id = Column(
        PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False,
    )
    push_type = Column(String(PUSH_TYPE_LENGTH), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class MobileDigestPost(Base):
    """The claim that a person's daily digest was posted in a workspace on one local date."""

    __tablename__ = "mobile_digest_posts"
    __table_args__ = ({"extend_existing": True},)

    workspace_id = Column(
        PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True,
    )
    # Integer, matching users.id; resolve it with core.auth.principal.resolve_user_pk.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    # The date in the person's phone's timezone (§4.5), not the server's.
    local_date = Column(Date, primary_key=True)
    posted_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
