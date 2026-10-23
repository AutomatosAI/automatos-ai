"""Paid subscriptions and the payment provider's events (PRD-265 US-010).

- ``billing_accounts``: one row per workspace that has ever gone to checkout — its
  provider customer, its current subscription and the paid period whose monthly
  credits were granted (``credited_period_end`` + ``period_credits``, so an upgrade
  mid-period grants only the difference and a renewal grants a fresh month).
  ``provider_account`` records which provider account the customer lives in, so a
  later move to a company's account can tell old customers from new. A downgrade
  waiting for the period end is ``schedule_id`` + ``scheduled_plan`` +
  ``scheduled_at``; ``business_offer_until`` is a platform admin's Business offer.
- ``billing_events``: every verified webhook, stored before it is processed,
  unique per ``(provider, event_id)`` so a redelivery changes nothing. A row that
  granted credits keeps ``payment_ref`` (the payment that paid) and
  ``grant_source`` (the ledger source it granted), which is how a refund or a
  dispute finds the credits to take back.

Billing records key on ``workspace_id``, never on a Clerk user ID. The Alembic
revision ``prd265_billing`` declares the same names.
"""
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database.base import Base

EVENT_STATUSES = ("received", "processed", "ignored", "failed", "needs_review")


class BillingAccount(Base):
    """A workspace's customer and subscription at the payment provider."""

    __tablename__ = "billing_accounts"

    workspace_id = Column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True,
    )
    provider = Column(String(16), nullable=False, default="stripe", server_default="stripe")
    provider_account = Column(String(64), nullable=True)
    customer_id = Column(String(64), nullable=False, unique=True)
    subscription_id = Column(String(64), nullable=True)
    subscription_status = Column(String(24), nullable=True)
    # The plan the subscription pays for (from the last paid invoice).
    plan = Column(String(32), nullable=True)
    current_period_end = Column(DateTime(timezone=True), nullable=True)
    cancel_at_period_end = Column(Boolean, nullable=False, default=False, server_default="false")
    # The paid period whose monthly credits were granted, and how many.
    credited_period_start = Column(DateTime(timezone=True), nullable=True)
    credited_period_end = Column(DateTime(timezone=True), nullable=True)
    period_credits = Column(Integer, nullable=False, default=0, server_default="0")
    # A downgrade waiting for the period end (a Stripe subscription schedule).
    schedule_id = Column(String(64), nullable=True)
    scheduled_plan = Column(String(32), nullable=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    # A platform admin offered Business to this workspace until then (by invite).
    business_offer_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = ({"extend_existing": True},)


class BillingEvent(Base):
    """One verified webhook from the payment provider, and what it did."""

    __tablename__ = "billing_events"

    id = Column(BigInteger, primary_key=True)
    provider = Column(String(16), nullable=False)
    event_id = Column(String(255), nullable=False)
    event_type = Column(String(64), nullable=False)
    livemode = Column(Boolean, nullable=True)
    status = Column(String(16), nullable=False, default="received", server_default="received")
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True)
    payment_ref = Column(String(255), nullable=True)
    grant_source = Column(String(300), nullable=True)
    amount_cents = Column(Integer, nullable=True)
    note = Column(Text, nullable=True)
    attempts = Column(Integer, nullable=False, default=0, server_default="0")
    payload = Column(JSONB, nullable=False)
    received_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("provider", "event_id", name="uq_billing_events_provider_event"),
        CheckConstraint(
            "status IN ('received', 'processed', 'ignored', 'failed', 'needs_review')",
            name="ck_billing_events_status",
        ),
        # A refund finds the grant its payment made.
        Index("ix_billing_events_payment_ref", "payment_ref", postgresql_where=text("payment_ref IS NOT NULL")),
        {"extend_existing": True},
    )
