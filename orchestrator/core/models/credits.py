"""The credit ledger behind hosted Studio's plans and trial (PRD-265 US-002).

1 credit = 1 US cent of AI usage at Automatos prices. Two tables:

- ``credit_ledger`` is append-only: one row per credit movement per bucket. A
  movement comes from one source event (a metered model call, a trial grant, an
  admin action, a Stripe event) named by ``source_type`` + ``source_ref``; the
  unique key on ``(source_type, source_ref, bucket)`` means the same event can
  never land twice.
  A debit that spans buckets (trial, then monthly, then top-up) writes one row per
  bucket it drew from. ``bucket = 'overhead'`` rows carry no credits: they record
  platform cost that is absorbed (PRD-265 D-3), for margin reporting.
- ``credit_balances`` holds one row per workspace, updated in the same
  transaction as every ledger insert (``services/credits/ledger.py``), so the
  pre-call gate reads one row. A balance may go negative (a refund clawback, or
  calls metered after it ran out): spending then pauses.

Billing records key on ``workspace_id``, never on a Clerk user ID (the Clerk
cutover relinks user IDs). The Alembic revision ``prd265_credit_ledger`` declares
the same names.
"""
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID

from core.database.base import Base

LEDGER_KINDS = ("grant", "topup", "debit", "refund", "adjust", "expire")
LEDGER_BUCKETS = ("trial", "monthly", "topup", "overhead")


class CreditLedgerEntry(Base):
    """One credit movement in one bucket. Never updated or deleted."""

    __tablename__ = "credit_ledger"

    id = Column(BigInteger, primary_key=True)
    # CASCADE: erasing a workspace erases its billing history with it.
    workspace_id = Column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False,
    )
    kind = Column(String(16), nullable=False)
    bucket = Column(String(16), nullable=False)
    credits = Column(Integer, nullable=False)  # signed: + adds to the bucket, - takes from it
    expires_at = Column(DateTime(timezone=True), nullable=True)
    source_type = Column(String(32), nullable=False)
    source_ref = Column(String(255), nullable=False)
    actor = Column(String(255), nullable=True)
    reason = Column(Text, nullable=True)
    # A metered call's row: its band (``metered``), the credits charged and the
    # platform's real cost for it, so margin per model and per plan shows.
    band = Column(String(16), nullable=True)
    charged_credits = Column(Integer, nullable=True)
    actual_cost_usd = Column(Numeric(14, 6), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("source_type", "source_ref", "bucket", name="uq_credit_ledger_source"),
        CheckConstraint(
            "kind IN ('grant', 'topup', 'debit', 'refund', 'adjust', 'expire')",
            name="ck_credit_ledger_kind",
        ),
        CheckConstraint(
            "bucket IN ('trial', 'monthly', 'topup', 'overhead')", name="ck_credit_ledger_bucket",
        ),
        Index("ix_credit_ledger_workspace_created", "workspace_id", "created_at"),
        {"extend_existing": True},
    )


class CreditBalance(Base):
    """A workspace's spendable credits, by bucket (the gate's one-row read)."""

    __tablename__ = "credit_balances"

    workspace_id = Column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True,
    )
    trial = Column(Integer, nullable=False, default=0, server_default="0")
    monthly = Column(Integer, nullable=False, default=0, server_default="0")
    topup = Column(Integer, nullable=False, default=0, server_default="0")
    # End of the paid period the monthly credits belong to (set by billing, W2).
    period_end = Column(DateTime(timezone=True), nullable=True)
    # The last balance alert sent (0 none, 1 running low, 2 run out), so each fires
    # once; it drops back when credits are added (services/credits/alerts.py).
    alert_level = Column(SmallInteger, nullable=False, default=0, server_default="0")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = ({"extend_existing": True},)
