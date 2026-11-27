"""A product: an outside app that runs its customers' workspaces on Automatos (PRD-274 Wave 5, §5.1, §8-3).

A platform admin creates the record (``api/admin_products.py``). It says who pays (the owner workspace, §8-2),
what each customer's workspace starts with (the agent roster to clone and the scopes of the key it is minted,
and whether finding it again mints another)
and how long an end user's memories are kept when the product records no retention of its own (decision 28).

The product's own key authenticates it to Automatos: only its PBKDF2 hash is stored (``services/products``),
and the plaintext is shown once, when it is minted. With that key the product finds or creates its customers'
workspaces (``api/products.py``, ``services/product_workspaces``), each carrying ``settings.product_id``.

The owner workspace is ``owner_workspace_id``, not ``workspace_id``: a product is the platform's record, not a
row of the owner's workspace, so a move or an export of the owner workspace doesn't take it, and its purge is
refused while it owns one (``services/workspace_purge``; the foreign key is the backstop).
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Boolean, CheckConstraint, Column, Date, DateTime, ForeignKey, Integer, SmallInteger, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

PRODUCT_SLUG_MAX = 64
PRODUCT_NAME_MAX = 255
KEY_TYPES = ("public", "server")


class Product(Base):
    """One product, created by a platform admin."""

    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("key_type IN ('public', 'server')", name="ck_products_key_type"),
        CheckConstraint("default_retention_days IS NULL OR default_retention_days >= 1",
                        name="ck_products_default_retention_days"),
        CheckConstraint("workspace_monthly_credit_cap IS NULL OR workspace_monthly_credit_cap >= 1",
                        name="ck_products_workspace_monthly_credit_cap"),
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(PRODUCT_SLUG_MAX), nullable=False, unique=True)
    name = Column(String(PRODUCT_NAME_MAX), nullable=False)
    owner_workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="RESTRICT"),
                                nullable=False, index=True)
    key_hash = Column(String(128), nullable=False, unique=True)
    key_prefix = Column(String(32), nullable=False)  # the key's first characters, to tell keys apart
    # The roster each customer's workspace starts with: marketplace agents by slug, and the one the others
    # report to (provision_vertical's roster seeding).
    agent_slugs = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    ops_manager_slug = Column(String(100), nullable=True)
    # The key each customer's workspace is minted: its type and scopes (api/api_keys.VALID_PERMISSIONS).
    key_type = Column(String(16), nullable=False, default="server", server_default=text("'server'"))
    key_scopes = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    # Finding a customer's workspace again mints no new key while it has an active one of that type (its copy is
    # the product's to keep); False mints one on every find, as Shopify's re-provision does.
    reuse_existing_key = Column(Boolean, nullable=False, default=True, server_default=text("true"))
    default_retention_days = Column(Integer, nullable=True)  # None: an end user's memories are kept until erased
    # The credits each of its workspaces may spend of the owner's in a calendar month (UTC); a workspace's own
    # ``settings.product_credit_cap`` overrides it. None: no cap, the owner's balance limits (services/credits/caps).
    workspace_monthly_credit_cap = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class ProductWorkspaceSpend(Base):
    """What a product's workspace spent of its owner's credits in one calendar month (UTC), moved in the same
    transaction as each debit (``services/credits/payer.debit``), so the credit gate reads its cap in one row. The
    ledger stays the record (``credit_ledger.sub_tenant_id``); this is its running total. ``notified_level`` is the
    owner's last notice this month: 1 at 80% of the cap, 2 at the cap."""

    __tablename__ = "product_workspace_spend"

    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    month = Column(Date, primary_key=True)
    credits = Column(Integer, nullable=False, default=0, server_default=text("0"))
    notified_level = Column(SmallInteger, nullable=False, default=0, server_default=text("0"))
