"""A product: an outside app that runs its customers' workspaces on Automatos (PRD-274 Wave 5, §5.1, §8-3).

A platform admin creates the record (``api/admin_products.py``). It says who pays (the owner workspace, §8-2),
what each customer's workspace starts with (the agent roster to clone and the scopes of the key it is minted)
and how long an end user's memories are kept when the product records no retention of its own (decision 28).

The product's own key authenticates it to Automatos: only its PBKDF2 hash is stored (``services/products``),
and the plaintext is shown once, when it is minted. Later PRDs of the wave read the record: 5b finds or creates
a customer's workspace with that key, and the workspace carries ``settings.product_id``.

The owner workspace is ``owner_workspace_id``, not ``workspace_id``: a product is the platform's record, not a
row of the owner's workspace, so a move or an export of the owner workspace doesn't take it, and its purge is
refused while it owns one (``services/workspace_purge``; the foreign key is the backstop).
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, text
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
    default_retention_days = Column(Integer, nullable=True)  # None: an end user's memories are kept until erased
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
