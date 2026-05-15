"""PRD-008-A.1: backfill empty Site.capabilities from type defaults

Sites created or backfilled before ``Site.effective_capabilities`` landed
may carry ``capabilities = {}``, which makes the dashboard hide
type-derived features (e.g. CartIdlePanel returns null when has_cart is
falsy). The runtime property already heals this, but this migration
also heals stored data so direct DB readers and analytics see the
right state.

Revision ID: prd008a_capabilities_backfill
Revises: prd008a_widget_event_log
Create Date: 2026-05-15
"""
from __future__ import annotations

from alembic import op


revision = "prd008a_capabilities_backfill"
down_revision = "prd008a_widget_event_log"
branch_labels = None
depends_on = None


# Mirror of orchestrator.core.models.sites.derive_default_capabilities.
# Kept inline so the migration is self-contained and replayable.
_SHOPIFY = """
    jsonb_build_object(
        'has_cart', true,
        'has_catalog', true,
        'has_volume_discounts', false,
        'has_customer_records', true,
        'has_working_hours_source', true,
        'supports_theme_override', true
    )
"""

_NON_SHOPIFY = """
    jsonb_build_object(
        'has_cart', false,
        'has_catalog', false,
        'has_volume_discounts', false,
        'has_customer_records', false,
        'has_working_hours_source', false,
        'supports_theme_override', false
    )
"""


def upgrade() -> None:
    # Only touch rows whose stored capabilities are missing or empty so
    # we don't clobber any flag a connector may have already flipped.
    op.execute(
        f"""
        UPDATE sites
           SET capabilities = {_SHOPIFY}
         WHERE type = 'shopify'
           AND (capabilities IS NULL OR capabilities = '{{}}'::jsonb
                OR NOT (capabilities ? 'has_cart'));
        """
    )
    op.execute(
        f"""
        UPDATE sites
           SET capabilities = {_NON_SHOPIFY}
         WHERE type IN ('wix', 'woocommerce', 'custom')
           AND (capabilities IS NULL OR capabilities = '{{}}'::jsonb
                OR NOT (capabilities ? 'has_cart'));
        """
    )


def downgrade() -> None:
    # Backfill is forward-only — no destructive downgrade. Reverting the
    # migration leaves stored capabilities populated, which is harmless.
    pass
