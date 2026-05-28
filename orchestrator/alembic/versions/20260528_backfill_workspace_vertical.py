"""PRD-141 Phase 1 — backfill workspace.settings.vertical for Shopify workspaces.

Every existing Shopify workspace (identified by the presence of
``settings.shopify_domain``) must have ``settings.vertical = 'shopify'``
set BEFORE ``orchestrator/api/widgets/chat.py`` is rewired to dispatch by
vertical (US-010). Without this backfill, dispatch would fall through to
the generic plugin for live INBUILD traffic and break PRD-007 product-page
openers + PRD-008-B cart-idle popups.

Verify count matches expected Shopify workspace count after upgrade:

    SELECT count(*) FROM workspaces
    WHERE settings ? 'shopify_domain'
      AND settings ->> 'vertical' = 'shopify';

should equal::

    SELECT count(*) FROM workspaces WHERE settings ? 'shopify_domain';

Idempotency: the WHERE clause excludes rows already carrying a ``vertical``
key, so a second run is a no-op (UPDATE rowcount = 0).

Downgrade only removes the ``vertical`` key from rows we set it on (Shopify
workspaces where ``vertical = 'shopify'``); manually-set values on other
workspaces are preserved.

Revision ID: prd141_backfill_workspace_vertical
Revises: prd008a4_channel_drivers
Create Date: 2026-05-28
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "prd141_backfill_workspace_vertical"
down_revision = "prd008a4_channel_drivers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE workspaces
               SET settings = jsonb_set(settings, '{vertical}', '"shopify"', true)
             WHERE settings ? 'shopify_domain'
               AND NOT settings ? 'vertical'
            """
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE workspaces
               SET settings = settings - 'vertical'
             WHERE settings ? 'shopify_domain'
               AND settings ->> 'vertical' = 'shopify'
            """
        )
    )
