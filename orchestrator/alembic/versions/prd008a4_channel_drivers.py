"""PRD-008-A.4: add mode/webhook_url/last_verified/last_error to
channel_connections + backfill from workspace integrations bag.

Revision ID: prd008a4_channel_drivers
Revises: prd008a_capabilities_backfill
Create Date: 2026-05-22 13:00:00.000000

Adds the four columns the per-driver model needs:

- ``mode`` text — 'webhook' | 'polling'. Default 'webhook' because
  every existing row was created in webhook mode (the polling adapter
  has never had its required library installed in prod, per Railway
  logs).
- ``webhook_url`` text — the URL the platform POSTs inbound traffic to.
  Populated by the driver's install_webhook at Connect time.
- ``last_verified`` timestamptz — when verify() last returned ok.
- ``last_error`` text — the most recent driver-reported error string.
  Surfaces to the dashboard so merchants debug without log-trawling.

Backfill
--------
The legacy ``workspace.settings.integrations.{telegram,slack}_bot_token``
bag is the only thing that's been used in prod for years. For any
workspace that has those keys but no matching channel_connections row,
we create the row here so the new code path immediately works without
the merchant having to re-connect.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "prd008a4_channel_drivers"
down_revision = "prd008a_capabilities_backfill"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    op.add_column(
        "channel_connections",
        sa.Column(
            "mode",
            sa.String(20),
            nullable=False,
            server_default=sa.text("'webhook'"),
        ),
    )
    op.add_column(
        "channel_connections",
        sa.Column("webhook_url", sa.Text(), nullable=True),
    )
    op.add_column(
        "channel_connections",
        sa.Column(
            "last_verified",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "channel_connections",
        sa.Column("last_error", sa.Text(), nullable=True),
    )

    # ------------------------------------------------------------------
    # Backfill — copy legacy integrations bag into channel_connections.
    #
    # Selecting workspaces that have either bot_token key in their
    # ``settings.integrations`` JSON and don't yet have a matching
    # channel_connections row for that platform. We use INSERT ...
    # SELECT ... WHERE NOT EXISTS so the migration is idempotent.
    # ------------------------------------------------------------------
    op.execute(
        sa.text(
            """
            INSERT INTO channel_connections (
                id, workspace_id, platform, config, status, mode,
                metadata, message_count, created_at, updated_at
            )
            SELECT
                gen_random_uuid(),
                w.id,
                'telegram',
                jsonb_build_object('bot_token', w.settings -> 'integrations' ->> 'telegram_bot_token'),
                'active',
                'webhook',
                jsonb_build_object(
                    'default_target', w.settings -> 'integrations' ->> 'telegram_default_chat_id',
                    'backfilled_from', 'integrations.telegram_bot_token'
                ),
                0,
                NOW(),
                NOW()
            FROM workspaces w
            WHERE w.settings -> 'integrations' ->> 'telegram_bot_token' IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM channel_connections c
                WHERE c.workspace_id = w.id AND c.platform = 'telegram'
              )
            """
        )
    )
    op.execute(
        sa.text(
            """
            INSERT INTO channel_connections (
                id, workspace_id, platform, config, status, mode,
                metadata, message_count, created_at, updated_at
            )
            SELECT
                gen_random_uuid(),
                w.id,
                'slack',
                jsonb_build_object(
                    'bot_token', w.settings -> 'integrations' ->> 'slack_bot_token',
                    'default_channel', w.settings -> 'integrations' ->> 'slack_default_channel'
                ),
                'active',
                'webhook',
                jsonb_build_object(
                    'default_target', w.settings -> 'integrations' ->> 'slack_default_channel',
                    'backfilled_from', 'integrations.slack_bot_token'
                ),
                0,
                NOW(),
                NOW()
            FROM workspaces w
            WHERE w.settings -> 'integrations' ->> 'slack_bot_token' IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM channel_connections c
                WHERE c.workspace_id = w.id AND c.platform = 'slack'
              )
            """
        )
    )

    # ------------------------------------------------------------------
    # Repair: any existing row whose config is missing a bot_token but
    # whose workspace has it in the integrations bag — copy it across.
    # Covers the user-reported "I connected via the dashboard but the
    # token in the row was missing the bot_id prefix" case where the
    # integrations bag holds the correct full token.
    # ------------------------------------------------------------------
    op.execute(
        sa.text(
            """
            UPDATE channel_connections c
            SET config = jsonb_set(
                    COALESCE(c.config, '{}'::jsonb),
                    '{bot_token}',
                    to_jsonb(w.settings -> 'integrations' ->> 'telegram_bot_token'),
                    true
                ),
                updated_at = NOW()
            FROM workspaces w
            WHERE c.workspace_id = w.id
              AND c.platform = 'telegram'
              AND w.settings -> 'integrations' ->> 'telegram_bot_token' IS NOT NULL
              AND (
                  c.config IS NULL
                  OR c.config ->> 'bot_token' IS NULL
                  OR position(':' in COALESCE(c.config ->> 'bot_token', '')) = 0
              )
            """
        )
    )


def downgrade() -> None:
    op.drop_column("channel_connections", "last_error")
    op.drop_column("channel_connections", "last_verified")
    op.drop_column("channel_connections", "webhook_url")
    op.drop_column("channel_connections", "mode")
    # Backfilled rows are intentionally NOT deleted on downgrade — the
    # merchant likely wants those preserved even if the schema reverts.
