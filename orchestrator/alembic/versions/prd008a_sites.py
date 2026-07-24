"""PRD-008-A: Sites table + backfill from workspace.settings

Introduces the ``sites`` table — the 1:N child of ``workspaces`` that
owns per-merchant widget settings across all channel types
(Shopify, Wix, WooCommerce, custom embed).

For every existing workspace, backfills a default Site so that PRD-007
proactive engagement keeps working without disruption. The backfill
infers ``type`` from ``workspace.settings.shopify_domain`` if present,
otherwise creates a ``type='custom'`` Site.

Revision ID: prd008a_sites
Revises: 20260326_fix_installs_item_id
Create Date: 2026-05-14
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

revision = "prd008a_sites"
down_revision = "20260326_fix_installs_item_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sites",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("workspace_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("external_id", sa.String(255), nullable=True),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("settings", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("capabilities", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("secrets", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("workspace_id", "type", "external_id", name="uq_sites_workspace_type_external"),
    )
    op.create_index("idx_sites_workspace_id", "sites", ["workspace_id"])
    op.create_index("idx_sites_type_external", "sites", ["type", "external_id"])

    # Backfill: one Site per existing workspace.
    #
    # The capability defaults mirror ``derive_default_capabilities`` in
    # ``core.models.sites``. Keep them in sync; the model is the source
    # of truth at runtime — this is a one-shot snapshot at migration time.
    op.execute(
        """
        INSERT INTO sites (id, workspace_id, type, external_id, display_name, settings, capabilities)
        SELECT
            uuid_generate_v4(),
            w.id,
            CASE
                WHEN w.settings ? 'shopify_domain' THEN 'shopify'
                ELSE 'custom'
            END AS type,
            CASE
                WHEN w.settings ? 'shopify_domain' THEN w.settings->>'shopify_domain'
                ELSE NULL
            END AS external_id,
            COALESCE(
                w.settings->>'shopify_domain',
                w.name,
                'Default site'
            ) AS display_name,
            -- Move public widget config out of workspace.settings.
            -- ``widget_proactive`` is the only PRD-007 public key today.
            jsonb_build_object(
                'widget_proactive',
                COALESCE(w.settings->'widget_proactive', '{}'::jsonb)
            ) AS settings,
            CASE
                WHEN w.settings ? 'shopify_domain' THEN
                    jsonb_build_object(
                        'has_cart',                  true,
                        'has_catalog',               true,
                        'has_volume_discounts',      false,
                        'has_customer_records',      true,
                        'has_working_hours_source',  true,
                        'supports_theme_override',   true
                    )
                ELSE
                    jsonb_build_object(
                        'has_cart',                  false,
                        'has_catalog',               false,
                        'has_volume_discounts',      false,
                        'has_customer_records',      false,
                        'has_working_hours_source',  false,
                        'supports_theme_override',   false
                    )
            END AS capabilities
        FROM workspaces w
        WHERE NOT EXISTS (
            SELECT 1 FROM sites s WHERE s.workspace_id = w.id
        );
        """
    )

    # Move encrypted Shopify access token into sites.secrets — keep the
    # workspace.settings copy readable for ONE release as a read-only
    # fallback (removed in a follow-up migration once all call sites are
    # updated). Defensive: only update Sites that came from a Shopify
    # workspace and have no secrets set yet.
    op.execute(
        """
        UPDATE sites s
           SET secrets = jsonb_build_object(
                'shopify_access_token', w.settings->>'shopify_access_token'
           )
          FROM workspaces w
         WHERE s.workspace_id = w.id
           AND s.type = 'shopify'
           AND s.secrets IS NULL
           AND w.settings ? 'shopify_access_token';
        """
    )


def downgrade() -> None:
    op.drop_index("idx_sites_type_external", table_name="sites")
    op.drop_index("idx_sites_workspace_id", table_name="sites")
    op.drop_table("sites")
