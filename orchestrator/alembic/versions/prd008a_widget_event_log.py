"""PRD-008-A Phase 4: widget_event_log table

Append-only event log for the storefront widget. Source of truth for
the dashboard's per-Site telemetry rollups and any downstream
analytics sinks.

Revision ID: prd008a_widget_event_log
Revises: prd008a_sites
Create Date: 2026-05-14
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

revision = "prd008a_widget_event_log"
down_revision = "prd008a_sites"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "widget_event_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("site_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("event_data", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_widget_event_log_site_created",
        "widget_event_log",
        ["site_id", "created_at"],
    )
    op.create_index(
        "idx_widget_event_log_type_created",
        "widget_event_log",
        ["event_type", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_widget_event_log_type_created", table_name="widget_event_log")
    op.drop_index("idx_widget_event_log_site_created", table_name="widget_event_log")
    op.drop_table("widget_event_log")
