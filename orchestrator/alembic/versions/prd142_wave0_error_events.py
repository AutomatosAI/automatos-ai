"""PRD-142 Wave 0 US-001: error_events queryable sink

Append-only table backing the dashboard's "error rate by subsystem" tile.
Mirrors the PRD-008-A widget_event_log pattern: single table, JSONB
payload, two indexes that match the dashboard rollup queries.

Online-safe: creates a brand-new table only. No backfill, no NOT NULL
added to an existing large table, no data migration.

Standalone migration (down_revision = None) — the orchestrator alembic
config has many heads and this matches the established convention for
add-a-table changes (see add_job_title_to_agents.py).

Revision ID: prd142_wave0_error_events
Create Date: 2026-05-29
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID


revision = "prd142_wave0_error_events"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "error_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("subsystem", sa.String(64), nullable=False),
        sa.Column("operation", sa.String(128), nullable=False),
        sa.Column("error_type", sa.String(128), nullable=True),
        sa.Column("error_message", sa.String(500), nullable=True),
        sa.Column("workspace_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("agent_id", sa.Integer, nullable=True),
        sa.Column("action_name", sa.String(128), nullable=True),
        sa.Column(
            "event_data",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_error_events_subsystem_created",
        "error_events",
        ["subsystem", "created_at"],
    )
    op.create_index(
        "idx_error_events_workspace_created",
        "error_events",
        ["workspace_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_error_events_workspace_created", table_name="error_events")
    op.drop_index("idx_error_events_subsystem_created", table_name="error_events")
    op.drop_table("error_events")
