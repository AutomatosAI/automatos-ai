"""Add heartbeat_results and channel_connections tables

Revision ID: 20260215_heartbeat_channels
Revises: 20260214_seed_llm_models
Create Date: 2026-02-15

PRD-55: Autonomous Assistant Platform -- heartbeat scheduling and
multi-channel messaging infrastructure.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260215_heartbeat_channels"
down_revision = "20260214_seed_llm_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create heartbeat_results and channel_connections tables."""

    # ---------------------------------------------------------------
    # heartbeat_results
    # ---------------------------------------------------------------
    op.create_table(
        "heartbeat_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_type", sa.String(length=20), nullable=False),
        sa.Column("source_id", sa.String(length=100), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "findings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "actions_taken",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("tokens_used", sa.Integer(), nullable=True, default=0),
        sa.Column("cost", sa.Float(), nullable=True, default=0.0),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_heartbeat_results_workspace_id",
        "heartbeat_results",
        ["workspace_id"],
    )
    op.create_index(
        "ix_heartbeat_results_source",
        "heartbeat_results",
        ["source_type", "source_id"],
    )

    # ---------------------------------------------------------------
    # channel_connections
    # ---------------------------------------------------------------
    op.create_table(
        "channel_connections",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("platform", sa.String(length=50), nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="disconnected",
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default="{}",
        ),
        sa.Column("default_agent_id", sa.Integer(), nullable=True),
        sa.Column(
            "message_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("last_activity_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_channel_connections_workspace_id",
        "channel_connections",
        ["workspace_id"],
    )


def downgrade() -> None:
    """Drop heartbeat_results and channel_connections tables."""
    op.drop_index(
        "ix_channel_connections_workspace_id",
        table_name="channel_connections",
    )
    op.drop_table("channel_connections")

    op.drop_index(
        "ix_heartbeat_results_source",
        table_name="heartbeat_results",
    )
    op.drop_index(
        "ix_heartbeat_results_workspace_id",
        table_name="heartbeat_results",
    )
    op.drop_table("heartbeat_results")
