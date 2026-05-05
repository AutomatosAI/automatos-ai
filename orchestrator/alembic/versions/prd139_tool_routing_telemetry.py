"""PRD-139: Add tool routing telemetry columns to tool_execution_logs

Adds:
  - intent_cluster_id INTEGER FK NULL (references intent_classification_cache.id)
  - routing_source VARCHAR(20) NULL
  - telemetry_source VARCHAR(20) NULL DEFAULT 'production'

Also makes agent_id nullable (SET NULL on delete) to support non-agent tool calls.

Revision ID: prd139_tool_routing_telemetry
Revises: None (standalone)
Create Date: 2026-05-04
"""
from alembic import op
import sqlalchemy as sa

revision = "prd139_tool_routing_telemetry"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Make agent_id nullable to support non-agent tool calls
    op.alter_column(
        "tool_execution_logs",
        "agent_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    # Drop existing FK constraint and re-add with SET NULL
    # Production DB uses 'fk_tool_execution_agent' (not the SQLAlchemy default name)
    op.drop_constraint(
        "fk_tool_execution_agent",
        "tool_execution_logs",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_tool_execution_agent",
        "tool_execution_logs",
        "agents",
        ["agent_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Add PRD-139 telemetry columns
    op.add_column(
        "tool_execution_logs",
        sa.Column(
            "intent_cluster_id",
            sa.Integer(),
            sa.ForeignKey("intent_classification_cache.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "tool_execution_logs",
        sa.Column("routing_source", sa.String(20), nullable=True),
    )
    op.add_column(
        "tool_execution_logs",
        sa.Column("telemetry_source", sa.String(20), nullable=True, server_default="production"),
    )


def downgrade() -> None:
    op.drop_column("tool_execution_logs", "telemetry_source")
    op.drop_column("tool_execution_logs", "routing_source")
    op.drop_column("tool_execution_logs", "intent_cluster_id")

    # Revert agent_id to non-nullable with CASCADE
    op.drop_constraint(
        "fk_tool_execution_agent",
        "tool_execution_logs",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_tool_execution_agent",
        "tool_execution_logs",
        "agents",
        ["agent_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.alter_column(
        "tool_execution_logs",
        "agent_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
