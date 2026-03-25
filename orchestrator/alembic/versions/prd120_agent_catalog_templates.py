"""PRD-120: Create agent_catalog_templates table

Stores pre-built agent templates for the skills marketplace.
Each row represents a deployable agent template linked to a SKILL.md file.

Revision ID: prd120_agent_catalog_templates
Revises: prd82c_parallel_schema
Create Date: 2026-03-24
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "prd120_agent_catalog_templates"
down_revision = "prd82c_parallel_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_catalog_templates",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("slug", sa.String(150), unique=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("persona", sa.Text, nullable=True),
        sa.Column("skill_slug", sa.String(150), nullable=True),
        sa.Column("recommended_model", sa.String(100), nullable=True),
        sa.Column("recommended_tools", sa.JSON, server_default="[]", nullable=False),
        sa.Column("tags", sa.JSON, server_default="[]", nullable=False),
        sa.Column("icon", sa.String(10), nullable=True),
        sa.Column(
            "tier",
            sa.String(50),
            nullable=False,
            server_default="free",
        ),
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Indexes for common query patterns
    op.create_index(
        "idx_agent_catalog_templates_slug",
        "agent_catalog_templates",
        ["slug"],
        unique=True,
    )
    op.create_index(
        "idx_agent_catalog_templates_category",
        "agent_catalog_templates",
        ["category"],
    )
    op.create_index(
        "idx_agent_catalog_templates_workspace_id",
        "agent_catalog_templates",
        ["workspace_id"],
    )
    op.create_index(
        "idx_agent_catalog_templates_active",
        "agent_catalog_templates",
        ["is_active"],
        postgresql_where=sa.text("is_active = true"),
    )


def downgrade() -> None:
    op.drop_index("idx_agent_catalog_templates_active")
    op.drop_index("idx_agent_catalog_templates_workspace_id")
    op.drop_index("idx_agent_catalog_templates_category")
    op.drop_index("idx_agent_catalog_templates_slug")
    op.drop_table("agent_catalog_templates")
