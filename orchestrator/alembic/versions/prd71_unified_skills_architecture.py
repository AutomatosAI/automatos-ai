"""PRD-71: Unified Skills Architecture

- Make skills.workspace_id nullable (NULL = marketplace/global skill)
- Add skills.package_slug for plugin materialization tracking
- Add marketplace_plugins.materialized_skill_ids JSONB column
- Create workspace_enabled_skills junction table

Revision ID: prd71_unified_skills
Revises: c551b7ae7bec
Create Date: 2026-03-04
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision = "prd71_unified_skills"
down_revision = "c551b7ae7bec"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Make skills.workspace_id nullable (NULL = marketplace/global skill)
    op.alter_column(
        "skills",
        "workspace_id",
        existing_type=UUID(as_uuid=True),
        nullable=True,
    )

    # 2. Add package_slug to skills (tracks which plugin package materialized this skill)
    op.add_column(
        "skills",
        sa.Column("package_slug", sa.String(100), nullable=True),
    )

    # 3. Add materialized_skill_ids to marketplace_plugins
    op.add_column(
        "marketplace_plugins",
        sa.Column("materialized_skill_ids", JSONB, server_default="[]", nullable=True),
    )

    # 4. Create workspace_enabled_skills junction table
    op.create_table(
        "workspace_enabled_skills",
        sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("skill_id", sa.Integer(), sa.ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("enabled_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("enabled_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("idx_workspace_enabled_skills_workspace", "workspace_enabled_skills", ["workspace_id"])
    op.create_index("idx_workspace_enabled_skills_skill", "workspace_enabled_skills", ["skill_id"])


def downgrade() -> None:
    op.drop_index("idx_workspace_enabled_skills_skill", table_name="workspace_enabled_skills")
    op.drop_index("idx_workspace_enabled_skills_workspace", table_name="workspace_enabled_skills")
    op.drop_table("workspace_enabled_skills")
    op.drop_column("marketplace_plugins", "materialized_skill_ids")
    op.drop_column("skills", "package_slug")
    op.alter_column(
        "skills",
        "workspace_id",
        existing_type=UUID(as_uuid=True),
        nullable=False,
    )
