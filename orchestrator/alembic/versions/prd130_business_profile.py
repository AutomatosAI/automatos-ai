"""PRD-130: business_profiles table for the Business Intake Wizard

Revision ID: prd130_business_profile
Revises:
Create Date: 2026-04-11

Standalone migration — safe to run anytime. New table only, no existing
tables touched.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "prd130_business_profile"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "business_profiles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("domain", sa.Text(), nullable=False),
        sa.Column("archetype", sa.Text(), nullable=True),
        sa.Column("company_name", sa.Text(), nullable=True),
        sa.Column("sectors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("brands", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("standards", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("voice_notes", sa.Text(), nullable=True),
        sa.Column("goals", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("raw_map_urls", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("selected_urls", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("quality_findings", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("draft_plan", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="started"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_business_profiles_workspace_status",
        "business_profiles",
        ["workspace_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_business_profiles_workspace_status", table_name="business_profiles")
    op.drop_table("business_profiles")
