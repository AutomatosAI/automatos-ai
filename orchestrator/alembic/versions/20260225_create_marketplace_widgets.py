"""Create marketplace_widgets table

Revision ID: 20260225_mkt_widgets
Revises: 20260225_sdk_api_keys
Create Date: 2026-02-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260225_mkt_widgets"
down_revision = "20260225_sdk_api_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "marketplace_widgets",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("name", sa.VARCHAR(100), nullable=False, unique=True),
        sa.Column("display_name", sa.VARCHAR(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("long_description", sa.Text, nullable=True),
        sa.Column(
            "developer_id",
            sa.Integer,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("developer_name", sa.VARCHAR(200), nullable=True),
        sa.Column("version", sa.VARCHAR(20), nullable=True),
        sa.Column("changelog", sa.Text, nullable=True),
        sa.Column(
            "pricing_type",
            sa.VARCHAR(20),
            server_default="free",
            nullable=True,
        ),
        sa.Column("price_cents", sa.Integer, nullable=True),
        sa.Column(
            "currency",
            sa.VARCHAR(3),
            server_default="USD",
            nullable=True,
        ),
        sa.Column("icon_url", sa.Text, nullable=True),
        sa.Column(
            "screenshots",
            postgresql.JSONB,
            server_default="[]",
            nullable=True,
        ),
        sa.Column("readme", sa.Text, nullable=True),
        sa.Column("keywords", postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("categories", postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("bundle_url", sa.Text, nullable=True),
        sa.Column("bundle_size", sa.Integer, nullable=True),
        sa.Column("permissions", postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("min_plan", sa.VARCHAR(50), nullable=True),
        sa.Column(
            "install_count",
            sa.Integer,
            server_default="0",
            nullable=True,
        ),
        sa.Column(
            "rating_average",
            sa.Numeric(3, 2),
            server_default="0",
            nullable=True,
        ),
        sa.Column(
            "rating_count",
            sa.Integer,
            server_default="0",
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.VARCHAR(20),
            server_default="draft",
            nullable=True,
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "pricing_type IN ('free', 'one_time', 'subscription')",
            name="ck_mkt_widgets_pricing_type",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'review', 'published', 'suspended')",
            name="ck_mkt_widgets_status",
        ),
    )

    op.create_index(
        "ix_mkt_widgets_developer_id",
        "marketplace_widgets",
        ["developer_id"],
    )
    op.create_index(
        "ix_mkt_widgets_status_published",
        "marketplace_widgets",
        ["status"],
        postgresql_where=sa.text("status = 'published'"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_mkt_widgets_status_published", table_name="marketplace_widgets"
    )
    op.drop_index(
        "ix_mkt_widgets_developer_id", table_name="marketplace_widgets"
    )
    op.drop_table("marketplace_widgets")
