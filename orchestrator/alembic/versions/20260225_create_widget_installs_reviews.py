"""Create widget_installations and widget_reviews tables

Revision ID: 20260225_mkt_installs_reviews
Revises: 20260225_mkt_widgets
Create Date: 2026-02-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260225_mkt_installs_reviews"
down_revision = "20260225_mkt_widgets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── widget_installations ─────────────────────────────────────────────
    op.create_table(
        "widget_installations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "widget_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("marketplace_widgets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "use_count",
            sa.Integer,
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "installed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("uninstalled_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index(
        "ix_widget_installs_widget_id",
        "widget_installations",
        ["widget_id"],
    )
    op.create_index(
        "ix_widget_installs_workspace_id",
        "widget_installations",
        ["workspace_id"],
    )
    op.create_index(
        "ix_widget_installs_user_id",
        "widget_installations",
        ["user_id"],
    )

    # ── widget_reviews ───────────────────────────────────────────────────
    op.create_table(
        "widget_reviews",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "widget_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("marketplace_widgets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("title", sa.VARCHAR(200), nullable=True),
        sa.Column("body", sa.Text, nullable=True),
        sa.Column(
            "is_verified_purchase",
            sa.Boolean,
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.VARCHAR(20),
            server_default="published",
            nullable=False,
        ),
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
            "rating >= 1 AND rating <= 5",
            name="ck_widget_reviews_rating_range",
        ),
        sa.UniqueConstraint("widget_id", "user_id", name="uq_widget_reviews_widget_user"),
    )


def downgrade() -> None:
    op.drop_table("widget_reviews")
    op.drop_index("ix_widget_installs_user_id", table_name="widget_installations")
    op.drop_index("ix_widget_installs_workspace_id", table_name="widget_installations")
    op.drop_index("ix_widget_installs_widget_id", table_name="widget_installations")
    op.drop_table("widget_installations")
