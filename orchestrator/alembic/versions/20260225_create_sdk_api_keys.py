"""Create sdk_api_keys table

Revision ID: 20260225_sdk_api_keys
Revises: 20260225_ws_shares
Create Date: 2026-02-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260225_sdk_api_keys"
down_revision = "20260225_ws_shares"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    # Table may already exist (created by SQLAlchemy model auto-creation)
    from sqlalchemy import inspect as sa_inspect
    inspector = sa_inspect(conn)
    if "sdk_api_keys" not in inspector.get_table_names():
        op.create_table(
            "sdk_api_keys",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                server_default=sa.text("gen_random_uuid()"),
                primary_key=True,
            ),
            sa.Column(
                "workspace_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("name", sa.VARCHAR(200), nullable=False),
            sa.Column("key_prefix", sa.VARCHAR(8), nullable=False),
            sa.Column("key_hash", sa.VARCHAR(64), nullable=False),
            sa.Column("key_type", sa.VARCHAR(20), nullable=False),
            sa.Column("permissions", postgresql.ARRAY(sa.Text), nullable=True),
            sa.Column("rate_limit_requests", sa.Integer, nullable=True),
            sa.Column("rate_limit_tokens", sa.Integer, nullable=True),
            sa.Column("allowed_domains", postgresql.ARRAY(sa.Text), nullable=True),
            sa.Column("allowed_ips", postgresql.ARRAY(sa.Text), nullable=True),
            sa.Column(
                "is_active",
                sa.Boolean,
                server_default=sa.text("true"),
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "key_type IN ('public', 'server')",
                name="ck_sdk_api_keys_key_type",
            ),
        )

    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS ix_sdk_api_keys_key_hash
        ON sdk_api_keys (key_hash)
    """))
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS ix_sdk_api_keys_workspace_id
        ON sdk_api_keys (workspace_id)
    """))


def downgrade() -> None:
    op.drop_table("sdk_api_keys")
