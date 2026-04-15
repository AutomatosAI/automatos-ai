"""Add content_hash column to skills table for runtime freshness checks.

Builtin-core skills (like platform-management) are loaded from disk at
deploy time but used from the DB at runtime.  The content_hash column
stores a SHA-256 of the on-disk SKILL.md so the runtime can detect
stale DB rows and refresh them inline (~5 ms) instead of requiring a
full restart.

Revision ID: add_content_hash_to_skills
Create Date: 2026-04-15
"""
from alembic import op
import sqlalchemy as sa

revision = "add_content_hash_to_skills"
down_revision = None  # standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "skills",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("skills", "content_hash")
