"""Blog content → workspace files: add file_path, make content nullable

Blog posts migrate from storing markdown in a DB TEXT column to workspace
files (.md). The file_path column points to the workspace file; the content
column becomes a nullable fallback for pre-migration posts.

Revision ID: blog_content_to_workspace
Revises: prd_blog_posts
Create Date: 2026-03-27
"""
from alembic import op

revision = "blog_content_to_workspace"
down_revision = "prd_blog_posts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE blog_posts
            ADD COLUMN IF NOT EXISTS file_path VARCHAR(500);

        ALTER TABLE blog_posts
            ALTER COLUMN content DROP NOT NULL;
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE blog_posts
            ALTER COLUMN content SET NOT NULL;

        ALTER TABLE blog_posts
            DROP COLUMN IF EXISTS file_path;
    """)
