"""PRD Blog Widget: Create blog_posts table

Stores agent-authored and human-authored blog posts per workspace.
Public API serves published posts to the embeddable blog widget.

Revision ID: prd_blog_posts
Revises: None (standalone — safe to run anytime)
Create Date: 2026-03-23
"""
from alembic import op

revision = "prd_blog_posts"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS blog_posts (
            id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id            UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            author_agent_id         INTEGER REFERENCES agents(id) ON DELETE SET NULL,
            author_name             VARCHAR(255) NOT NULL,

            title                   VARCHAR(500) NOT NULL,
            slug                    VARCHAR(500) NOT NULL,
            excerpt                 VARCHAR(500),
            content                 TEXT NOT NULL,
            cover_image_url         VARCHAR(1000),

            tags                    TEXT[] DEFAULT '{}',
            category                VARCHAR(100),
            status                  VARCHAR(20) NOT NULL DEFAULT 'draft'
                                        CHECK (status IN ('draft', 'scheduled', 'published', 'archived')),

            published_at            TIMESTAMPTZ,
            scheduled_for           TIMESTAMPTZ,

            seo_title               VARCHAR(200),
            seo_description         VARCHAR(300),

            reading_time_minutes    INTEGER NOT NULL DEFAULT 1,
            view_count              INTEGER NOT NULL DEFAULT 0,

            created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT uq_blog_posts_workspace_slug UNIQUE (workspace_id, slug)
        );

        -- Primary listing query: published posts by workspace, newest first
        CREATE INDEX IF NOT EXISTS ix_blog_posts_listing
            ON blog_posts(workspace_id, status, published_at DESC);

        -- Slug lookup
        CREATE INDEX IF NOT EXISTS ix_blog_posts_slug
            ON blog_posts(workspace_id, slug);

        -- Category filtering
        CREATE INDEX IF NOT EXISTS ix_blog_posts_category
            ON blog_posts(workspace_id, category)
            WHERE category IS NOT NULL;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS blog_posts;")
