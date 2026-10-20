"""blog_posts review columns — who took editorial responsibility, when, and for which text (PRD-264 EUAI-08b, 20 Oct 2026).

Article 50(4) ¶2 of the AI Act: AI-written text published to inform the public is
disclosed as AI-generated unless a person reviewed it and holds editorial
responsibility. A post an agent wrote (``author_agent_id`` set) is labelled
``ai_generated`` until a workspace member marks it reviewed, which records
``reviewed_by_user_id``, ``reviewed_at`` and ``reviewed_content_sha256`` (the
SHA-256 of the text they reviewed). Editing the post clears all three, and the
review counts only while the post's text still hashes to the reviewed value: the
text lives in a workspace file an agent's file tools can rewrite.

Three nullable columns with no default: every existing post keeps NULL, so an agent's
post stays labelled until someone reviews it. ``IF NOT EXISTS`` keeps the upgrade a
no-op on a database that ``create_all`` built from the updated model. The names
match ``core.models.blog_post_review.BlogPostReviewColumns``.

Revision ID: prd264_blog_review
Revises: prd264_content_provenance
Create Date: 2026-10-20
"""
from alembic import op

revision = "prd264_blog_review"
down_revision = "prd264_content_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS reviewed_by_user_id VARCHAR(255);")
    op.execute("ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;")
    op.execute("ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS reviewed_content_sha256 VARCHAR(64);")


def downgrade() -> None:
    op.execute("ALTER TABLE blog_posts DROP COLUMN IF EXISTS reviewed_content_sha256;")
    op.execute("ALTER TABLE blog_posts DROP COLUMN IF EXISTS reviewed_at;")
    op.execute("ALTER TABLE blog_posts DROP COLUMN IF EXISTS reviewed_by_user_id;")
