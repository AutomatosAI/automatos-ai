"""PRD-251C Socials plans that run themselves — the ONE migration for Wave 4 (results and voice).

* US-C401 / C7: ``social_post_stats``, a published target's numbers at a reading (1 or 7
  days after it went out): its workspace, post and target, the reading, when it was read,
  the numbers the platform gave (JSON) and the action that read them. One row per target
  and reading (``uq_social_post_stats_target_reading``), indexed by post and by workspace
  and time. Deleting a post or a target deletes its rows.
* US-C401 / C8: ``social_voice_examples``, a post's copy as Auto drafted it and as a person
  approved it: its workspace, the post (SET NULL when the post goes), the two texts and
  when, indexed by workspace and time.

``core/models/socials.py`` declares the same shape, and ``tests/test_prd251cw4_migration.py``
holds the two together.

Create_all-first safe (the 89d89c250 rule): each table is created only when missing and
every index IF NOT EXISTS, so running it twice changes nothing.

Chains single-parent on prd251c_wave2 (Wave 3 needed no migration).

Revision ID: prd251c_wave4
Revises: prd251c_wave2
Create Date: 2026-10-04
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "prd251c_wave4"
down_revision = "prd251c_wave2"
branch_labels = None
depends_on = None

STATS, EXAMPLES = "social_post_stats", "social_voice_examples"
STATS_UNIQUE = "uq_social_post_stats_target_reading"
STATS_POST_INDEX = "ix_social_post_stats_post_id"
STATS_WORKSPACE_INDEX = "ix_social_post_stats_workspace_read"
EXAMPLES_INDEX = "ix_social_voice_examples_workspace_created"


def _json() -> sa.types.TypeEngine:
    return sa.JSON().with_variant(JSONB(), "postgresql")


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _uuid_key(column: str, table: str, ondelete: str, nullable: bool = False) -> sa.Column:
    return sa.Column(column, sa.Uuid(as_uuid=True), sa.ForeignKey(f"{table}.id", ondelete=ondelete), nullable=nullable)


def create_stats() -> None:
    """The results' table, when missing (create_all may have built it), and its indexes."""
    if not _has_table(STATS):
        op.create_table(
            STATS,
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            _uuid_key("workspace_id", "workspaces", "CASCADE"),
            _uuid_key("post_id", "social_posts", "CASCADE"),
            _uuid_key("target_id", "social_post_targets", "CASCADE"),
            sa.Column("reading", sa.Integer(), nullable=False),
            sa.Column("read_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("numbers", _json(), nullable=False),
            sa.Column("source_action", sa.String(128), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("target_id", "reading", name=STATS_UNIQUE),
        )
    op.create_index(STATS_POST_INDEX, STATS, ["post_id"], if_not_exists=True)
    op.create_index(STATS_WORKSPACE_INDEX, STATS, ["workspace_id", "read_at"], if_not_exists=True)


def create_examples() -> None:
    """The voice examples' table, when missing, and its index."""
    if not _has_table(EXAMPLES):
        op.create_table(
            EXAMPLES,
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            _uuid_key("workspace_id", "workspaces", "CASCADE"),
            _uuid_key("post_id", "social_posts", "SET NULL", nullable=True),
            sa.Column("draft", sa.Text(), nullable=False),
            sa.Column("approved", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
    op.create_index(EXAMPLES_INDEX, EXAMPLES, ["workspace_id", "created_at"], if_not_exists=True)


def upgrade() -> None:
    create_stats()
    create_examples()


def downgrade() -> None:
    if _has_table(EXAMPLES):
        op.drop_index(EXAMPLES_INDEX, table_name=EXAMPLES, if_exists=True)
        op.drop_table(EXAMPLES)
    if _has_table(STATS):
        op.drop_index(STATS_WORKSPACE_INDEX, table_name=STATS, if_exists=True)
        op.drop_index(STATS_POST_INDEX, table_name=STATS, if_exists=True)
        op.drop_table(STATS)
