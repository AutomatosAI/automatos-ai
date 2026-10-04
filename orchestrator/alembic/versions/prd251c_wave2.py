"""PRD-251C Socials plans that run themselves — the ONE migration for Wave 2 (weekly batches).

* US-C201 / C1: ``social_posts.batch_key``, the batch a weekly or monthly plan made the post
  in ("2026-W42", "2026-11"), nullable and indexed with ``campaign_id``
  (``ix_social_posts_campaign_batch``): the Queue groups a plan's batch by it and "Approve
  the week" approves it. Not content. A plan's rhythm, batch day and batch date live in its
  ``make`` JSON, so they need no column.

``core/models/socials.py`` declares the same shape, and ``tests/test_prd251cw2_migration.py``
holds the two together.

Create_all-first safe (the 89d89c250 rule): the column is added only when ``get_columns``
lacks it and the index IF NOT EXISTS, so running it twice changes nothing.

Chains single-parent on prd251b_wave3.

Revision ID: prd251c_wave2
Revises: prd251b_wave3
Create Date: 2026-10-04
"""
from __future__ import annotations

from typing import List

import sqlalchemy as sa
from alembic import op

revision = "prd251c_wave2"
down_revision = "prd251b_wave3"
branch_labels = None
depends_on = None

POSTS = "social_posts"
BATCH_COLUMN = "batch_key"
BATCH_INDEX = "ix_social_posts_campaign_batch"
# Batch mode (SQLite copies the table) reflects the one table alone, as prd251b_wave2 does.
BATCH_REFLECT = {"resolve_fks": False}


def _columns() -> List[str]:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(POSTS):
        return []
    return [column["name"] for column in inspector.get_columns(POSTS)]


def upgrade() -> None:
    present = _columns()
    if not present:
        return  # a schema without the table has nothing to add to
    if BATCH_COLUMN not in present:
        op.add_column(POSTS, sa.Column(BATCH_COLUMN, sa.String(16), nullable=True))
    op.create_index(BATCH_INDEX, POSTS, ["campaign_id", BATCH_COLUMN], if_not_exists=True)


def downgrade() -> None:
    present = _columns()
    if not present:
        return
    op.drop_index(BATCH_INDEX, table_name=POSTS, if_exists=True)
    if BATCH_COLUMN in present:
        with op.batch_alter_table(POSTS, reflect_kwargs=BATCH_REFLECT) as batch:
            batch.drop_column(BATCH_COLUMN)
