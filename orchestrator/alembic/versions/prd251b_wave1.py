"""PRD-251B Socials Studio — the ONE migration for Wave 1 (the Studio).

* US-B101 / B11: ``social_posts.planned_for`` (timestamptz, nullable), the slot a
  post is planned for before approval, with the index
  ``ix_social_posts_workspace_planned_for`` (workspace_id, planned_for). Not content:
  outside the hash, so moving a slot never voids an approval.
* US-B101 / B5: ``social_posts.length_seconds`` (integer, nullable), the video length
  the editor chose, with the CHECK ``ck_social_posts_length_seconds`` (NULL or > 0).
  Content: the hash covers it once set.
* US-B109: ``ck_social_posts_format`` gains ``'text'`` (the editor's Text only post:
  copy alone, no template, no media).

``core/models/socials.py`` declares the same shape, and
``tests/test_prd251bw1_migration.py`` holds the two together.

Create_all-first safe (the 89d89c250 rule): a backend that loaded the new model
before this ran has already built both columns, the index and both CHECKs. The
upgrade adds a column only when ``get_columns`` lacks it, creates the index IF NOT
EXISTS, adds the length CHECK only when absent by name, and replaces the format
CHECK by name (drop if present, add the widened one), so running it twice changes
nothing. Constraint changes go through ``batch_alter_table``: plain ALTERs on
Postgres; on SQLite (the unit tests), a copy of the table.

A later Wave 1 story that truly needs schema extends THIS revision, so the wave
stays one migration.

The downgrade restores the old format CHECK (a ``'text'`` post's format becomes
NULL first), drops the length CHECK, the index and both columns.

Chains single-parent on prd252_ticket_numbers, the merge revision #859 made for
the two heads #852 left.

Revision ID: prd251b_wave1
Revises: prd252_ticket_numbers
Create Date: 2026-10-02
"""
from __future__ import annotations

from typing import List

import sqlalchemy as sa
from alembic import op

revision = "prd251b_wave1"
down_revision = "prd252_ticket_numbers"
branch_labels = None
depends_on = None

TABLE = "social_posts"
POST_FORMAT_CHECK = "format IN ('video', 'image', 'carousel', 'fact_card', 'infographic', 'text')"
POST_FORMAT_CHECK_BEFORE = "format IN ('video', 'image', 'carousel', 'fact_card', 'infographic')"
POST_FORMAT_CHECK_NAME = "ck_social_posts_format"
POST_LENGTH_CHECK = "length_seconds IS NULL OR length_seconds > 0"
POST_LENGTH_CHECK_NAME = "ck_social_posts_length_seconds"
PLANNED_INDEX = "ix_social_posts_workspace_planned_for"
PLANNED_COLUMN = "planned_for"
LENGTH_COLUMN = "length_seconds"
# Batch mode (SQLite copies the table) reflects social_posts alone, as prd251_wave2 does.
POSTS_BATCH_REFLECT = {"resolve_fks": False}


def _has_table() -> bool:
    return sa.inspect(op.get_bind()).has_table(TABLE)


def _columns() -> List[str]:
    if not _has_table():
        return []
    return [column["name"] for column in sa.inspect(op.get_bind()).get_columns(TABLE)]


def _check_names() -> List[str]:
    if not _has_table():
        return []
    return [
        check["name"]
        for check in sa.inspect(op.get_bind()).get_check_constraints(TABLE)
        if check.get("name")
    ]


def add_post_columns() -> None:
    """Both columns and the index, each only when missing. A schema without
    ``social_posts`` (a partial test schema) has nothing to add them to."""
    columns = _columns()
    if not columns:
        return
    if PLANNED_COLUMN not in columns:
        op.add_column(TABLE, sa.Column(PLANNED_COLUMN, sa.DateTime(timezone=True), nullable=True))
    if LENGTH_COLUMN not in columns:
        op.add_column(TABLE, sa.Column(LENGTH_COLUMN, sa.Integer(), nullable=True))
    op.create_index(PLANNED_INDEX, TABLE, ["workspace_id", PLANNED_COLUMN], if_not_exists=True)


def replace_checks(format_check: str, *, with_length_check: bool) -> None:
    """The format CHECK by name (drop if present, add ``format_check``), and the
    length CHECK: added when ``with_length_check`` and absent, dropped otherwise."""
    if not _has_table():
        return
    names = _check_names()
    with op.batch_alter_table(TABLE, reflect_kwargs=POSTS_BATCH_REFLECT) as batch:
        if POST_FORMAT_CHECK_NAME in names:
            batch.drop_constraint(POST_FORMAT_CHECK_NAME, type_="check")
        batch.create_check_constraint(POST_FORMAT_CHECK_NAME, format_check)
        if with_length_check and POST_LENGTH_CHECK_NAME not in names:
            batch.create_check_constraint(POST_LENGTH_CHECK_NAME, POST_LENGTH_CHECK)
        if not with_length_check and POST_LENGTH_CHECK_NAME in names:
            batch.drop_constraint(POST_LENGTH_CHECK_NAME, type_="check")


def drop_post_columns() -> None:
    columns = _columns()
    if not columns:
        return
    op.drop_index(PLANNED_INDEX, table_name=TABLE, if_exists=True)
    with op.batch_alter_table(TABLE, reflect_kwargs=POSTS_BATCH_REFLECT) as batch:
        for name in (LENGTH_COLUMN, PLANNED_COLUMN):
            if name in columns:
                batch.drop_column(name)


def upgrade() -> None:
    add_post_columns()
    replace_checks(POST_FORMAT_CHECK, with_length_check=True)


def downgrade() -> None:
    if not _has_table():
        return
    # A text-only post has no format under the old CHECK.
    op.execute(sa.text(f"UPDATE {TABLE} SET format = NULL WHERE format = 'text'"))
    replace_checks(POST_FORMAT_CHECK_BEFORE, with_length_check=False)
    drop_post_columns()
