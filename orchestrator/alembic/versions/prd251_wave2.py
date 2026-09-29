"""PRD-251 Socials — the ONE migration for Wave 2 (the Socials tab).

* S2.4a (US-201, D2): ``social_campaigns``, the table series approval (S2.4,
  US-210) works on: ``id``, ``workspace_id`` (ON DELETE CASCADE), ``name``,
  ``approval_mode`` (``per_post`` | ``series``, CHECK
  ``ck_social_campaigns_approval_mode``), ``approved_hash_set`` (the content
  hashes a series approval approved: a JSON list, JSONB on Postgres),
  ``approved_by``/``approved_at``, ``created_by``, ``created_at``/``updated_at``,
  and the index ``ix_social_campaigns_workspace_created``.
* ``social_posts.campaign_id``, a bare column since Wave 0, gains its foreign key
  to ``social_campaigns.id`` ON DELETE SET NULL (deleting a campaign keeps its
  posts, unlinked) and the index ``ix_social_posts_campaign_id``. No code has
  written ``campaign_id`` before this wave, so every existing row holds NULL and
  the key validates without touching data.

* S2.2b (US-208): ``social_posts.preview``, the composer's last preview render:
  ``{"status", "content_hash", "files", "error", "at"}``. Not content: outside the
  content hash and ``media``, and never a move of the post's status. Nullable
  JSON (JSONB on Postgres), added only when missing.

``core/models/socials.py`` declares the same shape, and
``tests/test_prd251w2_campaigns.py`` holds the two together.

Create_all-first safe (the 89d89c250 rule): a backend that loaded the new models
before this migration ran has already built ``social_campaigns``, and on a fresh
database the foreign key and the post index too. The upgrade keeps a table that
exists and adds only the indexes it lacks, adds the foreign key only when
``get_foreign_keys`` shows it absent, and creates the post index IF NOT EXISTS, so
running it twice changes nothing. The key goes through ``batch_alter_table``: a
plain ALTER on Postgres, and on SQLite (the unit tests), which cannot add a
constraint to a table, a copy of the table.

A later Wave 2 story that truly needs schema extends THIS revision, so the wave
stays one migration, and every step it adds tolerates what ``create_all`` already
built.

The downgrade drops the foreign key, the post index and the table. The posts
stay, with their ``campaign_id`` values and no key behind them, and lose only
their ``preview`` (the preview files stay in storage).

Chains single-parent on prd251w1_merge_heads (the single head of main after
Wave 1 landed through #788).

Revision ID: prd251_wave2
Revises: prd251w1_merge_heads
Create Date: 2026-09-28
"""
from __future__ import annotations

from typing import List

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "prd251_wave2"
down_revision = "prd251w1_merge_heads"
branch_labels = None
depends_on = None

CAMPAIGN_APPROVAL_MODE_CHECK = "approval_mode IN ('per_post', 'series')"
CAMPAIGN_INDEXES = (
    ("ix_social_campaigns_workspace_created", ["workspace_id", "created_at"]),
)
# The name Postgres gives the model's key when create_all builds it; the model
# names it the same, so both writers agree and the downgrade finds it either way.
POST_CAMPAIGN_FK = "social_posts_campaign_id_fkey"
POST_CAMPAIGN_INDEX = "ix_social_posts_campaign_id"
# Where batch mode copies the table (SQLite), it reflects social_posts alone: by
# default it would also reflect every table the keys name (workspaces), which a
# partial schema need not hold.
POSTS_BATCH_REFLECT = {"resolve_fks": False}
POST_PREVIEW_COLUMN = "preview"


def _json():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _uuid():
    # Portable: native UUID on Postgres, CHAR(32) elsewhere (the model's type).
    return sa.Uuid(as_uuid=True)


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _create_missing_indexes(table: str, indexes) -> None:
    existing = {ix["name"] for ix in sa.inspect(op.get_bind()).get_indexes(table)}
    for name, columns in indexes:
        if name not in existing:
            op.create_index(name, table, columns)


def campaign_foreign_keys() -> List[str]:
    """The names of the keys from ``social_posts.campaign_id`` to ``social_campaigns``."""
    if not _has_table("social_posts"):
        return []
    return [
        fk["name"]
        for fk in sa.inspect(op.get_bind()).get_foreign_keys("social_posts")
        if fk["referred_table"] == "social_campaigns" and fk["constrained_columns"] == ["campaign_id"]
    ]


def create_social_campaigns() -> None:
    """``social_campaigns``, unless ``create_all`` already built it (then only its missing indexes)."""
    if _has_table("social_campaigns"):
        _create_missing_indexes("social_campaigns", CAMPAIGN_INDEXES)
        return
    op.create_table(
        "social_campaigns",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column(
            "workspace_id",
            _uuid(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("approval_mode", sa.String(16), nullable=False, server_default="per_post"),
        sa.Column("approved_hash_set", _json(), nullable=False),
        sa.Column("approved_by", sa.String(255), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(CAMPAIGN_APPROVAL_MODE_CHECK, name="ck_social_campaigns_approval_mode"),
    )
    for name, columns in CAMPAIGN_INDEXES:
        op.create_index(name, "social_campaigns", columns)


def link_posts_to_campaigns() -> None:
    """``social_posts.campaign_id`` → ``social_campaigns.id`` ON DELETE SET NULL, and its index.

    A schema without ``social_posts`` (a partial test schema) has nothing to link."""
    if not _has_table("social_posts"):
        return
    if not campaign_foreign_keys():
        with op.batch_alter_table("social_posts", reflect_kwargs=POSTS_BATCH_REFLECT) as batch:
            batch.create_foreign_key(
                POST_CAMPAIGN_FK, "social_campaigns", ["campaign_id"], ["id"], ondelete="SET NULL"
            )
    op.create_index(POST_CAMPAIGN_INDEX, "social_posts", ["campaign_id"], if_not_exists=True)


def unlink_posts_from_campaigns() -> None:
    """Drop the key (by the name the database holds) and the post index."""
    names = campaign_foreign_keys()
    if names:
        with op.batch_alter_table("social_posts", reflect_kwargs=POSTS_BATCH_REFLECT) as batch:
            for name in names:
                batch.drop_constraint(name, type_="foreignkey")
    if _has_table("social_posts"):
        op.drop_index(POST_CAMPAIGN_INDEX, table_name="social_posts", if_exists=True)


def _post_columns() -> List[str]:
    if not _has_table("social_posts"):
        return []
    return [column["name"] for column in sa.inspect(op.get_bind()).get_columns("social_posts")]


def add_post_preview_column() -> None:
    """US-208: ``social_posts.preview``, unless ``create_all`` already built it.
    A schema without ``social_posts`` (a partial test schema) has no table to add it to."""
    columns = _post_columns()
    if not columns or POST_PREVIEW_COLUMN in columns:
        return
    op.add_column("social_posts", sa.Column(POST_PREVIEW_COLUMN, _json(), nullable=True))


def drop_post_preview_column() -> None:
    if POST_PREVIEW_COLUMN in _post_columns():
        with op.batch_alter_table("social_posts", reflect_kwargs=POSTS_BATCH_REFLECT) as batch:
            batch.drop_column(POST_PREVIEW_COLUMN)


def upgrade() -> None:
    create_social_campaigns()
    link_posts_to_campaigns()
    add_post_preview_column()


def downgrade() -> None:
    drop_post_preview_column()
    unlink_posts_from_campaigns()
    for name, _columns in CAMPAIGN_INDEXES:
        op.drop_index(name, table_name="social_campaigns")
    op.drop_table("social_campaigns")
