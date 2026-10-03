"""PRD-251B Socials Studio — the ONE migration for Wave 2 (plans and the content bank).

* US-B201 / B6: ``social_campaigns`` gains the plan fields. ``kind`` (``campaign|plan``),
  ``status`` (``active|paused|ended``) and ``late_policy`` (``skip|next_slot``), each NOT
  NULL with a server default and a CHECK by name; ``goal``, ``audience``, ``starts_on``,
  ``ends_on``, ``timezone``; and the JSON ``cadence``, ``sources``, ``make``, ``research``
  and ``slot_overrides``, all nullable (NULL reads as empty).
* US-B201 / B7: ``social_posts.slot_key``, the planned slot a plan made the post for,
  unique with ``campaign_id`` (``uq_social_posts_campaign_slot_key``: a NULL key never
  collides, so the rule binds only where a key is set). Not content.
* US-B201 / B8: ``social_topics``, the plan's content bank: title, angle, facts (each
  with its source), formats, pinned_on, the post that used it and when, and its origin
  (``research|person``), indexed by plan and use.
* The editor's music choice (US-B109): ``social_posts.music``, a render setting like
  ``voice``: NULL plays the template's own track. Outside the content hash; the rendered
  files' digests carry what it changed.

``core/models/socials.py`` declares the same shape, and
``tests/test_prd251bw2_migration.py`` holds the two together.

Create_all-first safe (the 89d89c250 rule): a backend that loaded the new models before
this ran has already built the table, the columns, the index and the CHECKs. The
upgrade adds a column only when ``get_columns`` lacks it, creates the table only when
missing and every index IF NOT EXISTS, and adds a CHECK only when absent by name, so
running it twice changes nothing. CHECKs go through ``batch_alter_table``: plain ALTERs
on Postgres; on SQLite (the unit tests), a copy of the table.

Chains single-parent on prd251b_wave1.

Revision ID: prd251b_wave2
Revises: prd251b_wave1
Create Date: 2026-10-03
"""
from __future__ import annotations

from typing import List

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "prd251b_wave2"
down_revision = "prd251b_wave1"
branch_labels = None
depends_on = None

CAMPAIGNS, POSTS, TOPICS = "social_campaigns", "social_posts", "social_topics"
KIND_CHECK = ("ck_social_campaigns_kind", "kind IN ('campaign', 'plan')")
STATUS_CHECK = ("ck_social_campaigns_status", "status IN ('active', 'paused', 'ended')")
LATE_CHECK = ("ck_social_campaigns_late_policy", "late_policy IN ('skip', 'next_slot')")
CAMPAIGN_CHECKS = (KIND_CHECK, STATUS_CHECK, LATE_CHECK)
TOPIC_ORIGIN_CHECK = ("ck_social_topics_origin", "origin IN ('research', 'person')")
SLOT_INDEX = "uq_social_posts_campaign_slot_key"
TOPIC_PLAN_INDEX = "ix_social_topics_campaign_used"
TOPIC_WORKSPACE_INDEX = "ix_social_topics_workspace_id"
# Batch mode (SQLite copies the table) reflects the one table alone, as prd251_wave2 does.
BATCH_REFLECT = {"resolve_fks": False}


def _json() -> sa.types.TypeEngine:
    return sa.JSON().with_variant(JSONB(), "postgresql")


def _campaign_columns() -> List[sa.Column]:
    return [
        sa.Column("kind", sa.String(16), nullable=False, server_default="campaign"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("goal", sa.Text(), nullable=True),
        sa.Column("audience", sa.Text(), nullable=True),
        sa.Column("starts_on", sa.Date(), nullable=True),
        sa.Column("ends_on", sa.Date(), nullable=True),
        sa.Column("timezone", sa.String(64), nullable=True),
        sa.Column("cadence", _json(), nullable=True),
        sa.Column("sources", _json(), nullable=True),
        sa.Column("make", _json(), nullable=True),
        sa.Column("late_policy", sa.String(16), nullable=False, server_default="skip"),
        sa.Column("research", _json(), nullable=True),
        sa.Column("slot_overrides", _json(), nullable=True),
    ]


def _post_columns() -> List[sa.Column]:
    return [sa.Column("slot_key", sa.String(160), nullable=True), sa.Column("music", _json(), nullable=True)]


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _columns(name: str) -> List[str]:
    if not _has_table(name):
        return []
    return [column["name"] for column in sa.inspect(op.get_bind()).get_columns(name)]


def _check_names(name: str) -> List[str]:
    if not _has_table(name):
        return []
    return [c["name"] for c in sa.inspect(op.get_bind()).get_check_constraints(name) if c.get("name")]


def add_columns(table: str, columns: List[sa.Column]) -> None:
    """Each column only when the table lacks it; a schema without the table has nothing to add to."""
    present = _columns(table)
    if not present:
        return
    for column in columns:
        if column.name not in present:
            op.add_column(table, column)


def add_campaign_checks() -> None:
    names = _check_names(CAMPAIGNS)
    missing = [(name, sql) for name, sql in CAMPAIGN_CHECKS if name not in names]
    if not _has_table(CAMPAIGNS) or not missing:
        return
    with op.batch_alter_table(CAMPAIGNS, reflect_kwargs=BATCH_REFLECT) as batch:
        for name, sql in missing:
            batch.create_check_constraint(name, sql)


def create_topics() -> None:
    """The content bank, when missing (create_all may have built it), and its indexes."""
    if not _has_table(TOPICS):
        op.create_table(
            TOPICS,
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("workspace_id", sa.Uuid(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
            sa.Column("campaign_id", sa.Uuid(as_uuid=True), sa.ForeignKey("social_campaigns.id", ondelete="CASCADE"), nullable=False),
            sa.Column("title", sa.String(200), nullable=False),
            sa.Column("angle", sa.Text(), nullable=True),
            sa.Column("facts", _json(), nullable=False),
            sa.Column("formats", _json(), nullable=False),
            sa.Column("pinned_on", sa.Date(), nullable=True),
            sa.Column("used_post_id", sa.Uuid(as_uuid=True), sa.ForeignKey("social_posts.id", ondelete="SET NULL"), nullable=True),
            sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("origin", sa.String(16), nullable=False, server_default="person"),
            sa.Column("created_by", sa.String(255), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint(TOPIC_ORIGIN_CHECK[1], name=TOPIC_ORIGIN_CHECK[0]),
        )
    op.create_index(TOPIC_PLAN_INDEX, TOPICS, ["campaign_id", "used_at"], if_not_exists=True)
    op.create_index(TOPIC_WORKSPACE_INDEX, TOPICS, ["workspace_id"], if_not_exists=True)


def upgrade() -> None:
    add_columns(CAMPAIGNS, _campaign_columns())
    add_campaign_checks()
    add_columns(POSTS, _post_columns())
    if _has_table(POSTS):
        op.create_index(SLOT_INDEX, POSTS, ["campaign_id", "slot_key"], unique=True, if_not_exists=True)
    if _has_table(CAMPAIGNS) and _has_table(POSTS):
        create_topics()


def _drop_columns(table: str, names: List[str], checks: List[str]) -> None:
    present, present_checks = _columns(table), _check_names(table)
    if not present:
        return
    with op.batch_alter_table(table, reflect_kwargs=BATCH_REFLECT) as batch:
        for name in checks:
            if name in present_checks:
                batch.drop_constraint(name, type_="check")
        for name in names:
            if name in present:
                batch.drop_column(name)


def downgrade() -> None:
    op.drop_index(TOPIC_WORKSPACE_INDEX, table_name=TOPICS, if_exists=True)
    op.drop_index(TOPIC_PLAN_INDEX, table_name=TOPICS, if_exists=True)
    if _has_table(TOPICS):
        op.drop_table(TOPICS)
    op.drop_index(SLOT_INDEX, table_name=POSTS, if_exists=True)
    _drop_columns(POSTS, [c.name for c in _post_columns()], [])
    _drop_columns(CAMPAIGNS, [c.name for c in _campaign_columns()], [name for name, _sql in CAMPAIGN_CHECKS])
