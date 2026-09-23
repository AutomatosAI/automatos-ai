"""PRD-251 Socials — the ONE migration for Wave 0.

* Seeds the ``socials`` system-settings row (category ``socials``, key
  ``enabled``): the platform master switch (D1), a super-admin toggle in
  Settings → System Settings, the way ``prd207_voice_live`` seeds
  ``voice.live_enabled``. Its initial value is the deployment's
  ``SOCIALS_ENABLED_DEFAULT`` at migration time (default off), so a stack
  that ships with Socials on starts on, and the page shows the real state.
  After that the row is the switch: the super-admin's choice wins over the env.
* Seeds the platform-wide Composio deny list (category ``composio``, key
  ``denied_actions``, S0.6 / D16): a JSON list of action slugs that spend real
  money or act outside Automatos, refused for every caller before any network
  call (``core/composio/deny_list.py``). The list is data the super-admin edits
  in Settings → System Settings; this seed is its first value and its default.
* Creates the two D2 tables (``core/models/socials.py`` declares the same shape;
  ``tests/test_prd251_models.py`` holds the two together):
  - ``social_posts``: one row per post, its approval bound to ``content_hash`` (D6);
  - ``social_post_targets``: one row per channel per post, with a UNIQUE
    ``idempotency_key``.
  JSON columns are JSONB on Postgres (the ``JSON().with_variant`` of the model).
  The campaigns table is NOT created (D2: Wave 2, only if series approval ships).

Insert-if-absent seed: ``system_settings`` has no (category, key) unique
constraint, so the upgrade checks first (the voice precedent), and a re-run
never overwrites a super-admin's choice. The downgrade drops exactly what the
upgrade creates: the two tables with their indexes, and the seeded rows.

Chains single-parent on kb_multimodal_tables (the current single head).

Revision ID: prd251_socials
Revises: kb_multimodal_tables
Create Date: 2026-09-23
"""
from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "prd251_socials"
down_revision = "kb_multimodal_tables"
branch_labels = None
depends_on = None

SEED_CREATED_BY = "prd251"
SEEDED_CATEGORIES = ("socials", "composio")

# D16: the Composio actions no agent, Playbook or API caller may run. Buying
# credits, changing plans and deploying stay with a person, in the tool's own
# interface (Composio docs, higgsfield_mcp, verified 2026-09-23).
COMPOSIO_DENIED_ACTIONS_SEED = (
    "HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE",
    "HIGGSFIELD_MCP_CANCEL_TRIAL_AUTO_RENEWAL",
    "HIGGSFIELD_MCP_CONFIRM_TRIAL_CANCEL",
    "HIGGSFIELD_MCP_CREATE_WEBSITE",
    "HIGGSFIELD_MCP_DEPLOY_WEBSITE",
    "HIGGSFIELD_MCP_PUBLISH_WEBSITE",
    "HIGGSFIELD_MCP_PARTICIPATE_IN_CONTEST",
    "HIGGSFIELD_MCP_APPS_INVOKE",
)

POST_STATUS_CHECK = (
    "status IN ('draft', 'rendering', 'needs_approval', 'changes_requested', 'approved', "
    "'scheduled', 'publishing', 'published', 'partially_published', 'failed', 'missed', "
    "'archived')"
)
POST_FORMAT_CHECK = "format IN ('video', 'image', 'carousel', 'fact_card', 'infographic')"
TARGET_POST_KIND_CHECK = (
    "post_kind IN ('text', 'image', 'carousel', 'video', 'reel', 'short', 'story')"
)
TARGET_STATUS_CHECK = "status IN ('pending', 'uploading', 'published', 'failed')"

POST_INDEXES = (
    ("ix_social_posts_workspace_status", ["workspace_id", "status"]),
    ("ix_social_posts_workspace_scheduled_for", ["workspace_id", "scheduled_for"]),
)
TARGET_INDEXES = (("ix_social_post_targets_post_id", ["post_id"]),)


def _json():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _uuid():
    # Portable: native UUID on Postgres, CHAR(32) elsewhere (the model's type).
    return sa.Uuid(as_uuid=True)


def _socials_settings_seed() -> tuple:
    from config import config

    default = "true" if config.SOCIALS_ENABLED_DEFAULT else "false"
    return (
        {
            "category": "socials",
            "key": "enabled",
            "value": default,
            "value_type": "boolean",
            "description": (
                "Socials master switch (PRD-251). OFF = no Socials tab and every "
                "/api/socials route answers 404, platform-wide. ON = each workspace "
                "can turn Socials on for itself. No redeploy."
            ),
            "is_sensitive": False,
            "is_required": True,
            "default_value": default,
        },
    )


def _composio_settings_seed() -> tuple:
    denied = json.dumps(list(COMPOSIO_DENIED_ACTIONS_SEED))
    return (
        {
            "category": "composio",
            "key": "denied_actions",
            "value": denied,
            "value_type": "json",
            "description": (
                "Composio deny list (PRD-251 D16): a JSON list of action slugs refused "
                "for every agent, Playbook and API caller in every workspace, before any "
                "network call and whatever the policy plane mode. Case-insensitive. "
                "Takes effect on the next call — no restart or redeploy."
            ),
            "is_sensitive": False,
            "is_required": True,
            "default_value": denied,
        },
    )


def _seed_settings(conn, rows) -> None:
    for row in rows:
        exists = conn.execute(
            sa.text("SELECT 1 FROM system_settings WHERE category = :category AND key = :key"),
            {"category": row["category"], "key": row["key"]},
        ).first()
        if exists:
            continue
        conn.execute(
            sa.text(
                "INSERT INTO system_settings "
                "(category, key, value, value_type, description, is_sensitive, "
                " is_required, default_value, created_by) "
                "VALUES (:category, :key, :value, :value_type, :description, "
                "        :is_sensitive, :is_required, :default_value, :created_by)"
            ),
            {**row, "created_by": SEED_CREATED_BY},
        )


def _create_social_posts() -> None:
    op.create_table(
        "social_posts",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column(
            "workspace_id",
            _uuid(),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("campaign_id", _uuid(), nullable=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("brief", sa.Text(), nullable=True),
        sa.Column("copy", _json(), nullable=False),
        sa.Column("format", sa.String(20), nullable=True),
        sa.Column("template_id", _uuid(), nullable=True),
        sa.Column("variables", _json(), nullable=False),
        sa.Column("sources", _json(), nullable=False),
        sa.Column("media", _json(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="draft"),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("approved_hash", sa.String(64), nullable=True),
        sa.Column("approved_by", sa.String(255), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("override_unsourced", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("review_log", _json(), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("timezone", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(POST_STATUS_CHECK, name="ck_social_posts_status"),
        sa.CheckConstraint(POST_FORMAT_CHECK, name="ck_social_posts_format"),
    )
    for name, columns in POST_INDEXES:
        op.create_index(name, "social_posts", columns)


def _create_social_post_targets() -> None:
    op.create_table(
        "social_post_targets",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column(
            "post_id",
            _uuid(),
            sa.ForeignKey("social_posts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("toolkit", sa.String(100), nullable=False),
        sa.Column("post_kind", sa.String(20), nullable=False),
        sa.Column("action_plan", _json(), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("remote_id", sa.String(255), nullable=True),
        sa.Column("permalink", sa.String(1000), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("idempotency_key", name="uq_social_post_targets_idempotency_key"),
        sa.CheckConstraint(TARGET_POST_KIND_CHECK, name="ck_social_post_targets_post_kind"),
        sa.CheckConstraint(TARGET_STATUS_CHECK, name="ck_social_post_targets_status"),
    )
    for name, columns in TARGET_INDEXES:
        op.create_index(name, "social_post_targets", columns)


def upgrade() -> None:
    _seed_settings(op.get_bind(), _socials_settings_seed() + _composio_settings_seed())
    _create_social_posts()
    _create_social_post_targets()


def downgrade() -> None:
    for name, _columns in TARGET_INDEXES:
        op.drop_index(name, table_name="social_post_targets")
    op.drop_table("social_post_targets")
    for name, _columns in POST_INDEXES:
        op.drop_index(name, table_name="social_posts")
    op.drop_table("social_posts")
    op.get_bind().execute(
        sa.text(
            "DELETE FROM system_settings WHERE category IN :categories AND created_by = :created_by"
        ).bindparams(sa.bindparam("categories", expanding=True)),
        {"categories": list(SEEDED_CATEGORIES), "created_by": SEED_CREATED_BY},
    )
