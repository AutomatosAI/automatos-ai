"""PRD-251 Socials — the ONE migration for Wave 0.

* Seeds the ``socials`` system-settings row (category ``socials``, key
  ``enabled``): the platform master switch (D1), a super-admin toggle in
  Settings → System Settings, the way ``prd207_voice_live`` seeds
  ``voice.live_enabled``. Its initial value is the deployment's
  ``SOCIALS_ENABLED_DEFAULT`` at migration time (default off), so a stack
  that ships with Socials on starts on, and the page shows the real state.
  After that the row is the switch: the super-admin's choice wins over the env.

Insert-if-absent: ``system_settings`` has no (category, key) unique constraint,
so the upgrade checks first (the voice precedent), and a re-run never
overwrites a super-admin's choice. The downgrade deletes only the rows this
revision created.

Chains single-parent on kb_multimodal_tables (the current single head).

Revision ID: prd251_socials
Revises: kb_multimodal_tables
Create Date: 2026-09-23
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "prd251_socials"
down_revision = "kb_multimodal_tables"
branch_labels = None
depends_on = None

SEED_CREATED_BY = "prd251"


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


def upgrade() -> None:
    _seed_settings(op.get_bind(), _socials_settings_seed())


def downgrade() -> None:
    op.get_bind().execute(
        sa.text(
            "DELETE FROM system_settings WHERE category = 'socials' AND created_by = :created_by"
        ),
        {"created_by": SEED_CREATED_BY},
    )
