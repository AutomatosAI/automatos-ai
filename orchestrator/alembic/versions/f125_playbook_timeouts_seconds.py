"""F125: a playbook's legacy millisecond timeouts become seconds, once.

From F125 the executor and the quality score read execution_config timeouts as
seconds (core/services/playbook_timeouts.py). Before, they guessed the unit from
the size, because the playbook editor saved milliseconds until PR #303
(2026-05-09), and rows from that time may still hold them.

This migration converts them once. A value of 100,000 or more becomes
value / 1000 seconds, never below the runtime floor, so the stored value is the
one that runs. As seconds, 100,000 would be 27.8 h, which is not a playbook
budget, and the old editor's defaults (120,000 and 600,000) are above it. Values
under 100,000 are seconds.

It is idempotent: every result is under 100,000. Each change is logged.
timeout_minutes and non-numeric values are never touched. The downgrade does
nothing: the milliseconds are not restored.

Revision ID: f125_playbook_timeouts_seconds
Revises: llm_usage_agent_name
Create Date: 2026-09-23
"""
import json
import logging

import sqlalchemy as sa
from alembic import op

revision = "f125_playbook_timeouts_seconds"
down_revision = "llm_usage_agent_name"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

LEGACY_MS_FROM = 100_000
STEP_KEYS = ("timeout_per_step", "per_step_timeout")
TOTAL_KEYS = ("total_timeout",)


def normalise(config, floors):
    """(the normalised config, one "key before -> after" line per change). Pure."""
    normalised = dict(config)
    changes = []
    for key, floor in floors.items():
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < LEGACY_MS_FROM:
            continue
        seconds = max(int(round(value / 1000)), int(floor))
        normalised[key] = seconds
        changes.append(f"{key} {value} -> {seconds}")
    return normalised, changes


def _floors():
    from config import config

    floors = {key: config.PLAYBOOK_MIN_STEP_TIMEOUT_SECONDS for key in STEP_KEYS}
    floors.update({key: config.PLAYBOOK_MIN_TOTAL_TIMEOUT_SECONDS for key in TOTAL_KEYS})
    return floors


def upgrade() -> None:
    floors = _floors()
    bind = op.get_bind()
    rows = bind.execute(sa.text(
        "SELECT id, execution_config FROM workflow_recipes WHERE execution_config IS NOT NULL"
    )).fetchall()
    changed = 0
    for recipe_id, config in rows:
        if not isinstance(config, dict):
            continue
        normalised, changes = normalise(config, floors)
        if not changes:
            continue
        bind.execute(
            sa.text("UPDATE workflow_recipes SET execution_config = CAST(:config AS jsonb) WHERE id = :id"),
            {"config": json.dumps(normalised), "id": recipe_id},
        )
        changed += 1
        logger.info("[f125] recipe %s: %s", recipe_id, "; ".join(changes))
    logger.info("[f125] %d of %d playbook configs held legacy millisecond timeouts", changed, len(rows))


def downgrade() -> None:
    """No-op: the milliseconds are not restored."""
