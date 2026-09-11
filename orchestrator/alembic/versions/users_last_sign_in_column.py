"""users.last_sign_in — make the column the console reads a guaranteed column.

``users.last_sign_in`` has been on the model (``core/models/core.py``) since the
Clerk cutover but NO migration ever declared it, so its existence rests entirely
on ``Base.metadata.create_all`` having built that table after the attribute
appeared. ``create_all`` CREATES missing tables; it never ALTERs an existing one
— so any database whose ``users`` table predates the attribute simply does not
have the column, silently, and nothing has ever noticed because nothing read it
(1 of 25 production rows were populated as of 2026-09-10).

The operator console now DOES read it — ``func.max(User.last_sign_in)`` in the
admin workspace list and detail routes — and a missing column there is a 500 on
every console load, not a blank cell. This migration closes that gap.

Production already carries the column (verified 2026-09-11 against
``information_schema.columns``: ``timestamp without time zone``), so upgrade is
a no-op there; ``IF NOT EXISTS`` keeps it a no-op on every create_all-first boot
too, matching the idempotent house style of ``prd240_llm_usage_cache_tokens``.

Nullable with no default on purpose: NULL is the honest "never seen", and the
console renders it as such rather than inventing an activity timestamp.

Revision ID: users_last_sign_in_column
Revises: prd240_llm_usage_cache_tokens
Create Date: 2026-09-11
"""

from alembic import op


revision = "users_last_sign_in_column"
down_revision = "prd240_llm_usage_cache_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_sign_in TIMESTAMP WITHOUT TIME ZONE;"
    )


def downgrade() -> None:
    # Deliberately a no-op. This migration only ever ADDS a column that most
    # databases (production included) already had, so dropping it on downgrade
    # would destroy sign-in history this migration never created. Reverting the
    # console feature does not require reverting the column.
    pass
