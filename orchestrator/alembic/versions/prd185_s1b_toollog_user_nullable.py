"""Telemetry rows must land without a user — repair the NOT NULL drift.

``tool_execution_logs.user_id`` is nullable in the model (composio_cache.py)
and in the prd142 DDL, and PRD-185 S1's design intent is explicit: an
unresolvable principal coerces to ``None`` so the row STILL lands. The live
production table predates all of that — it was born NOT NULL via
``create_all`` of an older model shape, and ``create_all`` never alters an
existing table, so the drift persisted invisibly.

Consequence (2026-08-01 Inbuild incident): every user-less tool execution —
heartbeat/cadence agent calls have no user by construction, and workspace
members whose Clerk id has no ``users`` row resolve to ``None`` — failed its
telemetry INSERT with NotNullViolation. The learning plane lost those rows,
and on paths that share the caller's session the failed flush poisoned the
caller's transaction.

Verified against production before writing this migration:
``information_schema.columns`` → ``user_id | is_nullable = NO``.

``ALTER COLUMN ... DROP NOT NULL`` is a no-op on an already-nullable column,
so this is safe wherever the drift does not exist (fresh clones, CI).
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "prd185_s1b_toollog_user_nullable"
down_revision = "prd223_w1_model_approval"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        "tool_execution_logs",
        "user_id",
        existing_type=sa.Integer(),
        nullable=True,
    )


def downgrade():
    # Best-effort: SET NOT NULL fails if user-less rows exist by then — which
    # is the expected state once heartbeat telemetry lands. Backfill or delete
    # NULL rows before downgrading.
    op.alter_column(
        "tool_execution_logs",
        "user_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
