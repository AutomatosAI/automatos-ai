"""Merge the PRD-236 W1 and PRD-237 heads — no schema change.

``prd236_w1_serving_provider`` (PR #700) and ``prd237_users_chat_sessions``
(PR #702) both chain onto the PRD-234 S1a head. #700 was merged into its
stacked base branch after that base had already landed on main, so main carried
only the PRD-237 head while the local stack carried both. This revision joins
them: a fresh database upgrades through one linear chain, and a database that
already applied both heads collapses its version table to this one row.
"""

from alembic import op  # noqa: F401 — a merge revision has nothing to execute


revision = "prd236w1_prd237_merge"
down_revision = ("prd236_w1_serving_provider", "prd237_users_chat_sessions")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
