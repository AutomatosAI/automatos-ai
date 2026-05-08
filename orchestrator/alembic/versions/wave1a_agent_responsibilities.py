"""Wave 1.A — Agent organisation: responsibilities

Adds the missing structured field that lets Auto reason about who owns what
without having to grep an agent's persona text. ``team`` and ``reports_to_id``
already exist (Mission Zero).

Idempotent — safe to run anywhere.
"""

from alembic import op


revision = "wave1a_agent_responsibilities"
down_revision = "dedupe_skills_unique_workspace_name"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agents
        ADD COLUMN IF NOT EXISTS responsibilities JSONB NOT NULL DEFAULT '[]'::jsonb;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS responsibilities;")
