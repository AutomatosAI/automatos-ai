"""Add job_title column to agents table

Lets users give each agent a short role label (e.g. "Lead Intelligence",
"Code Watchdog", "Memory Keeper") shown alongside their codename on the
roster card. Keeps the codenames as identity while making the team's
purpose scannable.

Standalone migration (down_revision = None).
"""

from alembic import op
import sqlalchemy as sa


revision = "add_job_title_to_agents"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agents",
        sa.Column("job_title", sa.String(length=120), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agents", "job_title")
