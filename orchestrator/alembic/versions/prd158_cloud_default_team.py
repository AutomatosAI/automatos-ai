"""prd158 cloud-sync per-connection default team

Revision ID: prd158_cloud_default_team
Revises: prd158_teams
Create Date: 2026-06-12

PRD-158 S2/Q5: a cloud-sync connection gets a default team applied to every
document synced from it (empty = public).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "prd158_cloud_default_team"
down_revision = "prd158_teams"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cloud_sync_config",
        sa.Column(
            "default_team_access",
            postgresql.ARRAY(sa.String()),
            server_default="{}",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("cloud_sync_config", "default_team_access")
