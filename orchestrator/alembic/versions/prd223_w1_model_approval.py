"""PRD-223 Wave 1 — per-workspace model approval columns.

workspace_models gains the governance surface: approval_status (quarantine a
model workspace-wide), approved_roles (opt-in role grants from the promotion
gate), approval_evidence (how the grant was earned). No new table — the
per-workspace × per-model join row already existed (PRD-223 §5 Component A).
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "prd223_w1_model_approval"
down_revision = "prd223_w0_model_policy"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "workspace_models",
        sa.Column("approval_status", sa.String(length=20), nullable=False, server_default="unreviewed"),
    )
    op.add_column("workspace_models", sa.Column("approved_roles", sa.JSON(), nullable=True))
    op.add_column("workspace_models", sa.Column("approval_evidence", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("workspace_models", "approval_evidence")
    op.drop_column("workspace_models", "approved_roles")
    op.drop_column("workspace_models", "approval_status")
