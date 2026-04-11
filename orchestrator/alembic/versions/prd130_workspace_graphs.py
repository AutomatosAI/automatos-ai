"""PRD-130: workspace_graphs table for durable Graphify persistence

Revision ID: prd130_workspace_graphs
Revises: prd130_business_profile
Create Date: 2026-04-11

Replaces the workspace-worker file write path for graph artefacts
(graph.json, meta.json, communities.json, graph.html, snapshots, reports).
The worker filesystem is not provisioned for every workspace, which caused
GraphifyService.build_graph() to succeed in memory but fail to persist.
This table gives us a guaranteed durable home keyed by (workspace_id, path).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "prd130_workspace_graphs"
down_revision = "prd130_business_profile"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workspace_graphs",
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("workspace_id", "path", name="pk_workspace_graphs"),
    )
    op.create_index(
        "ix_workspace_graphs_workspace",
        "workspace_graphs",
        ["workspace_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_workspace_graphs_workspace", table_name="workspace_graphs")
    op.drop_table("workspace_graphs")
