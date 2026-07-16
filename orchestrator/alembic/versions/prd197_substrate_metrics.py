"""PRD-197 S4: substrate_metric_events — per-seam retrieval telemetry.

One row per substrate search (documents / memory / field seams) with
candidates-returned, latency, and hit/empty/error status. Aggregated by
services/substrate_health.py into the Command Center substrate tile.
Pruned to SUBSTRATE_METRICS_RETENTION_DAYS by the memory-jobs sweep.

Chains onto prd204_w3_join_heads (the #545 x #548 heads join) so the
graph stays single-headed.

Revision ID: prd197_substrate_metrics
Revises: prd204_w3_join_heads
Create Date: 2026-07-16
"""

import sqlalchemy as sa
from alembic import op

revision = "prd197_substrate_metrics"
down_revision = "prd204_w3_join_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "substrate_metric_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("seam", sa.String(length=16), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("candidates", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "latency_ms", sa.Float(), nullable=False, server_default="0"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_substrate_metrics_seam_created",
        "substrate_metric_events",
        ["seam", "created_at"],
    )
    op.create_index(
        "idx_substrate_metrics_ws_created",
        "substrate_metric_events",
        ["workspace_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_substrate_metrics_ws_created", table_name="substrate_metric_events"
    )
    op.drop_index(
        "idx_substrate_metrics_seam_created", table_name="substrate_metric_events"
    )
    op.drop_table("substrate_metric_events")
