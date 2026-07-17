"""PRD-199 S5: drop the never-written stats columns from database sources.

``database_knowledge_sources.total_queries_executed`` and
``avg_query_time_ms`` were written by no code path, ever — permanent fake
zeros serialized into every source payload (the "no fake zeros" honest-UI
rule). If real per-source stats are ever wanted, derive them from
``database_query_audit`` (the dual-entry choke point), don't resurrect
unwritten counters.

Chains onto prd197_substrate_metrics (which chains onto the
prd204_w3_join_heads join) — single-headed by construction. The
prd197_substrate_metrics file rides in both #550 and this PR byte-identical,
so whichever merges first, git dedupes the other.

Revision ID: prd199_drop_fake_stats
Revises: prd197_substrate_metrics
Create Date: 2026-07-16
"""

import sqlalchemy as sa
from alembic import op

revision = "prd199_drop_fake_stats"
down_revision = "prd197_substrate_metrics"
branch_labels = None
depends_on = None

_TABLE = "database_knowledge_sources"


def upgrade() -> None:
    with op.batch_alter_table(_TABLE) as batch:
        batch.drop_column("total_queries_executed")
        batch.drop_column("avg_query_time_ms")


def downgrade() -> None:
    with op.batch_alter_table(_TABLE) as batch:
        batch.add_column(
            sa.Column("total_queries_executed", sa.Integer(), server_default="0")
        )
        batch.add_column(sa.Column("avg_query_time_ms", sa.Float(), nullable=True))
