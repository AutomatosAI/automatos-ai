"""PRD-187 S5 — drop the dead rival memory stacks' tables

The MemoryKnowledgeSystem models (modules/memory/storage/knowledge_system.py,
deleted in this PR) defined five tables that never carried production data:

  memory_items            (0 rows — AdvancedMemoryManager's store; the manager
                           and its /api/v1/memory router are deleted)
  knowledge_nodes         (0 rows)
  knowledge_edges         (0 rows — FK → knowledge_nodes)
  learning_outcomes       (0 rows)
  harness_prescriptions   (0 rows — the W4-S12 dual-write cutover scaffold;
                           the harness baseline JSON store remains the
                           authoritative read/write path, unchanged)

All verified 0 rows in prod by the Phase-2 memory dossier (§C.6/E.4). Every
consumer was repointed to the real stores in the same PR (memory_stats +
workspaces read memory_short_term; the harness dual-write block is removed).

IF EXISTS because a from-zero database never creates these (their models are
gone); CASCADE because knowledge_edges FKs knowledge_nodes.

Downgrade is intentionally a no-op (repo convention for 0-row drops, see
prd135_drop_bucket_*): recreating empty tables blindly is never the right
recovery move when no data was involved.

Revision ID: prd187_s5_drop_memory_relics
Revises: prd185_s7_msg_retrieval_ctx
Create Date: 2026-07-10
"""
from alembic import op

revision = "prd187_s5_drop_memory_relics"
down_revision = "prd185_s7_msg_retrieval_ctx"
branch_labels = None
depends_on = None

_TABLES = (
    "knowledge_edges",
    "knowledge_nodes",
    "memory_items",
    "learning_outcomes",
    "harness_prescriptions",
)


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f'DROP TABLE IF EXISTS {table} CASCADE')


def downgrade() -> None:
    # Intentional no-op — all five tables held 0 rows when dropped; see module
    # docstring for the recovery posture.
    pass
