"""PRD-196 S3 (P2-15) — audit_logs composite index (workspace_id, created_at)

The audit-log read view (S3) and the retention sweep (S5) both scan
``audit_logs`` by ``workspace_id`` then order/filter by ``created_at`` — and the
table (``core/workspaces/audit.py``) had NO index on either column, so every
read and every sweep was a full scan. One composite index
``ix_audit_logs_workspace_created (workspace_id, created_at)`` serves both: the
ctx-workspace-scoped newest-first read and the ``created_at < cutoff`` batched
delete.

Chain: PRD-195 (#536) merged first with its fossil-DROP migration
(``prd195_drop_authz_fossil_tables``) chained onto prd191, so this migration
re-chains onto THAT (linear: prd191 -> prd195_drop -> prd196_s3) to keep exactly
one head. The entrypoint runs ``alembic upgrade heads`` on boot.

Revision ID: prd196_audit_logs_ws_created_idx
Revises: prd195_drop_authz_fossil_tables
Create Date: 2026-07-11
"""
from alembic import op

revision = "prd196_audit_logs_ws_created_idx"
down_revision = "prd195_drop_authz_fossil_tables"
branch_labels = None
depends_on = None

INDEX_NAME = "ix_audit_logs_workspace_created"


def upgrade() -> None:
    op.create_index(
        INDEX_NAME,
        "audit_logs",
        ["workspace_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="audit_logs")
