"""harness_task_ledger — the HARNESS task ledger, in the database.

F156: which done [HARNESS] board tasks self-management applied, and which
wait for an owner's or admin's /approve, used to live in
harness/applied_tasks.json on the workspace volume. It moves here, written only
by the HARNESS service; the first read for a workspace with nothing here
imports its old file (HarnessService._import_legacy_ledger), so no change is
applied twice. Idempotent (IF NOT EXISTS) so a create_all-first boot is safe.

Revision ID: f156_harness_task_ledger
Revises: f155_chats_widget_key_id
Create Date: 2026-09-25
"""
from alembic import op

revision = "f156_harness_task_ledger"
down_revision = "f155_chats_widget_key_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS harness_task_ledger (
            id SERIAL PRIMARY KEY,
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            board_task_id INTEGER NOT NULL,
            state VARCHAR(20) NOT NULL,
            entry JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_harness_task_ledger_task UNIQUE (workspace_id, board_task_id),
            CONSTRAINT ck_harness_task_ledger_state CHECK (state IN ('applied', 'held'))
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS harness_task_ledger;")
