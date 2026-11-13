"""An API run's Idempotency-Key files one ticket (9 Nov 2026).

PRD-274, "Automatos for products", Wave 1 (§5.2). ``POST /api/agents/{id}/execute`` keeps a
request's ``Idempotency-Key`` as its ticket's ``source_id`` (with ``source_type = 'api'``). One
partial unique index makes the key file one ticket per workspace, even when two requests race:

* ``uq_board_tasks_api_idempotency`` on ``board_tasks (workspace_id, source_id)`` where
  ``source_type = 'api' AND source_id IS NOT NULL``.

``core/models/core.py`` (``BoardTask``) declares the same index.

Create_all-first safe: ``CREATE UNIQUE INDEX IF NOT EXISTS``, so a database ``create_all`` built is
left as it is. No existing row can collide: ``api`` tickets are new in Wave 1 and none carried a
``source_id`` before this. It changes no row. The downgrade drops the index.

Revision ID: execute_idempotency_20261109
Revises: run_callbacks_20261109
Create Date: 2026-11-09
"""
from alembic import op

revision = "execute_idempotency_20261109"
down_revision = "run_callbacks_20261109"
branch_labels = None
depends_on = None

_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_board_tasks_api_idempotency ON board_tasks (workspace_id, source_id) "
    "WHERE source_type = 'api' AND source_id IS NOT NULL;"
)


def upgrade() -> None:
    op.execute(_INDEX)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_board_tasks_api_idempotency;")
