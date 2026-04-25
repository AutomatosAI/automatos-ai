"""PRD-135 §12 — Bucket 6: drop 4 misc legacy tables

Final cleanup pass.

Tables dropped:
  playbooks         — legacy "learning patterns" table. The canonical
                     Playbook in the current platform is `workflow_recipes`
                     (alive: 388+ runtime calls). This is from a much
                     earlier wave that was abandoned.
  schema_versions   — superseded by alembic_version. Was a manual
                     migration tracker before alembic was adopted.
  code_symbols      — graphify-style code-graph tables that were
  code_edges          designed but never wired into the platform's own
                     code-graph indexing.

All 4: zero rows, zero inbound code edges, zero runtime hits.

Smoke test after drop: none — these are pure legacy with no live
analogues to test.

Rollback: schema-only snapshot at
graphify-out/snapshots/bucket-6-pre-drop.sql.

Note: PRD-135 §12 calls this "Bucket 6 — 5 tables" but the dead-tables
list (51 total) accounted for: B1=11, B2=10, B3=7, B4=4, B5=15, B6=4.
4 is the correct count.

Revision ID: prd135_drop_bucket_6
Revises: None  (independent head, matches alembic convention in this repo)
Create Date: 2026-04-25
"""

from alembic import op


# revision identifiers, used by Alembic.
# (Short ID — alembic_version.version_num is varchar(32))
revision = "prd135_drop_bucket_6"
down_revision = None
branch_labels = None
depends_on = None


_LEGACY_TABLES = (
    "playbooks",
    "schema_versions",
    "code_symbols",
    "code_edges",
)


def upgrade() -> None:
    # CASCADE drops outbound FK constraints with the table itself.
    # IF EXISTS makes the migration idempotent.
    for table in _LEGACY_TABLES:
        op.execute(f'DROP TABLE IF EXISTS public."{table}" CASCADE;')


def downgrade() -> None:
    # See module docstring: no-op by design. Restore from
    # graphify-out/snapshots/bucket-6-pre-drop.sql if recovery needed.
    pass
