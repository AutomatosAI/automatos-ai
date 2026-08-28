"""PRD-225 S1 — the asks model: extend approval_grants into free-text questions

A question **is** a grant whose decision is words instead of a boolean (CLAUDE.md
"no new tables when an existing one fits"). This is the WAVE'S ONLY schema change
— exactly one revision, chained onto the single head so ``alembic heads`` stays 1.

Additive + idempotent columns on ``approval_grants``:
  - ``kind``               — 'approval' (default, existing rows) | 'question'
  - ``question_md``        — the ask, markdown
  - ``options``            — optional discrete choices (rendered as buttons)
  - ``answer_text``        — the human's free-text answer
  - ``answered_by``        — actor ref, e.g. 'user:42' (mirrors ``granted_by``)
  - ``answered_at``        — when the answer landed
  - ``asked_by_agent_id``  — who raised the ask (mirrors ``agent_id``)
  - ``channel_refs``       — outbound delivery correlation, e.g.
                             ``{"telegram": {"chat_id": …, "message_id": …}}``

Status vocabulary is REUSED, not extended: pending = open ask, granted = answered,
denied = dismissed, expired via the existing ``expires_at``. The trail is rows —
re-asks are new rows against the same (subject_type, subject_id), read off the
existing hot index; no qa-history JSONB.

The ``ix_approval_grants_kind`` partial index serves the Questions tab's list
query (``workspace_id + kind='question' + status``) without a full scan.

``ADD COLUMN IF NOT EXISTS`` makes every step a no-op where it already applied,
so this is safe on fresh clones, CI, and any partially-migrated environment.
"""

from alembic import op


revision = "prd225_s1_asks_on_grants"
down_revision = "prd222_veteran_skip_backfill"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE approval_grants "
        "ADD COLUMN IF NOT EXISTS kind VARCHAR(16) NOT NULL DEFAULT 'approval';"
    )
    op.execute("ALTER TABLE approval_grants ADD COLUMN IF NOT EXISTS question_md TEXT;")
    op.execute("ALTER TABLE approval_grants ADD COLUMN IF NOT EXISTS options JSONB;")
    op.execute("ALTER TABLE approval_grants ADD COLUMN IF NOT EXISTS answer_text TEXT;")
    op.execute(
        "ALTER TABLE approval_grants ADD COLUMN IF NOT EXISTS answered_by VARCHAR(255);"
    )
    op.execute(
        "ALTER TABLE approval_grants ADD COLUMN IF NOT EXISTS answered_at TIMESTAMPTZ;"
    )
    op.execute(
        "ALTER TABLE approval_grants ADD COLUMN IF NOT EXISTS asked_by_agent_id INTEGER;"
    )
    op.execute("ALTER TABLE approval_grants ADD COLUMN IF NOT EXISTS channel_refs JSONB;")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_approval_grants_kind "
        "ON approval_grants (workspace_id, kind, status);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_approval_grants_kind;")
    op.execute("ALTER TABLE approval_grants DROP COLUMN IF EXISTS channel_refs;")
    op.execute("ALTER TABLE approval_grants DROP COLUMN IF EXISTS asked_by_agent_id;")
    op.execute("ALTER TABLE approval_grants DROP COLUMN IF EXISTS answered_at;")
    op.execute("ALTER TABLE approval_grants DROP COLUMN IF EXISTS answered_by;")
    op.execute("ALTER TABLE approval_grants DROP COLUMN IF EXISTS answer_text;")
    op.execute("ALTER TABLE approval_grants DROP COLUMN IF EXISTS options;")
    op.execute("ALTER TABLE approval_grants DROP COLUMN IF EXISTS question_md;")
    op.execute("ALTER TABLE approval_grants DROP COLUMN IF EXISTS kind;")
