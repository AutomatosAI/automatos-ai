"""PRD-202 S1 (P2-21) — persist each skill's description as its L1 trigger text

The Agent Skills open standard loads a skill's ``name`` + ``description`` on
every turn (L1), and its body only when the ``description`` matches the task
(L2 on trigger). For that to work the ``description`` must live where the L1
loader reads it — ``skill_metadata`` (the JSONB the loader treats as L1) — not
only on the ``description`` column.

This is a pure, idempotent DATA backfill: for every skill that has a
``description`` but whose ``skill_metadata`` lacks a ``description`` key, copy it
in. It writes no schema and touches no ``skill_source`` string — the canonical
provenance scheme (:mod:`modules.agents.services.skill_source_scheme`) *resolves*
legacy values at read time, so no risky reshape of the overloaded git join key
is needed here (that repair is the Q3 data-pass, Gerard's cadence call).

Chain: onto the mainline tip ``prd196_audit_logs_ws_created_idx`` (which already
carries PRD-191's agent_skills unique/priority work — untouched here). The
deploy entrypoint runs ``alembic upgrade heads`` on boot, so this self-applies.

Revision ID: prd202_s1_skill_l1_trigger_backfill
Revises: prd196_audit_logs_ws_created_idx
Create Date: 2026-07-14
"""
from alembic import op

revision = "prd202_s1_skill_l1_trigger_backfill"
down_revision = "prd196_audit_logs_ws_created_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Postgres JSONB: seed skill_metadata.description from the description column
    # wherever it is missing. jsonb_set on a coalesced '{}' handles NULL metadata.
    op.execute(
        """
        UPDATE skills
        SET skill_metadata = jsonb_set(
            COALESCE(skill_metadata, '{}'::jsonb),
            '{description}',
            to_jsonb(description),
            true
        )
        WHERE description IS NOT NULL
          AND description <> ''
          AND (
            skill_metadata IS NULL
            OR NOT (skill_metadata ? 'description')
            OR COALESCE(skill_metadata->>'description', '') = ''
          )
        """
    )


def downgrade() -> None:
    # Data-only enrichment; nothing structural to reverse. The L1 loader also
    # falls back to the description column, so a downgrade is a no-op.
    pass
