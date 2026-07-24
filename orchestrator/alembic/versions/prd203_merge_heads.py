"""PRD-203: collapse the three-headed revision forest back to one head.

After PRD-203 merged current main, ``alembic heads`` returned three divergent
heads — all children of ``prd196_audit_logs_ws_created_idx``:

    - prd200_s2_drop_checkpoint_count      (PRD-200, merged to main)
    - prd202_s1_skill_l1_trigger_backfill  (PRD-202, merged to main)
    - prd203_voice_turns                   (this PR, V·S6)

PRD-200 and PRD-202 merged as separate PRs off the same parent without a merge
revision, so main already carried two heads; PRD-203's voice_turns table added
the third. Three heads make ``alembic upgrade head`` (singular) ambiguous and
fail the from-zero "exactly one head" bar. This is a **merge revision only** —
no schema operations — joining the three lineages so ``alembic heads`` returns
exactly one revision (the deploy entrypoint's ``alembic upgrade heads`` still
applies all three tables regardless).

Do NOT add ``CREATE``/``ALTER`` here — a merge revision that mutates schema is
exactly what makes a later from-zero squash lossy (mirrors prd176_merge_heads).

Revision ID: prd203_merge_heads
Revises: prd200_s2_drop_checkpoint_count, prd202_s1_skill_l1_trigger_backfill, prd203_voice_turns
Create Date: 2026-07-15
"""

# A pure merge point — no operations.
revision = "prd203_merge_heads"
down_revision = (
    "prd200_s2_drop_checkpoint_count",
    "prd202_s1_skill_l1_trigger_backfill",
    "prd203_voice_turns",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op: this revision only merges lineages."""


def downgrade() -> None:
    """No-op: splitting back into three heads needs no schema change."""
