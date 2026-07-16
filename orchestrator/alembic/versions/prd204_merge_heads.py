"""PRD-204: collapse the two-headed revision forest back to one head.

At PRD-204 branch time ``alembic heads`` returns two divergent heads:

    - prd201_s1_msg_context_trace  (PRD-201, merged to main as its own child
      of prd196_audit_logs_ws_created_idx)
    - prd203_merge_heads           (the PRD-203 merge revision, which joined
      prd200 / prd202 / prd203_voice_turns but predated PRD-201's merge and
      so never included it)

Two heads make ``alembic upgrade head`` (singular) ambiguous and fail the
CI "exactly one head" gate. This is a **merge revision only** -- no schema
operations -- joining the two lineages so ``alembic heads`` returns exactly
one revision. The PRD-204 watch-registry migration chains onto this merge.

Do NOT add ``CREATE``/``ALTER`` here -- a merge revision that mutates schema
is exactly what makes a later from-zero squash lossy (mirrors
prd176_merge_heads / prd203_merge_heads).

Revision ID: prd204_merge_heads
Revises: prd201_s1_msg_context_trace, prd203_merge_heads
Create Date: 2026-07-16
"""

# A pure merge point -- no operations.
revision = "prd204_merge_heads"
down_revision = (
    "prd201_s1_msg_context_trace",
    "prd203_merge_heads",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op: this revision only merges lineages."""


def downgrade() -> None:
    """No-op: splitting back into two heads needs no schema change."""
