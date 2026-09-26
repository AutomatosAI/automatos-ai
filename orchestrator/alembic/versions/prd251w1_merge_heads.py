"""Join the night-fix and PRD-251 Wave 1 lineages back into one head.

Both start at ``f049_prd251_merge_heads``:

    - f155_chats_widget_key_id -> f156_harness_task_ledger -> llm_usage_execution_index
                            (F155, F156 and F153, the customer-night fixes)
    - prd251_wave1          (PRD-251 Wave 1, the video engine)

With both in one tree, ``alembic heads`` returns two revisions and the from-zero
"exactly one head" bar (test_prd209_alembic_single_head) fails. This is a
**merge revision only**, with no schema operations (mirrors f049_prd251_merge_heads).

Revision ID: prd251w1_merge_heads
Revises: llm_usage_execution_index, prd251_wave1
Create Date: 2026-09-26
"""

# A pure merge point — no operations.
revision = "prd251w1_merge_heads"
down_revision = (
    "llm_usage_execution_index",
    "prd251_wave1",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op: this revision only merges lineages."""


def downgrade() -> None:
    """No-op: splitting back into two heads needs no schema change."""
