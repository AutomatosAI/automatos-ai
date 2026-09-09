"""Merge the two heads the local tree carries — the PRD-236/237 landing merge
and the calendar chain — so the analytics revision has ONE parent.

No schema change. (2026-09-09)
"""

revision = "prd240_merge_heads"
down_revision = ("prd236w1_prd237_merge", "calendar_scheduled_board_tasks")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
