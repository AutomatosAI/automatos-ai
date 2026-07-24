"""Wave-3 close-out: collapse the two-headed revision forest back to one head.

After PRD-201 (#543) merged alongside the PRD-203 merge revision (#542),
``alembic heads`` returned two divergent heads:

    - prd201_s1_msg_context_trace  (PRD-201 S1, message context trace)
    - prd203_merge_heads           (the PRD-203 three-way merge point)

Both PRs were cut from the same parent window and merged separately without a
joining revision, so main carries two heads and the from-zero "exactly one
head" CI bar has been red on every PR since. This is a **merge revision
only** — no schema operations — joining the two lineages so ``alembic heads``
returns exactly one revision (the deploy entrypoint's ``alembic upgrade
heads`` applied both lineages regardless, so prod state is unaffected).

Do NOT add ``CREATE``/``ALTER`` here — a merge revision that mutates schema is
exactly what makes a later from-zero squash lossy (mirrors prd176_merge_heads
and prd203_merge_heads).

Revision ID: w3_post201_merge_heads
Revises: prd201_s1_msg_context_trace, prd203_merge_heads
Create Date: 2026-07-16
"""

# A pure merge point — no operations.
revision = "w3_post201_merge_heads"
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
