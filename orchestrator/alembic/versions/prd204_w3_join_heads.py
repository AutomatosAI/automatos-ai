"""Join the re-forked revision heads left by #545 x #548 landing together.

PR #545 (``prd204_merge_heads``) and PR #548 (``w3_post201_merge_heads``)
each independently merged the SAME two heads (``prd201_s1_msg_context_trace``
+ ``prd203_merge_heads``) — they were built in parallel from the same base
and both landed on 2026-07-16. Merging both re-forked the graph:

    - w3_post201_merge_heads   (leaf — nothing chains onto it)
    - prd204_watch_registry    (chains onto prd204_merge_heads)

Neither file can simply be deleted: the deploy entrypoint runs
``alembic upgrade heads`` on boot, so deployed databases have already
recorded both lineages in ``alembic_version`` — removing either revision
would break every such environment on its next boot. The safe repair is
this third merge revision joining the two current heads, restoring the
CI "exactly one head" gate.

This is a **merge revision only** -- no schema operations. Do NOT add
``CREATE``/``ALTER`` here -- a merge revision that mutates schema is exactly
what makes a later from-zero squash lossy (mirrors prd176_merge_heads /
prd203_merge_heads / prd204_merge_heads).

Revision ID: prd204_w3_join_heads
Revises: w3_post201_merge_heads, prd204_watch_registry
Create Date: 2026-07-16
"""

# A pure merge point -- no operations.
revision = "prd204_w3_join_heads"
down_revision = (
    "w3_post201_merge_heads",
    "prd204_watch_registry",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op: this revision only merges lineages."""


def downgrade() -> None:
    """No-op: splitting back into two heads needs no schema change."""
