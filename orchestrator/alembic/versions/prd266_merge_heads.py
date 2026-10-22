"""PRD-266/265: collapse the two-headed revision fork back to one head.

PRD-265 (credit ledger, #1259, 21 Oct 09:13) and PRD-266 (mobile devices, #1264,
21 Oct 16:08) both branched their migration from the same parent
(``clerk_user_relinks_20261019``) without a merge revision — PRD-266 merged
seven hours after PRD-265 without rebasing. Production had already run
``alembic upgrade heads`` against the two-head state, stamping BOTH
``prd265_credit_ledger`` and ``prd266_mobile_devices`` as independently applied.

A prior fix attempted to resolve this by rewriting ``prd266_mobile_devices``'s
``down_revision`` to chain onto ``prd265_credit_ledger`` directly. That broke
production: a database already stamped at both (now-)overlapping revisions
raised ``Requested revision prd266_mobile_devices overlaps with other
requested revisions prd265_credit_ledger`` and crash-looped the API. Rewriting
an already-deployed revision's lineage doesn't work once any real database has
stamped it — only a merge revision resolves a database that already has either,
or both, parents applied. ``prd266_mobile_devices``'s own ``down_revision`` is
reverted to its original ``clerk_user_relinks_20261019``; this is a **merge
revision only** — no schema operations — joining the two lineages so
``alembic heads`` returns exactly one revision again (mirrors
prd203_merge_heads, prd176_merge_heads).

Do NOT add ``CREATE``/``ALTER`` here — see prd203_merge_heads's note on why a
merge revision must stay a pure graph join.

Revision ID: prd266_merge_heads
Revises: prd265_credit_ledger, prd266_mobile_devices
Create Date: 2026-10-22
"""

# A pure merge point — no operations.
revision = "prd266_merge_heads"
down_revision = (
    "prd265_credit_ledger",
    "prd266_mobile_devices",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op: this revision only merges lineages."""


def downgrade() -> None:
    """No-op: splitting back into two heads needs no schema change."""
