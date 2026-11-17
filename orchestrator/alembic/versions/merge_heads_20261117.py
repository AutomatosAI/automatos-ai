"""Join the two heads PRD-251D Wave 3 and the chat turn keys left on main (17 Nov 2026).

``prd251d_wave3`` (social_posts.look, #1539) and ``chat_turn_keys_20261116`` (the chat_turn_keys
table, #1534) were both written on ``credit_ledger_sub_tenant_20261113`` and merged the same
morning, minutes apart, each green against a main that did not hold the other. Main was left
with two heads.

The container runs ``alembic upgrade heads`` (plural), so a database deployed from that main has
both applied and both stamped. Neither revision is re-chained onto the other: a database stamped
at both refuses the rewritten lineage ("Requested revision ... overlaps with other requested
revisions ..."), which is what crash-looped the API in October (see prd266_merge_heads). A merge
revision is right for a database that has either parent, both, or neither.

A pure graph join: no schema operations, so there is nothing to guard for a database
``create_all`` built. Do not add CREATE or ALTER here.

Revision ID: merge_heads_20261117
Revises: prd251d_wave3, chat_turn_keys_20261116
Create Date: 2026-11-17
"""

# A pure merge point: no operations.
revision = "merge_heads_20261117"
down_revision = (
    "prd251d_wave3",
    "chat_turn_keys_20261116",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Nothing to do: this revision only joins the two lineages."""


def downgrade() -> None:
    """Nothing to do: splitting back into two heads changes no schema."""
