"""Collapse the two-headed revision fork of 16-17 Nov 2026 back to one head.

Two revisions were written on the same parent, ``credit_ledger_sub_tenant_20261113``, and merged
within the same morning without a merge revision:

* ``chat_turn_keys_20261116``: a server key's chat turn under an Idempotency-Key (#1534);
* ``prd251d_wave3``: a social post's own look on its template (#1539).

``alembic heads`` then returned both. A database may already have either, or both, applied, so
neither revision's ``down_revision`` is rewritten (see prd266_merge_heads for why a rewrite breaks a
database that has stamped both). This is a **merge revision only**, with no schema operations, so
``alembic heads`` returns exactly one revision again (mirrors prd266_merge_heads).

Do NOT add ``CREATE``/``ALTER`` here: a merge revision stays a pure graph join.

Revision ID: merge_heads_20261117
Revises: chat_turn_keys_20261116, prd251d_wave3
Create Date: 2026-11-17
"""

# A pure merge point: no operations.
revision = "merge_heads_20261117"
down_revision = (
    "chat_turn_keys_20261116",
    "prd251d_wave3",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op: this revision only merges lineages."""


def downgrade() -> None:
    """No-op: splitting back into two heads needs no schema change."""
