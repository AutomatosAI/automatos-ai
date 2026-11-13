"""The product workspace a debit was for (13 Nov 2026).

PRD-274, "Automatos for products", Wave 5d (decisions 24 and 25). A product's workspace has no credits of its
own: its product's owner workspace pays (``services/credits/payer.py``). ``credit_ledger.sub_tenant_id`` (a new
nullable column) records which of the product's workspaces a debit on the owner's balance was for. It is not
named ``workspace_id`` and has no foreign key, so the owner's usage per workspace outlives that workspace's purge;
the purge's table sweep, which reads ``workspace_id`` columns, leaves it. A partial index serves the owner's usage
per workspace.

``core/models/credits.py`` declares the same. Create_all-first safe: ``ADD COLUMN IF NOT EXISTS`` and
``CREATE INDEX IF NOT EXISTS``; it changes no row. The downgrade drops both.

Revision ID: credit_ledger_sub_tenant_20261113
Revises: product_workspaces_20261113
Create Date: 2026-11-13
"""
from alembic import op

revision = "credit_ledger_sub_tenant_20261113"
down_revision = "product_workspaces_20261113"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE credit_ledger ADD COLUMN IF NOT EXISTS sub_tenant_id UUID;")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_credit_ledger_sub_tenant ON credit_ledger (workspace_id, sub_tenant_id) "
        "WHERE sub_tenant_id IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_credit_ledger_sub_tenant;")
    op.execute("ALTER TABLE credit_ledger DROP COLUMN IF EXISTS sub_tenant_id;")
