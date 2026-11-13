"""A spend cap per product workspace (13 Nov 2026).

PRD-274 Wave 5, the spend cap proposed in #1520. Two additions, no row changed:

- ``products.workspace_monthly_credit_cap`` (a new nullable column, at least 1): the credits each of the product's
  workspaces may spend of the owner's in a calendar month (UTC). None: no cap.
- ``product_workspace_spend`` (a new table): each product workspace's spend per month and the owner's last notice,
  moved with each debit, so the credit gate reads the cap in one row. ``workspace_id`` cascades with the workspace.

``core/models/products.py`` declares the same. Create_all-first safe: ``ADD COLUMN IF NOT EXISTS``, the check added
only when missing, ``CREATE TABLE IF NOT EXISTS``. The downgrade drops both.

Revision ID: product_spend_cap_20261113
Revises: credit_ledger_sub_tenant_20261113
Create Date: 2026-11-13
"""
from alembic import op

revision = "product_spend_cap_20261113"
down_revision = "credit_ledger_sub_tenant_20261113"
branch_labels = None
depends_on = None

_CHECK = """
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_products_workspace_monthly_credit_cap') THEN
    ALTER TABLE products ADD CONSTRAINT ck_products_workspace_monthly_credit_cap
      CHECK (workspace_monthly_credit_cap IS NULL OR workspace_monthly_credit_cap >= 1);
  END IF;
END $$;
"""

_TABLE = """
CREATE TABLE IF NOT EXISTS product_workspace_spend (
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    month DATE NOT NULL,
    credits INTEGER NOT NULL DEFAULT 0,
    notified_level SMALLINT NOT NULL DEFAULT 0,
    PRIMARY KEY (workspace_id, month)
);
"""


def upgrade() -> None:
    op.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS workspace_monthly_credit_cap INTEGER;")
    op.execute(_CHECK)
    op.execute(_TABLE)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS product_workspace_spend;")
    op.execute("ALTER TABLE products DROP CONSTRAINT IF EXISTS ck_products_workspace_monthly_credit_cap;")
    op.execute("ALTER TABLE products DROP COLUMN IF EXISTS workspace_monthly_credit_cap;")
