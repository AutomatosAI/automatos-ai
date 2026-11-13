"""A product's workspaces (13 Nov 2026).

PRD-274, "Automatos for products", Wave 5b (§5.1). A product's key finds or creates its customers' workspaces:

- ``products.reuse_existing_key`` (a new column, default true): finding a workspace again mints no new key while
  it has an active one of the product's key type; false mints one on every find.
- ``uq_workspaces_product_external_id`` (a new unique partial index on ``workspaces``): one workspace per
  ``settings.product_id`` and ``settings.source_external_id``, for workspaces that carry a product. No workspace
  carries one yet, so it builds over no rows and changes none.

``core/models/products.py`` and ``core/models/workspaces.py`` declare the same. Create_all-first safe:
``ADD COLUMN IF NOT EXISTS`` and ``CREATE UNIQUE INDEX IF NOT EXISTS``. The downgrade drops both.

Revision ID: product_workspaces_20261113
Revises: products_20261113
Create Date: 2026-11-13
"""
from alembic import op

revision = "product_workspaces_20261113"
down_revision = "products_20261113"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS reuse_existing_key BOOLEAN NOT NULL DEFAULT true;")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_workspaces_product_external_id "
        "ON workspaces ((settings->>'product_id'), (settings->>'source_external_id')) "
        "WHERE settings->>'product_id' IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_workspaces_product_external_id;")
    op.execute("ALTER TABLE products DROP COLUMN IF EXISTS reuse_existing_key;")
