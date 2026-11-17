"""PRD-251D Socials, one brief and one click — the ONE migration for Wave 3 (a post's look).

* US-D301 / E10: ``social_posts.look``, the post's own look on its template: light or dark,
  the accent strong or sparing, the logo on or off (``modules/socials/post_look.py``). Nullable
  JSON (JSONB on Postgres), NULL being the kit's own. A render setting like ``voice`` and
  ``music``: outside the content hash, so it changes no stored hash and no approval.

``core/models/socials.py`` declares the same column, and ``tests/test_prd251dw3_migration.py``
holds the two together.

Create_all-first safe (the 89d89c250 rule): the column is added only when ``get_columns`` lacks
it, the same test as ``ADD COLUMN IF NOT EXISTS`` but one SQLite also runs (the unit tests), so
running it twice changes nothing. It changes no row. The downgrade drops the column only when it
is there.

Chains single-parent on credit_ledger_sub_tenant_20261113, the head when it was written.

Revision ID: prd251d_wave3
Revises: credit_ledger_sub_tenant_20261113
Create Date: 2026-11-16
"""
from __future__ import annotations

from typing import List

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "prd251d_wave3"
down_revision = "credit_ledger_sub_tenant_20261113"
branch_labels = None
depends_on = None

POSTS = "social_posts"
LOOK_COLUMN = "look"
# Batch mode (SQLite copies the table) reflects the one table alone, as prd251c_wave2 does.
BATCH_REFLECT = {"resolve_fks": False}


def _json() -> sa.types.TypeEngine:
    return sa.JSON().with_variant(JSONB(), "postgresql")


def _columns() -> List[str]:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(POSTS):
        return []
    return [column["name"] for column in inspector.get_columns(POSTS)]


def upgrade() -> None:
    present = _columns()
    if not present:
        return  # a schema without the table has nothing to add to
    if LOOK_COLUMN not in present:
        op.add_column(POSTS, sa.Column(LOOK_COLUMN, _json(), nullable=True))


def downgrade() -> None:
    if LOOK_COLUMN not in _columns():
        return
    with op.batch_alter_table(POSTS, reflect_kwargs=BATCH_REFLECT) as batch:
        batch.drop_column(LOOK_COLUMN)
