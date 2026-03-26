"""Drop legacy item_id NOT NULL and FK on marketplace_installs

Revision ID: 20260326_fix_installs_item_id
Revises: None (standalone)
Create Date: 2026-03-26

The marketplace moved from a separate marketplace_items table to using
agents/recipes directly (owner_type='marketplace'). The install tracking
now uses marketplace_agent_id + cloned_agent_id, but the original item_id
column was left as NOT NULL with a FK to marketplace_items. This causes
500 errors on every install attempt.
"""

from alembic import op
import sqlalchemy as sa

revision = "20260326_fix_installs_item_id"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade():
    # Drop the FK constraint on item_id -> marketplace_items.id
    # The constraint name follows SQLAlchemy's auto-naming convention
    try:
        op.drop_constraint(
            "marketplace_installs_item_id_fkey",
            "marketplace_installs",
            type_="foreignkey",
        )
    except Exception:
        # Constraint may already be gone or named differently
        pass

    # Make item_id nullable so inserts no longer require it
    op.alter_column(
        "marketplace_installs",
        "item_id",
        existing_type=sa.Integer(),
        nullable=True,
    )


def downgrade():
    op.alter_column(
        "marketplace_installs",
        "item_id",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.create_foreign_key(
        "marketplace_installs_item_id_fkey",
        "marketplace_installs",
        "marketplace_items",
        ["item_id"],
        ["id"],
    )
