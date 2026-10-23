"""PRD-265 W2 (in Studio) — billing_accounts remembers scheduled downgrades and Business offers.

Payments moved inside Studio (Gerard, 23 Oct 2026): a downgrade now waits for the
period end through a Stripe subscription schedule the Billing tab can undo, and a
platform admin offers Business in-app instead of sending a link. Four nullable
columns on ``billing_accounts``:

- ``schedule_id`` / ``scheduled_plan`` / ``scheduled_at`` — the waiting downgrade;
- ``business_offer_until`` — when a platform admin's Business offer lapses.

``prd265_billing`` already ran in production with the table, so this is a separate
revision: nullable columns added without a default (no table rewrite), each with
``IF NOT EXISTS`` so a database ``create_all`` built from the models is untouched.

Revision ID: prd265_billing_in_studio
Revises: prd265_billing
Create Date: 2026-10-23
"""
from alembic import op

revision = "prd265_billing_in_studio"
down_revision = "prd265_billing"
branch_labels = None
depends_on = None

COLUMNS = (
    ("schedule_id", "VARCHAR(64)"),
    ("scheduled_plan", "VARCHAR(32)"),
    ("scheduled_at", "TIMESTAMPTZ"),
    ("business_offer_until", "TIMESTAMPTZ"),
)


def upgrade() -> None:
    for name, kind in COLUMNS:
        op.execute(f"ALTER TABLE billing_accounts ADD COLUMN IF NOT EXISTS {name} {kind}")


def downgrade() -> None:
    for name, _ in COLUMNS:
        op.execute(f"ALTER TABLE billing_accounts DROP COLUMN IF EXISTS {name}")
