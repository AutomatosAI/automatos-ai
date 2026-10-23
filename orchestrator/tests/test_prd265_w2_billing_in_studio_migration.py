"""PRD-265 W2 (payments in Studio) — the billing_accounts columns arrive by their own revision.

``prd265_billing`` already ran in production with ``billing_accounts``, so the four
columns the in-Studio flows need (a scheduled downgrade, a Business offer) come in
``prd265_billing_in_studio`` as nullable ``ADD COLUMN IF NOT EXISTS`` — never by
editing the revision that ran. The model and the migration name the same columns.
"""
from __future__ import annotations

from pathlib import Path

VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"
NEW_COLUMNS = ("schedule_id", "scheduled_plan", "scheduled_at", "business_offer_until")


def test_the_new_columns_come_in_their_own_revision():
    src = (VERSIONS / "prd265_billing_in_studio.py").read_text()
    assert 'down_revision = "prd265_billing"' in src
    assert "ADD COLUMN IF NOT EXISTS" in src and "DROP COLUMN IF EXISTS" in src
    for column in NEW_COLUMNS:
        assert f'"{column}"' in src, column


def test_the_revision_that_ran_is_left_as_it_ran():
    ran = (VERSIONS / "prd265_billing.py").read_text()
    for column in NEW_COLUMNS:
        assert column not in ran, column


def test_the_model_declares_the_same_columns_nullable():
    from core.models.billing import BillingAccount

    columns = BillingAccount.__table__.columns
    for column in NEW_COLUMNS:
        assert column in columns and columns[column].nullable, column
