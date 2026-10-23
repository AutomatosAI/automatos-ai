"""PRD-265 W2 — billing accounts and the payment provider's webhook events.

- ``billing_accounts``: one row per workspace with a provider customer (its
  subscription, the paid period whose credits were granted, and the provider
  account it lives in).
- ``billing_events``: every verified webhook, unique per ``(provider, event_id)``,
  stored before it is processed; ``payment_ref`` is indexed so a refund finds the
  grant its payment made.

New tables only: nothing existing is altered, so the revision is safe on a live
database. ``IF NOT EXISTS`` keeps it a no-op on a database that ``create_all``
built from the models (``core.models.billing``).

Revision ID: prd265_billing
Revises: prd266_merge_heads
Create Date: 2026-10-23
"""
from alembic import op

revision = "prd265_billing"
down_revision = "prd266_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS billing_accounts (
            workspace_id UUID PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
            provider VARCHAR(16) NOT NULL DEFAULT 'stripe',
            provider_account VARCHAR(64),
            customer_id VARCHAR(64) NOT NULL,
            subscription_id VARCHAR(64),
            subscription_status VARCHAR(24),
            plan VARCHAR(32),
            current_period_end TIMESTAMPTZ,
            cancel_at_period_end BOOLEAN NOT NULL DEFAULT false,
            credited_period_end TIMESTAMPTZ,
            period_credits INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT billing_accounts_customer_id_key UNIQUE (customer_id)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS billing_events (
            id BIGSERIAL PRIMARY KEY,
            provider VARCHAR(16) NOT NULL,
            event_id VARCHAR(255) NOT NULL,
            event_type VARCHAR(64) NOT NULL,
            livemode BOOLEAN,
            status VARCHAR(16) NOT NULL DEFAULT 'received',
            workspace_id UUID REFERENCES workspaces(id) ON DELETE SET NULL,
            payment_ref VARCHAR(255),
            grant_source VARCHAR(300),
            amount_cents INTEGER,
            note TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            payload JSONB NOT NULL,
            received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            processed_at TIMESTAMPTZ,
            CONSTRAINT uq_billing_events_provider_event UNIQUE (provider, event_id),
            CONSTRAINT ck_billing_events_status
                CHECK (status IN ('received', 'processed', 'ignored', 'failed', 'needs_review'))
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_billing_events_payment_ref "
        "ON billing_events (payment_ref) WHERE payment_ref IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS billing_events")
    op.execute("DROP TABLE IF EXISTS billing_accounts")
