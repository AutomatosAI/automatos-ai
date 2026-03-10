"""PRD-73: Add infrastructure_alerts table for monitoring stack

Revision ID: prd73_infra_alerts
Revises: prd71_unified_skills
Create Date: 2026-03-09
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "prd73_infra_alerts"
down_revision = None  # standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS infrastructure_alerts (
            id SERIAL PRIMARY KEY,
            fingerprint VARCHAR(64) NOT NULL,
            alertname VARCHAR(255) NOT NULL,
            severity VARCHAR(32) NOT NULL DEFAULT 'unknown',
            status VARCHAR(32) NOT NULL DEFAULT 'firing',
            service VARCHAR(255),
            instance VARCHAR(255),
            labels JSONB NOT NULL DEFAULT '{}',
            annotations JSONB NOT NULL DEFAULT '{}',
            starts_at TIMESTAMPTZ,
            ends_at TIMESTAMPTZ,
            generator_url TEXT,
            receiver VARCHAR(255),
            raw_payload JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            resolved_at TIMESTAMPTZ,
            UNIQUE(fingerprint, starts_at)
        );

        CREATE INDEX IF NOT EXISTS idx_infra_alerts_fingerprint
            ON infrastructure_alerts(fingerprint);
        CREATE INDEX IF NOT EXISTS idx_infra_alerts_status
            ON infrastructure_alerts(status);
        CREATE INDEX IF NOT EXISTS idx_infra_alerts_severity
            ON infrastructure_alerts(severity);
        CREATE INDEX IF NOT EXISTS idx_infra_alerts_alertname
            ON infrastructure_alerts(alertname);
        CREATE INDEX IF NOT EXISTS idx_infra_alerts_created
            ON infrastructure_alerts(created_at DESC);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS infrastructure_alerts;")
