"""PRD-72: Fix Composio usage trigger to UPSERT agent_app_features

The original trigger only UPDATEs existing rows but never INSERTs new ones.
This means tool usage is never tracked if the feature row doesn't already exist.

Revision ID: prd72_fix_composio_trigger
Revises: None
Create Date: 2026-03-09
"""
from alembic import op

revision = "prd72_fix_composio_trigger"
down_revision = None  # standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION increment_feature_usage()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.status = 'success' THEN
                INSERT INTO agent_app_features (agent_id, app_name, action_name, enabled, usage_count, last_used_at, created_at, updated_at)
                VALUES (NEW.agent_id, NEW.app_name, NEW.action_name, TRUE, 1, NEW.executed_at, NOW(), NOW())
                ON CONFLICT (agent_id, app_name, action_name) DO UPDATE
                SET
                    usage_count = agent_app_features.usage_count + 1,
                    last_used_at = NEW.executed_at,
                    updated_at = NOW();
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    # Restore original trigger function (UPDATE only)
    op.execute("""
        CREATE OR REPLACE FUNCTION increment_feature_usage()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.status = 'success' THEN
                UPDATE agent_app_features
                SET
                    usage_count = usage_count + 1,
                    last_used_at = NEW.executed_at
                WHERE
                    agent_id = NEW.agent_id
                    AND app_name = NEW.app_name
                    AND action_name = NEW.action_name;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
