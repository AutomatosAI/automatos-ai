"""PRD-74 Phase 2: Add voice_profiles table and agent voice assignment

Revision ID: prd74_voice_profiles
Revises: prd73_infra_alerts
Create Date: 2026-03-09
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "prd74_voice_profiles"
down_revision = None  # standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS voice_profiles (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id    UUID NOT NULL REFERENCES workspaces(id),
            name            TEXT NOT NULL,
            provider        TEXT NOT NULL DEFAULT 'kokoro',
            voice_id        TEXT NOT NULL,
            reference_audio TEXT,
            settings        JSONB DEFAULT '{}',
            is_default      BOOLEAN DEFAULT FALSE,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_voice_profiles_workspace
            ON voice_profiles(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_voice_profiles_provider
            ON voice_profiles(provider);

        -- Per-agent voice assignment
        ALTER TABLE agents
            ADD COLUMN IF NOT EXISTS voice_profile_id UUID REFERENCES voice_profiles(id);
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE agents DROP COLUMN IF EXISTS voice_profile_id;
        DROP TABLE IF EXISTS voice_profiles;
    """)
