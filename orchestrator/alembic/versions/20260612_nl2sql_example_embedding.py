"""Add embedding column to nl2sql_training_examples (PRD-160 S3)

Revision ID: 20260612_nl2sql_example_embedding
Revises: prd142_wave5_drop_dead_tables
Create Date: 2026-06-12

PRD-160 S3 persists the question embedding for each training pair (it was
computed then discarded) so verified pairs can be retrieved by cosine
similarity for few-shot generation. Stored as JSONB to match the platform's
semantic-embedding convention (PRD-64 agents.semantic_embedding is JSONB).
Idempotent — safe to apply regardless of migration ordering.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260612_nl2sql_example_embedding'
down_revision = 'prd142_wave5_drop_dead_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'nl2sql_training_examples'
                  AND column_name = 'embedding'
            ) THEN
                ALTER TABLE nl2sql_training_examples ADD COLUMN embedding JSONB;
            END IF;
        END
        $$;
    """))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'nl2sql_training_examples'
                  AND column_name = 'embedding'
            ) THEN
                ALTER TABLE nl2sql_training_examples DROP COLUMN embedding;
            END IF;
        END
        $$;
    """))
