"""Add semantic routing columns to agents table (PRD-64)

Revision ID: 20260224_semantic_routing
Revises: 20260218_nl2sql_training
Create Date: 2026-02-24

Adds:
- semantic_embedding JSONB column for pre-computed 2048-dim embedding vectors
- semantic_text_hash VARCHAR(64) for SHA-256 change detection (skip re-embed on no-op)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260224_semantic_routing'
down_revision = '20260218_nl2sql_training'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('agents', sa.Column('semantic_embedding', postgresql.JSONB(), nullable=True))
    op.add_column('agents', sa.Column('semantic_text_hash', sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column('agents', 'semantic_text_hash')
    op.drop_column('agents', 'semantic_embedding')
