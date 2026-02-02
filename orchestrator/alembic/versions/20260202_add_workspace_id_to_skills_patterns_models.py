"""add workspace_id to skills, patterns, and llm_models

Revision ID: 20260202_workspace_isolation
Revises:
Create Date: 2026-02-02 15:45:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260202_workspace_isolation'
down_revision = '20260201_recipe_executions'
branch_labels = None
depends_on = None


def upgrade():
    """Add workspace_id to skills, patterns, and llm_models for multi-tenant isolation."""

    # Add workspace_id to skills table
    op.add_column('skills', sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_skills_workspace_id', 'skills', 'workspaces', ['workspace_id'], ['id'])
    op.create_index('idx_skills_workspace', 'skills', ['workspace_id'])

    # Add workspace_id to patterns table
    op.add_column('patterns', sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_patterns_workspace_id', 'patterns', 'workspaces', ['workspace_id'], ['id'])
    op.create_index('idx_patterns_workspace', 'patterns', ['workspace_id'])

    # Add workspace_id to llm_models table
    op.add_column('llm_models', sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_llm_models_workspace_id', 'llm_models', 'workspaces', ['workspace_id'], ['id'])
    op.create_index('idx_llm_models_workspace', 'llm_models', ['workspace_id'])

    # Migrate existing data: Set workspace_id to first workspace (or leave NULL for system-wide items)
    # For skills and patterns, assign to first workspace
    op.execute("""
        UPDATE skills
        SET workspace_id = (SELECT id FROM workspaces ORDER BY created_at LIMIT 1)
        WHERE workspace_id IS NULL;
    """)

    op.execute("""
        UPDATE patterns
        SET workspace_id = (SELECT id FROM workspaces ORDER BY created_at LIMIT 1)
        WHERE workspace_id IS NULL;
    """)

    # For llm_models, keep NULL for system-wide models (can be shared across workspaces)
    # Only set workspace_id for custom user models if needed

    # Now make workspace_id NOT NULL for skills and patterns (required for isolation)
    op.alter_column('skills', 'workspace_id', nullable=False)
    op.alter_column('patterns', 'workspace_id', nullable=False)
    # Keep llm_models.workspace_id nullable (system models can be shared)


def downgrade():
    """Remove workspace_id columns."""

    # Skills
    op.drop_index('idx_skills_workspace', table_name='skills')
    op.drop_constraint('fk_skills_workspace_id', 'skills', type_='foreignkey')
    op.drop_column('skills', 'workspace_id')

    # Patterns
    op.drop_index('idx_patterns_workspace', table_name='patterns')
    op.drop_constraint('fk_patterns_workspace_id', 'patterns', type_='foreignkey')
    op.drop_column('patterns', 'workspace_id')

    # LLM Models
    op.drop_index('idx_llm_models_workspace', table_name='llm_models')
    op.drop_constraint('fk_llm_models_workspace_id', 'llm_models', type_='foreignkey')
    op.drop_column('llm_models', 'workspace_id')
