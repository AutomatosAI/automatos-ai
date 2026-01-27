
"""add_ownership_columns_manual

Revision ID: c551b7ae7bec
Revises: c11bfd9ba27c
Create Date: 2026-01-27 09:44:56.541672

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c551b7ae7bec'
down_revision = '102088374664'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agents
    op.add_column('agents', sa.Column('created_by_user_id', sa.Integer(), nullable=True))
    op.add_column('agents', sa.Column('is_shared', sa.Boolean(), server_default='true', nullable=True))
    op.create_foreign_key('fk_agents_created_by_user', 'agents', 'users', ['created_by_user_id'], ['id'])

    # Workflows
    op.add_column('workflows', sa.Column('created_by_user_id', sa.Integer(), nullable=True))
    op.add_column('workflows', sa.Column('is_shared', sa.Boolean(), server_default='true', nullable=True))
    op.create_foreign_key('fk_workflows_created_by_user', 'workflows', 'users', ['created_by_user_id'], ['id'])

    # Documents
    op.add_column('documents', sa.Column('created_by_user_id', sa.Integer(), nullable=True))
    op.add_column('documents', sa.Column('is_shared', sa.Boolean(), server_default='true', nullable=True))
    op.create_foreign_key('fk_documents_created_by_user', 'documents', 'users', ['created_by_user_id'], ['id'])

    # External Knowledge
    op.add_column('external_knowledge', sa.Column('created_by_user_id', sa.Integer(), nullable=True))
    op.add_column('external_knowledge', sa.Column('is_shared', sa.Boolean(), server_default='true', nullable=True))
    op.create_foreign_key('fk_ek_created_by_user', 'external_knowledge', 'users', ['created_by_user_id'], ['id'])


def downgrade() -> None:
    op.drop_constraint('fk_ek_created_by_user', 'external_knowledge', type_='foreignkey')
    op.drop_column('external_knowledge', 'is_shared')
    op.drop_column('external_knowledge', 'created_by_user_id')

    op.drop_constraint('fk_documents_created_by_user', 'documents', type_='foreignkey')
    op.drop_column('documents', 'is_shared')
    op.drop_column('documents', 'created_by_user_id')

    op.drop_constraint('fk_workflows_created_by_user', 'workflows', type_='foreignkey')
    op.drop_column('workflows', 'is_shared')
    op.drop_column('workflows', 'created_by_user_id')

    op.drop_constraint('fk_agents_created_by_user', 'agents', type_='foreignkey')
    op.drop_column('agents', 'is_shared')
    op.drop_column('agents', 'created_by_user_id')
