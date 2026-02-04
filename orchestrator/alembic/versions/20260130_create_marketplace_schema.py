"""Create marketplace schema for community marketplace feature

Revision ID: 20260130_marketplace_schema
Revises: 20260129_merge_heads
Create Date: 2026-01-30

This migration creates the foundational tables for the Community Marketplace feature:
- marketplace_items: stores all marketplace items (agents, recipes, skills, LLMs, tools)
- marketplace_installs: tracks what users have installed from the marketplace
- marketplace_submissions: approval queue for user-submitted marketplace items
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import func

# revision identifiers, used by Alembic.
revision = '20260130_marketplace_schema'
down_revision = '20260129_merge'
branch_labels = None
depends_on = None


def upgrade():
    """Create marketplace tables."""

    # Create marketplace_items table
    op.create_table(
        'marketplace_items',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('type', sa.String(50), nullable=False, index=True),  # 'agent', 'recipe', 'skill', 'llm', 'tool'
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('creator_name', sa.String(255), nullable=False),
        sa.Column('creator_user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('icon', sa.String(500), nullable=True),  # Emoji or icon URL
        sa.Column('category', sa.String(100), nullable=True, index=True),
        sa.Column('tags', postgresql.JSONB, server_default='[]'),
        sa.Column('install_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('is_featured', sa.Boolean(), server_default='false', nullable=False, index=True),
        sa.Column('is_approved', sa.Boolean(), server_default='false', nullable=False, index=True),
        sa.Column('version', sa.String(50), server_default='1.0.0', nullable=False),
        sa.Column('metadata', postgresql.JSONB, server_default='{}'),  # Type-specific data
        sa.Column('created_at', sa.DateTime(), server_default=func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=func.now(), onupdate=func.now(), nullable=False),
        sa.CheckConstraint("type IN ('agent', 'recipe', 'skill', 'llm', 'tool')", name='valid_marketplace_type')
    )

    # Create indexes for marketplace_items
    op.create_index('idx_marketplace_items_type_approved', 'marketplace_items', ['type', 'is_approved'])
    op.create_index('idx_marketplace_items_featured', 'marketplace_items', ['is_featured', 'install_count'])

    # Create marketplace_installs table
    op.create_table(
        'marketplace_installs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('item_id', sa.Integer(), sa.ForeignKey('marketplace_items.id'), nullable=False, index=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('installed_at', sa.DateTime(), server_default=func.now(), nullable=False),
        sa.Column('cloned_resource_id', sa.Integer(), nullable=True),  # ID of the cloned agent/recipe/etc
        sa.Column('cloned_resource_type', sa.String(50), nullable=True)  # 'agent', 'recipe', etc
    )

    # Create composite index for user's installed items
    op.create_index('idx_marketplace_installs_user_item', 'marketplace_installs', ['user_id', 'item_id'])

    # Create marketplace_submissions table
    op.create_table(
        'marketplace_submissions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('item_id', sa.Integer(), sa.ForeignKey('marketplace_items.id'), nullable=False, index=True),
        sa.Column('submitted_by', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('status', sa.String(50), server_default='pending', nullable=False, index=True),  # 'pending', 'approved', 'rejected'
        sa.Column('reviewed_by', sa.Integer(), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('submitted_at', sa.DateTime(), server_default=func.now(), nullable=False),
        sa.CheckConstraint("status IN ('pending', 'approved', 'rejected')", name='valid_submission_status')
    )

    # Create index for pending submissions
    op.create_index('idx_marketplace_submissions_status', 'marketplace_submissions', ['status', 'submitted_at'])


def downgrade():
    """Drop marketplace tables."""
    op.drop_index('idx_marketplace_submissions_status', table_name='marketplace_submissions')
    op.drop_table('marketplace_submissions')

    op.drop_index('idx_marketplace_installs_user_item', table_name='marketplace_installs')
    op.drop_table('marketplace_installs')

    op.drop_index('idx_marketplace_items_featured', table_name='marketplace_items')
    op.drop_index('idx_marketplace_items_type_approved', table_name='marketplace_items')
    op.drop_table('marketplace_items')
