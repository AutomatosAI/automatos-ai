
"""
Alembic Migration: PRD-22 Anthropic Skills Integration
======================================================

This migration adds the complete database schema for Git-backed skill loading
with progressive disclosure support.

Revision ID: 005_anthropic_skills_integration
Revises: 004_xxx
Create Date: 2025-10-29
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers
revision = '005_anthropic_skills_integration'
down_revision = '004_tool_registry'  # Update with your current migration
branch_labels = None
depends_on = None


def upgrade():
    """
    Apply PRD-22 database schema changes.
    
    Changes:
    1. Add new columns to skills table for Git-backed skills
    2. Create skill_files table for progressive disclosure
    3. Create skill_sources table for Git repository tracking
    4. Create skill_versions table for version history
    5. Create skill_audit_log table for security auditing
    6. Add indexes for performance optimization
    """
    
    # =========================================================================
    # 1. Enhance skills table with new columns
    # =========================================================================
    print("Enhancing skills table with PRD-22 columns...")
    
    # Add prompt_template column for core skill content (Level 2)
    op.add_column('skills', sa.Column('prompt_template', sa.Text(), nullable=True))
    
    # Add skill_version for version tracking
    op.add_column('skills', sa.Column('skill_version', sa.String(20), nullable=True))
    
    # Add skill_source to track origin (builtin-seeds, anthropic-official, custom, etc.)
    op.add_column('skills', sa.Column('skill_source', sa.String(50), nullable=True))
    
    # Add Git repository information
    op.add_column('skills', sa.Column('git_repo_url', sa.Text(), nullable=True))
    op.add_column('skills', sa.Column('git_commit_sha', sa.String(40), nullable=True))
    op.add_column('skills', sa.Column('git_branch', sa.String(100), nullable=True))
    
    # Add filesystem_path for hybrid storage approach
    op.add_column('skills', sa.Column('filesystem_path', sa.Text(), nullable=True))
    
    # Add tags array for better categorization
    op.add_column('skills', sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Add metadata JSON for flexible skill properties
    op.add_column('skills', sa.Column('skill_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Add last_sync_at for tracking Git updates
    op.add_column('skills', sa.Column('last_sync_at', sa.DateTime(timezone=True), nullable=True))
    
    # Set defaults for existing rows
    op.execute("""
        UPDATE skills 
        SET skill_source = 'builtin-seeds',
            skill_version = '1.0.0',
            tags = '[]'::jsonb,
            skill_metadata = '{}'::jsonb
        WHERE skill_source IS NULL
    """)
    
    # =========================================================================
    # 2. Create skill_files table for progressive disclosure
    # =========================================================================
    print("Creating skill_files table...")
    
    op.create_table(
        'skill_files',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('skill_id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(500), nullable=False, comment='Relative path within skill directory'),
        sa.Column('file_type', sa.String(50), nullable=False, comment='core, resource, script, example'),
        sa.Column('content_summary', sa.Text(), nullable=True, comment='Brief summary for indexing'),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('estimated_tokens', sa.Integer(), nullable=True, comment='Estimated token count for this file'),
        sa.Column('load_level', sa.Integer(), nullable=False, default=3, comment='1=metadata, 2=core, 3=resource'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['skill_id'], ['skills.id'], ondelete='CASCADE'),
        sa.Index('idx_skill_files_skill_id', 'skill_id'),
        sa.Index('idx_skill_files_type', 'file_type'),
        sa.Index('idx_skill_files_level', 'load_level'),
    )
    
    # =========================================================================
    # 3. Create skill_sources table for Git repository tracking
    # =========================================================================
    print("Creating skill_sources table...")
    
    op.create_table(
        'skill_sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_name', sa.String(100), nullable=False, unique=True, comment='Unique identifier for this source'),
        sa.Column('source_type', sa.String(50), nullable=False, comment='git, upload, seed, marketplace'),
        sa.Column('git_url', sa.Text(), nullable=True, comment='Git repository URL'),
        sa.Column('git_branch', sa.String(100), nullable=True, default='main'),
        sa.Column('git_commit_sha', sa.String(40), nullable=True, comment='Current commit SHA'),
        sa.Column('local_cache_path', sa.Text(), nullable=False, comment='Path in ~/.automatos/skills/'),
        sa.Column('auto_update', sa.Boolean(), default=False, comment='Auto-update from Git on schedule'),
        sa.Column('update_frequency_hours', sa.Integer(), nullable=True, default=24),
        sa.Column('skills_discovered', sa.Integer(), default=0, comment='Number of skills found in this source'),
        sa.Column('status', sa.String(50), default='active', comment='active, error, syncing, inactive'),
        sa.Column('last_sync_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_sync_status', sa.String(50), nullable=True, comment='success, error, partial'),
        sa.Column('last_sync_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_skill_sources_type', 'source_type'),
        sa.Index('idx_skill_sources_status', 'status'),
        sa.Index('idx_skill_sources_name', 'source_name'),
    )
    
    # Insert default skill sources
    op.execute("""
        INSERT INTO skill_sources (source_name, source_type, local_cache_path, skills_discovered, status)
        VALUES 
            ('builtin-seeds', 'seed', '~/.automatos/skills/builtin-seeds/', 
             (SELECT COUNT(*) FROM skills WHERE skill_source = 'builtin-seeds' OR skill_source IS NULL), 
             'active'),
            ('anthropic-official', 'git', '~/.automatos/skills/anthropic-official/', 0, 'inactive')
    """)
    
    # =========================================================================
    # 4. Create skill_versions table for version history
    # =========================================================================
    print("Creating skill_versions table...")
    
    op.create_table(
        'skill_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('skill_id', sa.Integer(), nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('git_commit_sha', sa.String(40), nullable=True),
        sa.Column('changelog', sa.Text(), nullable=True),
        sa.Column('is_current', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['skill_id'], ['skills.id'], ondelete='CASCADE'),
        sa.Index('idx_skill_versions_skill_id', 'skill_id'),
        sa.Index('idx_skill_versions_current', 'is_current'),
        sa.UniqueConstraint('skill_id', 'version', name='uq_skill_version'),
    )
    
    # =========================================================================
    # 5. Create skill_audit_log table for security and tracking
    # =========================================================================
    print("Creating skill_audit_log table...")
    
    op.create_table(
        'skill_audit_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('skill_id', sa.Integer(), nullable=True),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(50), nullable=False, comment='create, update, delete, load, execute, sync'),
        sa.Column('action_details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('ip_address', sa.String(50), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, comment='success, error, warning'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['skill_id'], ['skills.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['source_id'], ['skill_sources.id'], ondelete='SET NULL'),
        sa.Index('idx_skill_audit_action', 'action'),
        sa.Index('idx_skill_audit_created', 'created_at'),
        sa.Index('idx_skill_audit_skill', 'skill_id'),
        sa.Index('idx_skill_audit_source', 'source_id'),
    )
    
    # =========================================================================
    # 6. Add performance indexes
    # =========================================================================
    print("Creating performance indexes...")
    
    # Composite indexes for common queries
    op.create_index('idx_skills_source_active', 'skills', ['skill_source', 'is_active'])
    op.create_index('idx_skills_category_active', 'skills', ['category', 'is_active'])
    
    # GIN index for tags array searching (PostgreSQL specific)
    op.execute("""
        CREATE INDEX idx_skills_tags_gin ON skills USING GIN (tags jsonb_path_ops)
    """)
    
    print("PRD-22 migration completed successfully!")


def downgrade():
    """
    Rollback PRD-22 database schema changes.
    
    Warning: This will remove all Git-backed skill data!
    """
    
    print("Rolling back PRD-22 migration...")
    
    # Drop indexes
    op.drop_index('idx_skills_tags_gin', table_name='skills')
    op.drop_index('idx_skills_category_active', table_name='skills')
    op.drop_index('idx_skills_source_active', table_name='skills')
    
    # Drop tables in reverse order
    op.drop_table('skill_audit_log')
    op.drop_table('skill_versions')
    op.drop_table('skill_sources')
    op.drop_table('skill_files')
    
    # Remove columns from skills table
    op.drop_column('skills', 'last_sync_at')
    op.drop_column('skills', 'skill_metadata')
    op.drop_column('skills', 'tags')
    op.drop_column('skills', 'filesystem_path')
    op.drop_column('skills', 'git_branch')
    op.drop_column('skills', 'git_commit_sha')
    op.drop_column('skills', 'git_repo_url')
    op.drop_column('skills', 'skill_source')
    op.drop_column('skills', 'skill_version')
    op.drop_column('skills', 'prompt_template')
    
    print("PRD-22 rollback completed!")
