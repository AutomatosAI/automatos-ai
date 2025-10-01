"""Add document_usage table for Context Engineering analytics

Revision ID: 20250930_tracking
Revises: 208275450a15
Create Date: 2025-09-30 12:00:00

This migration creates the document_usage table that tracks all document-related events
for real-time analytics in the Context Engineering page.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20250930_tracking'
down_revision = '208275450a15'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create document_usage table
    op.create_table('document_usage',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=True),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('results_count', sa.Integer(), server_default='0', nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default="'{}'::jsonb", nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name='document_usage_document_id_fkey', ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name='document_usage_pkey')
    )
    
    # Create indexes for performance
    op.create_index('idx_document_usage_event_type', 'document_usage', ['event_type'], unique=False)
    op.create_index('idx_document_usage_timestamp', 'document_usage', [sa.text('timestamp DESC')], unique=False)
    op.create_index('idx_document_usage_event_time', 'document_usage', ['event_type', sa.text('timestamp DESC')], unique=False)
    
    # Conditional index for non-null document_ids
    op.execute("""
        CREATE INDEX idx_document_usage_document 
        ON document_usage(document_id) 
        WHERE document_id IS NOT NULL
    """)
    
    # GIN index for JSONB metadata queries
    op.create_index('idx_document_usage_metadata', 'document_usage', ['metadata'], unique=False, postgresql_using='gin')
    
    # Add comments for documentation
    op.execute("COMMENT ON TABLE document_usage IS 'Tracks all document-related events for analytics and Context Engineering monitoring'")
    op.execute("COMMENT ON COLUMN document_usage.event_type IS 'Type of event: document_searched, rag_query, document_viewed, chunk_retrieved'")
    op.execute("COMMENT ON COLUMN document_usage.query IS 'Search or RAG query text for semantic analysis'")
    op.execute("COMMENT ON COLUMN document_usage.results_count IS 'Number of results returned (0 indicates no results)'")
    op.execute("COMMENT ON COLUMN document_usage.execution_time_ms IS 'Query execution time in milliseconds for performance tracking'")
    op.execute("COMMENT ON COLUMN document_usage.metadata IS 'JSON metadata: config_id, similarity_threshold, agent_id, etc.'")


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_document_usage_metadata', table_name='document_usage')
    op.drop_index('idx_document_usage_document', table_name='document_usage')
    op.drop_index('idx_document_usage_event_time', table_name='document_usage')
    op.drop_index('idx_document_usage_timestamp', table_name='document_usage')
    op.drop_index('idx_document_usage_event_type', table_name='document_usage')
    
    # Drop table
    op.drop_table('document_usage')

