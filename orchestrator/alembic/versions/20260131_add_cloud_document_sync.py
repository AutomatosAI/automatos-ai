"""Add cloud document sync tables for S3 Vectors integration

Revision ID: 20260131_cloud_sync
Revises: 20260129_merge
Create Date: 2026-01-31

PRD-42: Cloud Document Sync with S3 Vectors
Creates tables for cloud storage sync config, cloud document metadata,
and sync job tracking. Also extends composio_connections with sync fields.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID

# revision identifiers, used by Alembic.
revision = '20260131_cloud_sync'
down_revision = '20260129_merge'
branch_labels = None
depends_on = None


def upgrade():
    # -------------------------------------------------------------------------
    # cloud_sync_config: per-connection sync settings
    # -------------------------------------------------------------------------
    op.create_table(
        'cloud_sync_config',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('connection_id', sa.Integer(),
                  sa.ForeignKey('composio_connections.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('root_folder_path', sa.String(1000), nullable=False),
        sa.Column('sync_enabled', sa.Boolean(), server_default='true'),
        sa.Column('last_sync_at', sa.DateTime()),
        sa.Column('sync_frequency_minutes', sa.Integer(), server_default='30'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_unique_constraint(
        'uq_cloud_sync_config_connection', 'cloud_sync_config', ['connection_id']
    )

    # -------------------------------------------------------------------------
    # cloud_documents: metadata-only references to cloud files + vector storage
    # -------------------------------------------------------------------------
    op.create_table(
        'cloud_documents',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', PGUUID(as_uuid=True),
                  sa.ForeignKey('workspaces.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('connection_id', sa.Integer(),
                  sa.ForeignKey('composio_connections.id', ondelete='CASCADE'),
                  nullable=False),
        # Cloud storage identifiers
        sa.Column('app_name', sa.String(100), nullable=False),
        sa.Column('external_file_id', sa.String(255), nullable=False),
        sa.Column('file_name', sa.String(500), nullable=False),
        sa.Column('file_path', sa.String(1000)),
        sa.Column('mime_type', sa.String(100)),
        sa.Column('file_size', sa.BigInteger()),
        # S3 Vectors references (no local file storage)
        sa.Column('s3_vector_bucket', sa.String(255), nullable=False),
        sa.Column('s3_vector_index', sa.String(255), nullable=False,
                  server_default='documents-index'),
        sa.Column('chunk_count', sa.Integer(), server_default='0'),
        # Sync tracking
        sa.Column('cloud_modified_at', sa.DateTime()),
        sa.Column('last_synced_at', sa.DateTime()),
        sa.Column('sync_status', sa.String(50), server_default='pending'),
        sa.Column('sync_error', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_unique_constraint(
        'uq_cloud_documents_connection_file',
        'cloud_documents',
        ['connection_id', 'external_file_id'],
    )
    op.create_index('idx_cloud_documents_workspace', 'cloud_documents', ['workspace_id'])
    op.create_index('idx_cloud_documents_connection', 'cloud_documents', ['connection_id'])
    op.create_index('idx_cloud_documents_app', 'cloud_documents', ['app_name'])
    op.create_index('idx_cloud_documents_sync_status', 'cloud_documents', ['sync_status'])
    op.create_index('idx_cloud_documents_workspace_status', 'cloud_documents',
                    ['workspace_id', 'sync_status'])
    op.create_index('idx_cloud_documents_modified', 'cloud_documents',
                    ['connection_id', 'cloud_modified_at'])

    # -------------------------------------------------------------------------
    # cloud_sync_jobs: sync job tracking for progress/results
    # -------------------------------------------------------------------------
    op.create_table(
        'cloud_sync_jobs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', PGUUID(as_uuid=True),
                  sa.ForeignKey('workspaces.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('connection_id', sa.Integer(),
                  sa.ForeignKey('composio_connections.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('started_at', sa.DateTime()),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('files_synced', sa.Integer(), server_default='0'),
        sa.Column('files_skipped', sa.Integer(), server_default='0'),
        sa.Column('files_errored', sa.Integer(), server_default='0'),
        sa.Column('total_chunks_created', sa.Integer(), server_default='0'),
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('idx_cloud_sync_jobs_workspace_status', 'cloud_sync_jobs',
                    ['workspace_id', 'status'])

    # -------------------------------------------------------------------------
    # Extend composio_connections with sync metadata
    # -------------------------------------------------------------------------
    op.add_column('composio_connections',
                  sa.Column('sync_enabled', sa.Boolean(), server_default='false'))
    op.add_column('composio_connections',
                  sa.Column('total_documents_synced', sa.Integer(), server_default='0'))
    op.add_column('composio_connections',
                  sa.Column('last_successful_sync', sa.DateTime()))


def downgrade():
    op.drop_column('composio_connections', 'last_successful_sync')
    op.drop_column('composio_connections', 'total_documents_synced')
    op.drop_column('composio_connections', 'sync_enabled')
    op.drop_table('cloud_sync_jobs')
    op.drop_table('cloud_documents')
    op.drop_table('cloud_sync_config')
