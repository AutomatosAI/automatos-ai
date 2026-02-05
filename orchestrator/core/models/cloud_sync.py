"""
Cloud Document Sync ORM models (PRD-42).

Tables for tracking cloud storage sync configuration, cloud document metadata
(no file content stored), and sync job progress/results.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, String, Text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from core.database.base import Base


class CloudSyncConfig(Base):
    """Per-connection sync settings (one root folder per connected app)."""

    __tablename__ = "cloud_sync_config"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    connection_id = Column(
        Integer,
        ForeignKey("composio_connections.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    root_folder_path = Column(String(1000), nullable=False)
    sync_enabled = Column(Boolean, server_default="true")
    last_sync_at = Column(DateTime)
    sync_frequency_minutes = Column(Integer, server_default="30")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    connection = relationship("ComposioConnection", backref="sync_config", uselist=False)


class CloudDocument(Base):
    """Metadata-only reference to a cloud-stored file with S3 vector pointers."""

    __tablename__ = "cloud_documents"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    connection_id = Column(
        Integer,
        ForeignKey("composio_connections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,  # Nullable until document is processed
        index=True,
    )

    # Cloud storage identifiers
    app_name = Column(String(100), nullable=False, index=True)
    external_file_id = Column(String(255), nullable=False)
    file_name = Column(String(500), nullable=False)
    file_path = Column(String(1000))
    mime_type = Column(String(100))
    file_size = Column(BigInteger)

    # S3 Vectors references (no local file storage)
    s3_vector_bucket = Column(String(255), nullable=False)
    s3_vector_index = Column(String(255), nullable=False, server_default="documents-index")
    chunk_count = Column(Integer, server_default="0")

    # Sync tracking
    cloud_modified_at = Column(DateTime)
    last_synced_at = Column(DateTime)
    sync_status = Column(String(50), server_default="pending")  # pending|syncing|synced|error
    sync_error = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    connection = relationship("ComposioConnection", backref="cloud_documents")


class CloudSyncJob(Base):
    """Tracks individual sync job runs with progress and error details."""

    __tablename__ = "cloud_sync_jobs"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    connection_id = Column(
        Integer,
        ForeignKey("composio_connections.id", ondelete="CASCADE"),
        nullable=False,
    )

    status = Column(String(50), server_default="pending")  # pending|running|completed|failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Results
    files_synced = Column(Integer, server_default="0")
    files_skipped = Column(Integer, server_default="0")
    files_errored = Column(Integer, server_default="0")
    total_chunks_created = Column(Integer, server_default="0")
    error_message = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    connection = relationship("ComposioConnection", backref="sync_jobs")
