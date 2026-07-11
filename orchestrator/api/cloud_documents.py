"""
Cloud Documents API (PRD-42)
============================

Endpoints for cloud storage connection management, folder navigation,
document sync, disconnect, and RAG queries against S3 Vectors.
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext
from core.models.cloud_sync import CloudDocument, CloudSyncConfig, CloudSyncJob
from core.models.composio import ComposioConnection, ComposioEntity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cloud-documents", tags=["cloud-documents"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CloudConnectionResponse(BaseModel):
    id: int
    app_name: str
    status: str
    connected_at: Optional[datetime] = None
    sync_enabled: bool = False
    total_documents_synced: int = 0
    last_successful_sync: Optional[datetime] = None
    root_folder_path: Optional[str] = None


class FolderResponse(BaseModel):
    name: str
    path: str
    has_children: bool = True


class CloudFileResponse(BaseModel):
    external_file_id: str
    name: str
    path: str = ""
    mime_type: str = ""
    size: int = 0
    modified_at: Optional[str] = None
    is_synced: bool = False
    sync_status: str = "pending"
    chunk_count: int = 0
    last_synced_at: Optional[str] = None


class SelectFolderRequest(BaseModel):
    root_folder_path: str = Field(..., min_length=1)
    default_team_access: Optional[List[str]] = Field(
        default=None,
        description="PRD-158: teams applied to docs synced from this connection (normalized server-side)",
    )


class SelectFolderResponse(BaseModel):
    status: str = "success"
    root_folder_path: str
    sync_enabled: bool = True


class SyncTriggerResponse(BaseModel):
    job_id: int
    status: str
    message: str


class SyncJobResponse(BaseModel):
    id: int
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    files_synced: int = 0
    files_skipped: int = 0
    files_errored: int = 0
    total_chunks_created: int = 0
    error_message: Optional[str] = None


class SyncStatusResponse(BaseModel):
    connection_id: int
    app_name: str
    sync_enabled: bool
    last_sync_at: Optional[datetime] = None
    total_documents: int = 0
    synced: int = 0
    pending: int = 0
    errors: int = 0
    total_chunks: int = 0


class DisconnectResponse(BaseModel):
    status: str
    vectors_deleted: bool
    documents_affected: int


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    min_similarity: float = Field(0.5, ge=0.0, le=1.0)
    max_tokens: int = Field(2000, ge=100, le=10000)
    diversity_factor: float = Field(0.3, ge=0.0, le=1.0)


class RAGChunkResponse(BaseModel):
    document_id: Optional[int] = None
    file_name: str = ""
    app_name: str = ""
    file_path: str = ""
    chunk_index: int = 0
    content: str = ""
    similarity_score: float = 0.0


class RAGSourceResponse(BaseModel):
    document_id: Optional[int] = None
    file_name: str = ""
    app_name: str = ""
    file_path: str = ""
    chunks_used: int = 0


class RAGQueryResponse(BaseModel):
    query: str
    context: str = ""
    chunks: List[RAGChunkResponse] = []
    sources: List[RAGSourceResponse] = []
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_entity_for_workspace(db: Session, workspace_id) -> Optional[ComposioEntity]:
    return db.query(ComposioEntity).filter(
        ComposioEntity.workspace_id == workspace_id
    ).first()


def _get_connections_for_workspace(
    db: Session, workspace_id, app_name: Optional[str] = None,
) -> List[ComposioConnection]:
    entity = _get_entity_for_workspace(db, workspace_id)
    if not entity:
        return []
    q = db.query(ComposioConnection).filter(
        ComposioConnection.entity_id == entity.id,
    )
    if app_name:
        q = q.filter(ComposioConnection.app_name == app_name.upper())
    return q.all()


def _get_connection_or_404(
    db: Session, connection_id: int, workspace_id,
) -> ComposioConnection:
    entity = _get_entity_for_workspace(db, workspace_id)
    if not entity:
        raise HTTPException(status_code=404, detail="No cloud connections found")
    conn = db.query(ComposioConnection).filter(
        ComposioConnection.id == connection_id,
        ComposioConnection.entity_id == entity.id,
    ).first()
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    return conn


# ===================================================================
# US-009: Connection & folder endpoints
# ===================================================================

@router.get("/connections", response_model=List[CloudConnectionResponse])
async def list_connections(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    List all connected cloud storage providers for this workspace.

    Auto-discovers file storage tools by checking Composio categories:
    - document & file management
    - storage
    - cloud storage
    - productivity

    Returns ONLY tools that can store/sync files (Google Drive, Dropbox, OneDrive, Box, etc.)
    """
    try:
        # Get ALL connections for this workspace
        all_connections = _get_connections_for_workspace(db, ctx.workspace_id)

        # File storage categories to filter by
        FILE_STORAGE_CATEGORIES = [
            'document & file management',
            'storage',
            'cloud storage',
            'file sharing',
            'productivity',  # Some file tools are tagged as productivity
        ]

        # Known file storage apps (fallback if category data is missing)
        KNOWN_FILE_STORAGE_APPS = [
            'GOOGLEDRIVE',
            'DROPBOX',
            'ONEDRIVE',
            'BOX',
            'SHAREPOINT',
            'ICLOUD',
            'MEGA',
            'PCLOUD',
            'SYNC',
            'NEXTCLOUD',
            'OWNCLOUD',
        ]

        results = []
        for conn in all_connections:
            # Only show ACTIVE connections (skip "added", "pending", etc.)
            if conn.status.lower() != 'active':
                continue

            # Check if this is a file storage tool
            is_file_storage = False

            # Method 1: Check by app name (most reliable)
            if conn.app_name.upper() in KNOWN_FILE_STORAGE_APPS:
                is_file_storage = True

            # Method 2: Check categories from Composio metadata
            # Note: Categories might be stored in a separate table or as JSON metadata
            # For now, we'll use the app name method as the primary filter

            if not is_file_storage:
                continue  # Skip non-file-storage tools

            # Check for sync config
            sync_cfg = db.query(CloudSyncConfig).filter(
                CloudSyncConfig.connection_id == conn.id
            ).first()

            # Don't treat "/" as a valid root folder selection - it means "entire drive" which we want to require explicit selection
            root_path = sync_cfg.root_folder_path if sync_cfg and sync_cfg.root_folder_path != "/" else None

            results.append(CloudConnectionResponse(
                id=conn.id,
                app_name=conn.app_name,
                status=conn.status,
                connected_at=conn.connected_at,
                sync_enabled=conn.sync_enabled or False,
                total_documents_synced=conn.total_documents_synced or 0,
                last_successful_sync=conn.last_successful_sync,
                root_folder_path=root_path,
            ))

        logger.info(f"Found {len(results)} file storage connections out of {len(all_connections)} total")
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing cloud connections: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/connections/{connection_id}/folders",
    response_model=List[FolderResponse],
)
async def list_folders(
    connection_id: int,
    path: str = Query("/", description="Folder path to list"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List folders at a given path in cloud storage (lazy-load for tree UI)."""
    try:
        _get_connection_or_404(db, connection_id, ctx.workspace_id)

        from modules.rag.services.cloud_sync_service import CloudSyncService
        service = CloudSyncService(db)
        folders = await service.list_folders(connection_id, path)

        return [FolderResponse(**f) for f in folders]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/connections/{connection_id}/files",
    response_model=List[CloudFileResponse],
)
async def list_files(
    connection_id: int,
    path: str = Query("/", description="Folder path to list files"),
    recursive: bool = Query(False),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List files at a given path with sync status from database."""
    try:
        _get_connection_or_404(db, connection_id, ctx.workspace_id)

        from modules.rag.services.cloud_sync_service import CloudSyncService
        service = CloudSyncService(db)
        files = await service.list_files(connection_id, path, recursive=recursive)

        return [CloudFileResponse(**f) for f in files]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/connections/{connection_id}/select-folder",
    response_model=SelectFolderResponse,

    dependencies=[Depends(require_workspace_permission("workspace:manage"))],
)
async def select_folder(
    connection_id: int,
    body: SelectFolderRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Set the root sync folder for a cloud storage connection."""
    try:
        conn = _get_connection_or_404(db, connection_id, ctx.workspace_id)

        # PRD-158 S2/Q5: normalize the connection's default team(s) once, here.
        from core.team_access import ensure_teams
        default_team_access = ensure_teams(db, ctx.workspace_id, body.default_team_access or [])

        # Upsert sync config
        sync_cfg = db.query(CloudSyncConfig).filter(
            CloudSyncConfig.connection_id == connection_id
        ).first()

        if sync_cfg:
            sync_cfg.root_folder_path = body.root_folder_path
            sync_cfg.sync_enabled = True
            sync_cfg.default_team_access = default_team_access
        else:
            sync_cfg = CloudSyncConfig(
                connection_id=connection_id,
                root_folder_path=body.root_folder_path,
                sync_enabled=True,
                default_team_access=default_team_access,
            )
            db.add(sync_cfg)

        conn.sync_enabled = True
        db.commit()

        return SelectFolderResponse(
            root_folder_path=body.root_folder_path,
            sync_enabled=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error selecting folder: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================================================
# US-010: Sync trigger & disconnect endpoints
# ===================================================================

@router.post(
    "/connections/{connection_id}/sync",
    response_model=SyncTriggerResponse,

    dependencies=[Depends(require_workspace_permission("workspace:manage"))],
)
async def trigger_sync(
    connection_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Manually trigger a sync for a cloud storage connection."""
    try:
        _get_connection_or_404(db, connection_id, ctx.workspace_id)

        # Verify a specific root folder has been selected (not "/" which means entire drive)
        sync_cfg = db.query(CloudSyncConfig).filter_by(connection_id=connection_id).first()
        if not sync_cfg or not sync_cfg.root_folder_path or sync_cfg.root_folder_path == "/":
            raise HTTPException(
                status_code=400,
                detail="Please select a specific folder to sync first. Navigate to the folder in the browser and click 'Set as Root Folder'."
            )

        from modules.rag.services.cloud_sync_service import CloudSyncService
        service = CloudSyncService(db)
        job = await service.sync_folder(connection_id, ctx.workspace_id)

        return SyncTriggerResponse(
            job_id=job.id,
            status=job.status,
            message=f"Sync job {job.status}. "
                    f"Synced: {job.files_synced}, "
                    f"Skipped: {job.files_skipped}, "
                    f"Errors: {job.files_errored}",
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Cloud document sync validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid sync request")
    except Exception as e:
        logger.error(f"Error triggering sync: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sync-jobs/{job_id}", response_model=SyncJobResponse)
async def get_sync_job(
    job_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get status and results of a sync job."""
    try:
        job = db.query(CloudSyncJob).filter(
            CloudSyncJob.id == job_id,
            CloudSyncJob.workspace_id == ctx.workspace_id,
        ).first()
        if not job:
            raise HTTPException(status_code=404, detail="Sync job not found")

        return SyncJobResponse(
            id=job.id,
            status=job.status,
            started_at=job.started_at,
            completed_at=job.completed_at,
            files_synced=job.files_synced or 0,
            files_skipped=job.files_skipped or 0,
            files_errored=job.files_errored or 0,
            total_chunks_created=job.total_chunks_created or 0,
            error_message=job.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sync job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/connections/{connection_id}/sync-status",
    response_model=SyncStatusResponse,
)
async def get_sync_status(
    connection_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get current sync summary for a connection."""
    try:
        conn = _get_connection_or_404(db, connection_id, ctx.workspace_id)

        sync_cfg = db.query(CloudSyncConfig).filter(
            CloudSyncConfig.connection_id == connection_id
        ).first()

        # Aggregate stats from cloud_documents
        total = db.query(CloudDocument).filter(
            CloudDocument.connection_id == connection_id
        ).count()
        synced = db.query(CloudDocument).filter(
            CloudDocument.connection_id == connection_id,
            CloudDocument.sync_status == "synced",
        ).count()
        pending = db.query(CloudDocument).filter(
            CloudDocument.connection_id == connection_id,
            CloudDocument.sync_status == "pending",
        ).count()
        errors = db.query(CloudDocument).filter(
            CloudDocument.connection_id == connection_id,
            CloudDocument.sync_status == "error",
        ).count()

        from sqlalchemy import func
        total_chunks = db.query(func.coalesce(func.sum(CloudDocument.chunk_count), 0)).filter(
            CloudDocument.connection_id == connection_id,
            CloudDocument.sync_status == "synced",
        ).scalar()

        return SyncStatusResponse(
            connection_id=connection_id,
            app_name=conn.app_name,
            sync_enabled=conn.sync_enabled or False,
            last_sync_at=sync_cfg.last_sync_at if sync_cfg else None,
            total_documents=total,
            synced=synced,
            pending=pending,
            errors=errors,
            total_chunks=total_chunks or 0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete(
    "/connections/{connection_id}",
    response_model=DisconnectResponse,

    dependencies=[Depends(require_workspace_permission("workspace:manage"))],
)
async def disconnect_connection(
    connection_id: int,
    delete_vectors: bool = Query(False, description="Delete all S3 vectors for this connection"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Disconnect a cloud storage provider. Optionally delete vectors."""
    try:
        conn = _get_connection_or_404(db, connection_id, ctx.workspace_id)

        documents_affected = db.query(CloudDocument).filter(
            CloudDocument.connection_id == connection_id
        ).count()

        if delete_vectors and documents_affected > 0:
            # Delete vectors from S3 Vectors backend
            from modules.search.vector_store import get_vector_store
            backend = get_vector_store(
                backend="s3_vectors",
                workspace_id=str(ctx.workspace_id)
            )
            backend.delete_all_for_connection(connection_id)

            # Remove cloud_documents records
            db.query(CloudDocument).filter(
                CloudDocument.connection_id == connection_id
            ).delete()

        # Mark connection as disconnected
        conn.status = "disconnected"
        conn.sync_enabled = False

        # Remove sync config
        db.query(CloudSyncConfig).filter(
            CloudSyncConfig.connection_id == connection_id
        ).delete()

        db.commit()

        return DisconnectResponse(
            status="disconnected",
            vectors_deleted=delete_vectors,
            documents_affected=documents_affected,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error disconnecting: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================================================
# US-012: Cloud RAG query endpoint
# ===================================================================

@router.post("/rag/query", response_model=RAGQueryResponse, dependencies=[Depends(require_workspace_permission("documents:read"))])
async def cloud_rag_query(
    body: RAGQueryRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Query cloud documents using RAG across all connected storage providers."""
    try:
        from modules.search.vector_store import get_vector_store

        # Using real S3 Vectors backend (eu-west-1)
        backend = get_vector_store(
            backend="s3_vectors",
            workspace_id=str(ctx.workspace_id),
        )

        # Generate query embedding
        from core.llm.embedding_manager import EmbeddingManager
        embedding_mgr = EmbeddingManager()
        query_embedding = await embedding_mgr.generate_embedding(body.query)

        # Search S3 Vectors
        results = backend.search(
            query_embedding=query_embedding,
            limit=body.top_k * 2,  # over-fetch for diversity filtering
            min_score=body.min_similarity,
        )

        # Build response
        chunks = []
        sources_map = {}  # file_name -> source info

        for r in results[:body.top_k]:
            file_name = r.get("file_name", "")
            app_name = r.get("source", "")
            file_path = r.get("file_path", "")

            # Look up document_id from cloud_documents if possible
            doc_id = None
            ext_id = r.get("external_file_id")
            if ext_id:
                cd = db.query(CloudDocument).filter(
                    CloudDocument.external_file_id == ext_id,
                    CloudDocument.workspace_id == ctx.workspace_id,
                ).first()
                if cd:
                    doc_id = cd.id

            chunks.append(RAGChunkResponse(
                document_id=doc_id,
                file_name=file_name,
                app_name=app_name,
                file_path=file_path,
                chunk_index=r.get("chunk_index", 0),
                content=r.get("content", ""),
                similarity_score=r.get("score", 0.0),
            ))

            # Aggregate sources
            src_key = f"{app_name}:{file_name}"
            if src_key not in sources_map:
                sources_map[src_key] = RAGSourceResponse(
                    document_id=doc_id,
                    file_name=file_name,
                    app_name=app_name,
                    file_path=file_path,
                    chunks_used=0,
                )
            sources_map[src_key].chunks_used += 1

        # Format context string
        context_parts = []
        for c in chunks:
            context_parts.append(
                f"Source: {c.app_name} - {c.file_path} (Score: {c.similarity_score:.2f})\n"
                f"{c.content}\n"
            )
        context = "\n---\n".join(context_parts)

        # Rough token estimate
        total_tokens = len(context.split()) * 4 // 3

        return RAGQueryResponse(
            query=body.query,
            context=context,
            chunks=chunks,
            sources=list(sources_map.values()),
            total_tokens=total_tokens,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cloud RAG query error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
