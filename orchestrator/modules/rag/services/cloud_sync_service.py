"""
Cloud Sync Service (PRD-42)
============================

Orchestrates cloud document syncing:
- List folders/files from connected cloud storage via Composio
- Trigger sync: download → ingest → store vectors in S3 Vectors
- Track sync state in cloud_documents and cloud_sync_jobs tables
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.composio.entity_manager import EntityManager
from core.composio.tool_executor import ComposioToolExecutor
from core.models.cloud_sync import CloudDocument, CloudSyncConfig, CloudSyncJob
from core.models.composio import ComposioConnection
from modules.rag.ingestion.pipeline import IngestionPipeline
from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend

logger = logging.getLogger(__name__)

# Composio actions for listing files/folders per provider
_LIST_ACTIONS = {
    "GOOGLEDRIVE": "GOOGLEDRIVE_LIST_FILES",
    "DROPBOX": "DROPBOX_LIST_FOLDER",
    "ONEDRIVE": "ONEDRIVE_LIST_FILES",
    "BOX": "BOX_LIST_FOLDER_ITEMS",
}


class CloudSyncService:
    """
    Orchestrates cloud document sync using existing infrastructure.

    Usage::

        service = CloudSyncService(db)
        folders = await service.list_folders(connection_id=1, path="/")
        files = await service.list_files(connection_id=1, path="/Automatos")
        job = await service.sync_folder(connection_id=1, workspace_id=ws_id)
    """

    def __init__(self, db: Session):
        self.db = db
        self.entity_manager = EntityManager(db)
        self.executor = ComposioToolExecutor(db)

    # ------------------------------------------------------------------
    # Folder navigation
    # ------------------------------------------------------------------

    async def list_folders(
        self,
        connection_id: int,
        path: str = "/",
    ) -> List[Dict[str, Any]]:
        """
        List folders at the given path in cloud storage.

        Returns list of dicts: {name, path, has_children}
        """
        connection = self._get_connection(connection_id)
        app_name = connection.app_name.upper()
        workspace_id = self._workspace_id_for_connection(connection)

        action = _LIST_ACTIONS.get(app_name)
        if not action:
            raise ValueError(f"Unsupported app for folder listing: {app_name}")

        params = self._build_list_params(app_name, path, folders_only=True)

        result = await self.executor.execute(
            action=action,
            params=params,
            agent_id=0,
            workspace_id=workspace_id,
            app_name=app_name,
            skip_validation=True,
        )

        if not result.get("success"):
            logger.error(f"list_folders failed: {result.get('error')}")
            return []

        return self._parse_folder_list(result.get("data", {}), app_name)

    # ------------------------------------------------------------------
    # File listing with sync status
    # ------------------------------------------------------------------

    async def list_files(
        self,
        connection_id: int,
        path: str = "/",
        recursive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List files at the given path, enriched with sync status from DB.
        """
        connection = self._get_connection(connection_id)
        app_name = connection.app_name.upper()
        workspace_id = self._workspace_id_for_connection(connection)

        action = _LIST_ACTIONS.get(app_name)
        if not action:
            raise ValueError(f"Unsupported app for file listing: {app_name}")

        params = self._build_list_params(app_name, path, folders_only=False)

        result = await self.executor.execute(
            action=action,
            params=params,
            agent_id=0,
            workspace_id=workspace_id,
            app_name=app_name,
            skip_validation=True,
        )

        if not result.get("success"):
            logger.error(f"list_files failed: {result.get('error')}")
            return []

        cloud_files = self._parse_file_list(result.get("data", {}), app_name)

        # Enrich with sync status from cloud_documents table
        external_ids = [f["external_file_id"] for f in cloud_files]
        synced_docs = (
            self.db.query(CloudDocument)
            .filter(
                CloudDocument.connection_id == connection_id,
                CloudDocument.external_file_id.in_(external_ids),
            )
            .all()
        )
        synced_map = {doc.external_file_id: doc for doc in synced_docs}

        for f in cloud_files:
            doc = synced_map.get(f["external_file_id"])
            if doc:
                f["is_synced"] = True
                f["sync_status"] = doc.sync_status
                f["chunk_count"] = doc.chunk_count or 0
                f["last_synced_at"] = doc.last_synced_at.isoformat() if doc.last_synced_at else None
            else:
                f["is_synced"] = False
                f["sync_status"] = "pending"
                f["chunk_count"] = 0
                f["last_synced_at"] = None

        return cloud_files

    # ------------------------------------------------------------------
    # Sync orchestration
    # ------------------------------------------------------------------

    async def sync_folder(
        self,
        connection_id: int,
        workspace_id: UUID,
    ) -> CloudSyncJob:
        """
        Sync all files under the configured root folder for a connection.

        Creates a CloudSyncJob, lists files, downloads + ingests new/modified
        files, updates cloud_documents records, and returns the completed job.
        """
        connection = self._get_connection(connection_id)
        app_name = connection.app_name.upper()

        # Get sync config for root folder
        sync_config = (
            self.db.query(CloudSyncConfig)
            .filter(CloudSyncConfig.connection_id == connection_id)
            .first()
        )
        if not sync_config:
            raise ValueError(
                f"No sync config found for connection {connection_id}. "
                "Select a root folder first."
            )

        root_path = sync_config.root_folder_path

        # Create sync job
        job = CloudSyncJob(
            workspace_id=workspace_id,
            connection_id=connection_id,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        files_synced = 0
        files_skipped = 0
        files_errored = 0
        total_chunks = 0

        try:
            # List all files under root
            cloud_files = await self.list_files(
                connection_id=connection_id,
                path=root_path,
                recursive=True,
            )

            pipeline = IngestionPipeline()

            for cf in cloud_files:
                ext_id = cf["external_file_id"]
                file_name = cf.get("name", "unknown")
                modified_at = cf.get("modified_at")

                # Check existing cloud_documents record
                existing = (
                    self.db.query(CloudDocument)
                    .filter(
                        CloudDocument.connection_id == connection_id,
                        CloudDocument.external_file_id == ext_id,
                    )
                    .first()
                )

                # Skip if unchanged
                if existing and existing.sync_status == "synced":
                    if modified_at and existing.cloud_modified_at:
                        if modified_at <= existing.cloud_modified_at.isoformat():
                            files_skipped += 1
                            continue

                # Download + ingest
                try:
                    result = await pipeline.ingest_from_cloud(
                        app_name=app_name,
                        external_file_id=ext_id,
                        file_name=file_name,
                        workspace_id=workspace_id,
                        db_session=self.db,
                        metadata={
                            "source": "cloud",
                            "app_name": app_name,
                            "file_path": cf.get("path", ""),
                        },
                    )

                    if result.success:
                        # Upsert cloud_documents record
                        if existing:
                            existing.sync_status = "synced"
                            existing.last_synced_at = datetime.now(timezone.utc)
                            existing.cloud_modified_at = (
                                datetime.fromisoformat(modified_at)
                                if modified_at else None
                            )
                            existing.chunk_count = result.chunk_count
                            existing.sync_error = None
                        else:
                            new_doc = CloudDocument(
                                workspace_id=workspace_id,
                                connection_id=connection_id,
                                app_name=app_name,
                                external_file_id=ext_id,
                                file_name=file_name,
                                file_path=cf.get("path", ""),
                                mime_type=cf.get("mime_type"),
                                file_size=cf.get("size"),
                                s3_vector_bucket=f"automatos-vectors-{workspace_id}",
                                chunk_count=result.chunk_count,
                                cloud_modified_at=(
                                    datetime.fromisoformat(modified_at)
                                    if modified_at else None
                                ),
                                last_synced_at=datetime.now(timezone.utc),
                                sync_status="synced",
                            )
                            self.db.add(new_doc)

                        files_synced += 1
                        total_chunks += result.chunk_count
                    else:
                        # Mark error
                        if existing:
                            existing.sync_status = "error"
                            existing.sync_error = result.error
                        files_errored += 1

                    self.db.commit()

                except Exception as e:
                    logger.error(f"Sync failed for {file_name}: {e}")
                    files_errored += 1
                    if existing:
                        existing.sync_status = "error"
                        existing.sync_error = str(e)
                        self.db.commit()

            # Update job
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            job.files_synced = files_synced
            job.files_skipped = files_skipped
            job.files_errored = files_errored
            job.total_chunks_created = total_chunks

            # Update connection stats
            connection.total_documents_synced = (
                self.db.query(CloudDocument)
                .filter(
                    CloudDocument.connection_id == connection_id,
                    CloudDocument.sync_status == "synced",
                )
                .count()
            )
            connection.last_successful_sync = datetime.now(timezone.utc)

            # Update sync config
            sync_config.last_sync_at = datetime.now(timezone.utc)

            self.db.commit()
            self.db.refresh(job)

        except Exception as e:
            logger.error(f"Sync job {job.id} failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            job.files_synced = files_synced
            job.files_skipped = files_skipped
            job.files_errored = files_errored
            job.total_chunks_created = total_chunks
            self.db.commit()
            self.db.refresh(job)

        return job

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_connection(self, connection_id: int) -> ComposioConnection:
        connection = self.db.query(ComposioConnection).get(connection_id)
        if not connection:
            raise ValueError(f"Connection {connection_id} not found")
        return connection

    def _workspace_id_for_connection(self, connection: ComposioConnection) -> UUID:
        """Resolve workspace_id from a ComposioConnection → ComposioEntity."""
        from core.models.composio import ComposioEntity

        entity = self.db.query(ComposioEntity).get(connection.entity_id)
        if not entity:
            raise ValueError(f"Entity {connection.entity_id} not found")
        return entity.workspace_id

    @staticmethod
    def _build_list_params(
        app_name: str, path: str, folders_only: bool = False,
    ) -> dict:
        if app_name == "GOOGLEDRIVE":
            q = f"'{path}' in parents and trashed = false"
            if folders_only:
                q += " and mimeType = 'application/vnd.google-apps.folder'"
            return {"q": q}
        if app_name == "DROPBOX":
            return {"path": path if path != "/" else ""}
        if app_name in ("ONEDRIVE", "BOX"):
            return {"path": path}
        return {"path": path}

    @staticmethod
    def _parse_folder_list(
        data: Dict[str, Any], app_name: str,
    ) -> List[Dict[str, Any]]:
        """Normalize provider-specific folder list to standard format."""
        items = data.get("files") or data.get("entries") or data.get("items") or []
        folders = []
        for item in items:
            # Different providers use different keys
            is_folder = (
                item.get("mimeType") == "application/vnd.google-apps.folder"
                or item.get(".tag") == "folder"
                or item.get("type") == "folder"
                or item.get("folder") is not None
            )
            if not is_folder:
                continue
            folders.append({
                "name": item.get("name") or item.get("title") or "",
                "path": item.get("path_display") or item.get("path") or item.get("name", ""),
                "has_children": True,  # assume expandable
            })
        return folders

    @staticmethod
    def _parse_file_list(
        data: Dict[str, Any], app_name: str,
    ) -> List[Dict[str, Any]]:
        """Normalize provider-specific file list to standard format."""
        items = data.get("files") or data.get("entries") or data.get("items") or []
        files = []
        for item in items:
            is_folder = (
                item.get("mimeType") == "application/vnd.google-apps.folder"
                or item.get(".tag") == "folder"
                or item.get("type") == "folder"
            )
            if is_folder:
                continue
            files.append({
                "external_file_id": item.get("id") or item.get("path_display") or "",
                "name": item.get("name") or item.get("title") or "",
                "path": item.get("path_display") or item.get("path") or "",
                "mime_type": item.get("mimeType") or item.get("mime_type") or "",
                "size": item.get("size") or item.get("fileSize") or 0,
                "modified_at": (
                    item.get("modifiedTime")
                    or item.get("server_modified")
                    or item.get("lastModifiedDateTime")
                    or item.get("modified_at")
                ),
            })
        return files
