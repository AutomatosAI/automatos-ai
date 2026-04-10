"""
Deliverable Service (PRD-129: Workspace Outputs Hub)
====================================================

Owns CRUD for the ``deliverables`` table — metadata about every agent output
(reports, images, documents, code, slides, spreadsheets, archives, audio, video).

Design principles
-----------------
- Registration is idempotent via ``ON CONFLICT (workspace_id, file_path)`` —
  callers can re-register safely (e.g. file overwrite, backfill reruns).
- ``register()`` does NOT touch WorkspaceClient. Callers that already have
  ``file_size_bytes`` pass it in — avoids a hot-path HTTP round-trip on every
  file write.
- File content is only fetched on ``get_deliverable(include_content=True)``.
- Soft delete via ``deleted_at``; the unique constraint is partial
  (``WHERE deleted_at IS NULL``), so re-creating a path after delete works.
- All SQL uses ``sqlalchemy.text()`` with bound params. No string interpolation
  of user input.
- All public methods return dict envelopes with a ``success`` key.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification: file extension → artifact_type
# ---------------------------------------------------------------------------

EXTENSION_TO_ARTIFACT: Dict[str, str] = {
    # Images
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".svg": "image", ".bmp": "image", ".ico": "image",
    ".tiff": "image", ".tif": "image",
    # Reports / markdown
    ".md": "report", ".markdown": "report", ".rst": "report",
    # Documents
    ".pdf": "document", ".doc": "document", ".docx": "document",
    ".odt": "document", ".txt": "document", ".rtf": "document",
    # Slides
    ".ppt": "slide", ".pptx": "slide", ".odp": "slide", ".key": "slide",
    # Spreadsheets
    ".xls": "spreadsheet", ".xlsx": "spreadsheet", ".ods": "spreadsheet",
    ".csv": "spreadsheet", ".tsv": "spreadsheet",
    # Code
    ".py": "code", ".js": "code", ".ts": "code", ".tsx": "code", ".jsx": "code",
    ".go": "code", ".rs": "code", ".java": "code", ".kt": "code", ".swift": "code",
    ".c": "code", ".cpp": "code", ".h": "code", ".hpp": "code", ".cs": "code",
    ".rb": "code", ".php": "code", ".sh": "code", ".bash": "code", ".zsh": "code",
    ".sql": "code", ".yaml": "code", ".yml": "code", ".json": "code",
    ".toml": "code", ".html": "code", ".css": "code", ".scss": "code",
    # Archives
    ".zip": "archive", ".tar": "archive", ".gz": "archive", ".bz2": "archive",
    ".7z": "archive", ".rar": "archive",
    # Audio
    ".mp3": "audio", ".wav": "audio", ".ogg": "audio", ".flac": "audio",
    ".m4a": "audio", ".aac": "audio",
    # Video
    ".mp4": "video", ".mov": "video", ".avi": "video", ".mkv": "video",
    ".webm": "video",
}


def _slugify(value: str) -> str:
    """Convert a string to a kebab-case slug."""
    value = (value or "").lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:80]


def _infer_artifact_type(file_path: str) -> str:
    """Infer artifact_type from file extension; fallback 'document'."""
    if not file_path:
        return "document"
    _, ext = os.path.splitext(file_path.lower())
    return EXTENSION_TO_ARTIFACT.get(ext, "document")


def _humanize_basename(file_path: str) -> str:
    """Turn 'weekly-sales-report.md' → 'Weekly Sales Report'."""
    base = os.path.basename(file_path or "")
    stem, _ = os.path.splitext(base)
    cleaned = re.sub(r"[-_]+", " ", stem).strip()
    return cleaned.title() if cleaned else base


class DeliverableService:
    """Service for managing deliverables (agent outputs) in a workspace."""

    def __init__(self, db: Session, workspace_id: UUID | str):
        self.db = db
        self.workspace_id = workspace_id

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        file_path: str,
        title: Optional[str] = None,
        source_type: str = "chat",
        source_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
        summary: Optional[str] = None,
        storage_type: str = "workspace",
        file_type: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        preview_url: Optional[str] = None,
        preview_type: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        status: str = "ready",
    ) -> Dict[str, Any]:
        """Idempotently register a deliverable.

        Uses ON CONFLICT on ``uq_deliverables_workspace_path`` so that a
        re-register (agent overwrites a file) updates metadata in place
        instead of creating a duplicate row.
        """
        if not file_path:
            return {"success": False, "error": "file_path is required"}

        inferred_type = artifact_type or _infer_artifact_type(file_path)
        final_title = title or _humanize_basename(file_path)
        file_name = os.path.basename(file_path)
        if file_type is None:
            _, ext = os.path.splitext(file_path)
            file_type = ext.lstrip(".") or None

        try:
            result = self.db.execute(
                text("""
                    INSERT INTO deliverables (
                        workspace_id, source_type, source_id,
                        agent_id, agent_name,
                        artifact_type, title, summary,
                        storage_type, file_path, file_name, file_type, file_size_bytes,
                        preview_url, preview_type,
                        extra, status, created_at, updated_at
                    ) VALUES (
                        :workspace_id, :source_type, :source_id,
                        :agent_id, :agent_name,
                        :artifact_type, :title, :summary,
                        :storage_type, :file_path, :file_name, :file_type, :file_size_bytes,
                        :preview_url, :preview_type,
                        CAST(:extra AS JSONB), :status, NOW(), NOW()
                    )
                    ON CONFLICT (workspace_id, file_path)
                        WHERE deleted_at IS NULL
                    DO UPDATE SET
                        source_type     = EXCLUDED.source_type,
                        source_id       = EXCLUDED.source_id,
                        agent_id        = COALESCE(EXCLUDED.agent_id, deliverables.agent_id),
                        agent_name      = COALESCE(EXCLUDED.agent_name, deliverables.agent_name),
                        artifact_type   = EXCLUDED.artifact_type,
                        title           = EXCLUDED.title,
                        summary         = COALESCE(EXCLUDED.summary, deliverables.summary),
                        storage_type    = EXCLUDED.storage_type,
                        file_name       = EXCLUDED.file_name,
                        file_type       = EXCLUDED.file_type,
                        file_size_bytes = COALESCE(EXCLUDED.file_size_bytes, deliverables.file_size_bytes),
                        preview_url     = COALESCE(EXCLUDED.preview_url, deliverables.preview_url),
                        preview_type    = COALESCE(EXCLUDED.preview_type, deliverables.preview_type),
                        extra           = deliverables.extra || EXCLUDED.extra,
                        status          = EXCLUDED.status,
                        updated_at      = NOW()
                    RETURNING id, (xmax = 0) AS inserted
                """),
                {
                    "workspace_id": str(self.workspace_id),
                    "source_type": source_type,
                    "source_id": source_id,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "artifact_type": inferred_type,
                    "title": final_title,
                    "summary": summary,
                    "storage_type": storage_type,
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_type": file_type,
                    "file_size_bytes": file_size_bytes,
                    "preview_url": preview_url,
                    "preview_type": preview_type,
                    "extra": json.dumps(extra or {}),
                    "status": status,
                },
            )
            row = result.fetchone()
            self.db.commit()

            deliverable_id = str(row[0]) if row else None
            was_insert = bool(row[1]) if row else False

            logger.info(
                "[DeliverableService] %s deliverable %s (%s) path=%s",
                "Registered" if was_insert else "Updated",
                deliverable_id, inferred_type, file_path,
            )

            return {
                "success": True,
                "deliverable_id": deliverable_id,
                "created": was_insert,
                "artifact_type": inferred_type,
                "title": final_title,
            }
        except Exception as exc:
            self.db.rollback()
            logger.error(
                "[DeliverableService] register() failed path=%s: %s",
                file_path, exc, exc_info=True,
            )
            return {"success": False, "error": f"register failed: {exc}"}

    # ------------------------------------------------------------------
    # list_deliverables
    # ------------------------------------------------------------------

    def list_deliverables(
        self,
        *,
        artifact_type: Optional[str] = None,
        source_type: Optional[str] = None,
        agent_id: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 24,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List deliverables with filters. Excludes soft-deleted rows."""
        limit = max(1, min(int(limit or 24), 100))
        offset = max(0, int(offset or 0))

        conditions = ["d.workspace_id = :workspace_id", "d.deleted_at IS NULL"]
        params: Dict[str, Any] = {"workspace_id": str(self.workspace_id)}

        if artifact_type:
            conditions.append("d.artifact_type = :artifact_type")
            params["artifact_type"] = artifact_type
        if source_type:
            conditions.append("d.source_type = :source_type")
            params["source_type"] = source_type
        if agent_id is not None:
            conditions.append("d.agent_id = :agent_id")
            params["agent_id"] = agent_id
        if date_from:
            conditions.append("d.created_at >= :date_from")
            params["date_from"] = date_from
        if date_to:
            conditions.append("d.created_at <= :date_to")
            params["date_to"] = date_to
        if search:
            conditions.append("(d.title ILIKE :search OR d.summary ILIKE :search OR d.file_path ILIKE :search)")
            params["search"] = f"%{search}%"

        where = " AND ".join(conditions)

        try:
            total = self.db.execute(
                text(f"SELECT COUNT(*) FROM deliverables d WHERE {where}"),
                params,
            ).scalar() or 0

            rows = self.db.execute(
                text(f"""
                    SELECT
                        d.id, d.workspace_id,
                        d.source_type, d.source_id,
                        d.agent_id,
                        COALESCE(d.agent_name, a.name) AS agent_name,
                        d.artifact_type, d.title, d.summary,
                        d.storage_type, d.file_path, d.file_name, d.file_type, d.file_size_bytes,
                        d.preview_url, d.preview_type,
                        d.extra, d.status,
                        d.created_at, d.updated_at
                    FROM deliverables d
                    LEFT JOIN agents a ON a.id = d.agent_id
                    WHERE {where}
                    ORDER BY d.created_at DESC
                    LIMIT :limit OFFSET :offset
                """),
                {**params, "limit": limit, "offset": offset},
            ).fetchall()

            return {
                "success": True,
                "deliverables": [self._row_to_dict(r) for r in rows],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            logger.error(
                "[DeliverableService] list_deliverables() failed: %s",
                exc, exc_info=True,
            )
            return {
                "success": False,
                "error": f"list failed: {exc}",
                "deliverables": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
            }

    # ------------------------------------------------------------------
    # get_deliverable
    # ------------------------------------------------------------------

    async def get_deliverable(
        self,
        deliverable_id: str,
        include_content: bool = False,
    ) -> Dict[str, Any]:
        """Fetch a single deliverable. Optionally reads file content."""
        try:
            row = self.db.execute(
                text("""
                    SELECT
                        d.id, d.workspace_id,
                        d.source_type, d.source_id,
                        d.agent_id,
                        COALESCE(d.agent_name, a.name) AS agent_name,
                        d.artifact_type, d.title, d.summary,
                        d.storage_type, d.file_path, d.file_name, d.file_type, d.file_size_bytes,
                        d.preview_url, d.preview_type,
                        d.extra, d.status,
                        d.created_at, d.updated_at
                    FROM deliverables d
                    LEFT JOIN agents a ON a.id = d.agent_id
                    WHERE d.id = :id
                      AND d.workspace_id = :workspace_id
                      AND d.deleted_at IS NULL
                """),
                {"id": deliverable_id, "workspace_id": str(self.workspace_id)},
            ).fetchone()

            if not row:
                return {"success": False, "error": "Deliverable not found"}

            data = self._row_to_dict(row)

            if include_content and data["storage_type"] == "workspace":
                if data["artifact_type"] == "image":
                    # Images stream via URL, not inline text.
                    data["content"] = None
                    data["content_url"] = data.get("preview_url") or data["file_path"]
                else:
                    ws_client = WorkspaceClient(str(self.workspace_id))
                    file_result = await ws_client.read_file(data["file_path"])
                    if file_result.get("success"):
                        data["content"] = file_result.get("content", "")
                    else:
                        data["content"] = None
                        data["content_error"] = file_result.get("error", "Could not read file")

            return {"success": True, "deliverable": data}
        except Exception as exc:
            logger.error(
                "[DeliverableService] get_deliverable(%s) failed: %s",
                deliverable_id, exc, exc_info=True,
            )
            return {"success": False, "error": f"get failed: {exc}"}

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate counts for the workspace: total, by_type, by_agent."""
        try:
            total = self.db.execute(
                text("""
                    SELECT COUNT(*)
                    FROM deliverables
                    WHERE workspace_id = :workspace_id
                      AND deleted_at IS NULL
                """),
                {"workspace_id": str(self.workspace_id)},
            ).scalar() or 0

            by_type_rows = self.db.execute(
                text("""
                    SELECT artifact_type, COUNT(*) AS cnt
                    FROM deliverables
                    WHERE workspace_id = :workspace_id
                      AND deleted_at IS NULL
                    GROUP BY artifact_type
                    ORDER BY cnt DESC
                """),
                {"workspace_id": str(self.workspace_id)},
            ).fetchall()

            by_agent_rows = self.db.execute(
                text("""
                    SELECT
                        d.agent_id,
                        COALESCE(d.agent_name, a.name, 'Unknown') AS agent_name,
                        COUNT(*) AS cnt
                    FROM deliverables d
                    LEFT JOIN agents a ON a.id = d.agent_id
                    WHERE d.workspace_id = :workspace_id
                      AND d.deleted_at IS NULL
                    GROUP BY d.agent_id, COALESCE(d.agent_name, a.name, 'Unknown')
                    ORDER BY cnt DESC
                """),
                {"workspace_id": str(self.workspace_id)},
            ).fetchall()

            return {
                "success": True,
                "total": total,
                "by_type": {r.artifact_type: r.cnt for r in by_type_rows},
                "by_agent": [
                    {
                        "agent_id": r.agent_id,
                        "agent_name": r.agent_name,
                        "count": r.cnt,
                    }
                    for r in by_agent_rows
                ],
            }
        except Exception as exc:
            logger.error(
                "[DeliverableService] get_stats() failed: %s",
                exc, exc_info=True,
            )
            return {
                "success": False,
                "error": f"stats failed: {exc}",
                "total": 0,
                "by_type": {},
                "by_agent": [],
            }

    # ------------------------------------------------------------------
    # soft_delete
    # ------------------------------------------------------------------

    def soft_delete(self, deliverable_id: str) -> Dict[str, Any]:
        """Mark deliverable as deleted (sets deleted_at = NOW())."""
        try:
            row = self.db.execute(
                text("""
                    UPDATE deliverables
                    SET deleted_at = NOW(), updated_at = NOW()
                    WHERE id = :id
                      AND workspace_id = :workspace_id
                      AND deleted_at IS NULL
                    RETURNING id
                """),
                {"id": deliverable_id, "workspace_id": str(self.workspace_id)},
            ).fetchone()
            self.db.commit()

            if not row:
                return {"success": False, "error": "Deliverable not found"}

            logger.info(
                "[DeliverableService] Soft-deleted deliverable %s",
                deliverable_id,
            )
            return {"success": True, "deliverable_id": deliverable_id}
        except Exception as exc:
            self.db.rollback()
            logger.error(
                "[DeliverableService] soft_delete(%s) failed: %s",
                deliverable_id, exc, exc_info=True,
            )
            return {"success": False, "error": f"delete failed: {exc}"}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        """Convert a SQLAlchemy row to a JSON-serialisable dict."""
        extra = row.extra
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except (ValueError, TypeError):
                extra = {}
        elif extra is None:
            extra = {}

        return {
            "id": str(row.id),
            "workspace_id": str(row.workspace_id),
            "source_type": row.source_type,
            "source_id": row.source_id,
            "agent_id": row.agent_id,
            "agent_name": row.agent_name,
            "artifact_type": row.artifact_type,
            "title": row.title,
            "summary": row.summary,
            "storage_type": row.storage_type,
            "file_path": row.file_path,
            "file_name": row.file_name,
            "file_type": row.file_type,
            "file_size_bytes": row.file_size_bytes,
            "preview_url": row.preview_url,
            "preview_type": row.preview_type,
            "extra": extra,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
