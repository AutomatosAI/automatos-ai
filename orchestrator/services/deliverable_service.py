"""
Deliverable Service (PRD-133b corrected: Unified Outputs via view)
==================================================================

Reads the ``v_workspace_outputs`` view which UNIONs:
  * ``blog_posts``   — owned by BlogService
  * ``agent_reports``— owned by ReportService
  * ``deliverables`` — ad-hoc artifacts (code/images/etc.) with no native home

Write paths branch by artifact_type:
  * ``blog_post`` / ``report`` are rejected from ``register()`` — use
    BlogService or ReportService directly (those are the source of truth).
  * ``soft_delete()`` routes the UPDATE at the correct source table based on
    the row's ``artifact_type``.
  * ``apply_retention()`` targets ``agent_reports`` directly (the only place
    with ``source_type='heartbeat'`` after PRD-133b).

Design principles
-----------------
- One write path per artifact type. No shadow writes, no drift.
- ``register()`` for ad-hoc artifacts stays idempotent via
  ``ON CONFLICT (workspace_id, file_path)``.
- ``register()`` does NOT touch WorkspaceClient. Callers that already have
  ``file_size_bytes`` pass it in — avoids a hot-path HTTP round-trip.
- File content is only fetched on ``get_deliverable(include_content=True)``.
  Blog posts return inline content from ``blog_posts.content`` (no workspace
  fetch needed — the content lives in the DB).
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
from urllib.parse import quote
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

# Maximum bytes of inline content returned by get_deliverable(include_content=True).
# Larger files are truncated with a content_truncated=True flag so callers can
# offer a download instead. Prevents OOM on huge files.
MAX_INLINE_CONTENT_BYTES = 1_000_000  # 1 MB


def _workspace_file_url(workspace_id: str | UUID, file_path: str) -> str:
    """Build a URL to serve a workspace file to the browser.

    Uses `/files/raw` for binary formats (images, PDFs, etc.) which returns
    actual bytes with correct MIME types. Uses `/files/content` for text files
    which returns JSON for the code viewer.
    """
    _, ext = os.path.splitext(file_path)
    _BINARY_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
        ".pdf", ".docx", ".xlsx", ".pptx", ".zip", ".tar", ".gz",
        ".mp4", ".mp3", ".wav", ".ogg", ".webm",
    }
    if ext.lower() in _BINARY_EXTENSIONS:
        return f"/api/workspaces/{workspace_id}/files/raw?path={quote(file_path)}"
    return f"/api/workspaces/{workspace_id}/files/content?path={quote(file_path)}"


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


# Artifact types that should auto-register when agents write files. Archives,
# audio, and video are excluded because agents rarely produce them as primary
# deliverables (usually intermediate/temp files).
AGENT_REGISTERABLE_ARTIFACT_TYPES = frozenset({
    "report", "image", "document", "slide", "spreadsheet", "code",
})


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

        Idempotency key is ``(workspace_id, file_path)``. If you need history
        (e.g. daily heartbeat reports), include a timestamp or UUID in the
        ``file_path`` — otherwise the same path will overwrite. Reports
        service already does this via ``reports/{agent}/{ts}_{uuid}_{slug}.md``.
        """
        if not file_path:
            return {"success": False, "error": "file_path is required"}

        inferred_type = artifact_type or _infer_artifact_type(file_path)

        # Guard against reintroducing the PRD-129/133 double-write. Blog posts
        # live in ``blog_posts`` (BlogService), reports in ``agent_reports``
        # (ReportService). The Outputs view already surfaces both — callers
        # must write to the native service, never register() here.
        if inferred_type in ("blog_post", "report"):
            return {
                "success": False,
                "error": (
                    f"register() refuses artifact_type='{inferred_type}'. "
                    "Blog posts and reports are written by their native "
                    "services and surfaced via v_workspace_outputs."
                ),
            }

        final_title = title or _humanize_basename(file_path)
        file_name = os.path.basename(file_path)
        if file_type is None:
            _, ext = os.path.splitext(file_path)
            file_type = ext.lstrip(".") or None

        # Auto-wire preview_url for workspace-stored artifacts so the frontend
        # can render images via <img src> and offer real Download links for
        # everything else. Callers may override by passing preview_url explicitly.
        if preview_url is None and storage_type == "workspace":
            preview_url = _workspace_file_url(self.workspace_id, file_path)
            if preview_type is None:
                preview_type = "image" if inferred_type == "image" else "file"

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
        source_type_exclude: Optional[str] = None,
        source_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 24,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List deliverables with filters. Excludes soft-deleted rows.

        ``source_type_exclude`` is a comma-separated list of source_types to
        exclude (e.g. ``"heartbeat"``). Used by the redesigned Deliverables
        feed to keep agent self-status noise out of the main grid.

        ``source_id`` (PRD-164 S3) scopes to one originating mission/task/
        heartbeat — the mission-page Deliverables tab and the
        platform_list_deliverables tool filter on it.
        """
        limit = max(1, min(int(limit or 24), 100))
        offset = max(0, int(offset or 0))

        conditions = ["o.workspace_id = :workspace_id", "o.deleted_at IS NULL"]
        params: Dict[str, Any] = {"workspace_id": str(self.workspace_id)}

        if artifact_type:
            conditions.append("o.artifact_type = :artifact_type")
            params["artifact_type"] = artifact_type
        if source_type:
            conditions.append("o.source_type = :source_type")
            params["source_type"] = source_type
        if source_type_exclude:
            excluded = [s.strip() for s in source_type_exclude.split(",") if s.strip()]
            if excluded:
                placeholders = ", ".join(f":excl_src_{i}" for i in range(len(excluded)))
                conditions.append(f"o.source_type NOT IN ({placeholders})")
                for i, s in enumerate(excluded):
                    params[f"excl_src_{i}"] = s
        if source_id:
            conditions.append("o.source_id = :source_id")
            params["source_id"] = str(source_id)
        if agent_id is not None:
            conditions.append("o.agent_id = :agent_id")
            params["agent_id"] = agent_id
        if date_from:
            conditions.append("o.created_at >= :date_from")
            params["date_from"] = date_from
        if date_to:
            conditions.append("o.created_at <= :date_to")
            params["date_to"] = date_to
        if search:
            conditions.append("(o.title ILIKE :search OR o.summary ILIKE :search OR o.file_path ILIKE :search)")
            params["search"] = f"%{search}%"

        where = " AND ".join(conditions)

        try:
            total = self.db.execute(
                text(f"SELECT COUNT(*) FROM v_workspace_outputs o WHERE {where}"),
                params,
            ).scalar() or 0

            rows = self.db.execute(
                text(f"""
                    SELECT
                        o.id, o.workspace_id,
                        o.source_type, o.source_id,
                        o.agent_id,
                        COALESCE(o.agent_name, a.name) AS agent_name,
                        o.artifact_type, o.title, o.summary,
                        o.storage_type, o.file_path, o.file_name, o.file_type, o.file_size_bytes,
                        o.preview_url, o.preview_type,
                        o.extra, o.status,
                        o.created_at, o.updated_at
                    FROM v_workspace_outputs o
                    LEFT JOIN agents a ON a.id = o.agent_id
                    WHERE {where}
                    ORDER BY o.created_at DESC
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
                        o.id, o.workspace_id,
                        o.source_type, o.source_id,
                        o.agent_id,
                        COALESCE(o.agent_name, a.name) AS agent_name,
                        o.artifact_type, o.title, o.summary,
                        o.storage_type, o.file_path, o.file_name, o.file_type, o.file_size_bytes,
                        o.preview_url, o.preview_type,
                        o.extra, o.status,
                        o.created_at, o.updated_at
                    FROM v_workspace_outputs o
                    LEFT JOIN agents a ON a.id = o.agent_id
                    WHERE o.id = :id
                      AND o.workspace_id = :workspace_id
                      AND o.deleted_at IS NULL
                """),
                {"id": deliverable_id, "workspace_id": str(self.workspace_id)},
            ).fetchone()

            if not row:
                return {"success": False, "error": "Deliverable not found"}

            data = self._row_to_dict(row)

            if include_content:
                # Blog posts store content inline in blog_posts.content — no
                # workspace round-trip needed. Side-effect: also fixes the
                # historic "blog posts render empty" bug caused by NULL
                # blog_posts.file_path when BlogService._write_content failed.
                if data["artifact_type"] == "blog_post":
                    blog_row = self.db.execute(
                        text("SELECT content FROM blog_posts WHERE id = :id"),
                        {"id": deliverable_id},
                    ).fetchone()
                    data["content_url"] = data.get("preview_url")
                    data["content"] = (blog_row[0] if blog_row else "") or ""
                    data["content_truncated"] = False
                elif data["storage_type"] == "workspace" and data["file_path"]:
                    # Always compute content_url fresh — older rows may have
                    # preview_url pointing at /files/content which is JSON, not binary.
                    data["content_url"] = _workspace_file_url(
                        self.workspace_id, data["file_path"]
                    )

                    if data["artifact_type"] == "image":
                        # Images stream via URL, never inline.
                        data["content"] = None
                    else:
                        ws_client = WorkspaceClient(str(self.workspace_id))
                        file_result = await ws_client.read_file(data["file_path"])
                        if file_result.get("success"):
                            content = file_result.get("content", "") or ""
                            # Cap inline content to prevent OOM on huge files —
                            # caller can hit content_url for the full stream.
                            if len(content.encode("utf-8", errors="ignore")) > MAX_INLINE_CONTENT_BYTES:
                                truncated = content.encode("utf-8", errors="ignore")[
                                    :MAX_INLINE_CONTENT_BYTES
                                ].decode("utf-8", errors="ignore")
                                data["content"] = truncated
                                data["content_truncated"] = True
                            else:
                                data["content"] = content
                                data["content_truncated"] = False
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
                    FROM v_workspace_outputs
                    WHERE workspace_id = :workspace_id
                      AND deleted_at IS NULL
                """),
                {"workspace_id": str(self.workspace_id)},
            ).scalar() or 0

            by_type_rows = self.db.execute(
                text("""
                    SELECT artifact_type, COUNT(*) AS cnt
                    FROM v_workspace_outputs
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
                        o.agent_id,
                        COALESCE(o.agent_name, a.name, 'Unknown') AS agent_name,
                        COUNT(*) AS cnt
                    FROM v_workspace_outputs o
                    LEFT JOIN agents a ON a.id = o.agent_id
                    WHERE o.workspace_id = :workspace_id
                      AND o.deleted_at IS NULL
                    GROUP BY o.agent_id, COALESCE(o.agent_name, a.name, 'Unknown')
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

    # Map artifact_type → native table for soft-delete. Everything else
    # (code/image/document/etc.) lives in deliverables.
    _SOFT_DELETE_TARGET = {
        "blog_post": "blog_posts",
        "report":    "agent_reports",
    }

    def soft_delete(self, deliverable_id: str) -> Dict[str, Any]:
        """Mark output as deleted (sets deleted_at = NOW()) on the correct
        source table based on artifact_type."""
        try:
            # Resolve artifact_type via the view so we UPDATE the right table.
            row = self.db.execute(
                text("""
                    SELECT artifact_type
                      FROM v_workspace_outputs
                     WHERE id = :id
                       AND workspace_id = :workspace_id
                       AND deleted_at IS NULL
                """),
                {"id": deliverable_id, "workspace_id": str(self.workspace_id)},
            ).fetchone()

            if not row:
                return {"success": False, "error": "Deliverable not found"}

            target_table = self._SOFT_DELETE_TARGET.get(row.artifact_type, "deliverables")
            # Table name is from a fixed allow-list above — safe to interpolate.
            self.db.execute(
                text(f"""
                    UPDATE {target_table}
                       SET deleted_at = NOW(), updated_at = NOW()
                     WHERE id = :id
                       AND workspace_id = :workspace_id
                       AND deleted_at IS NULL
                """),
                {"id": deliverable_id, "workspace_id": str(self.workspace_id)},
            )
            self.db.commit()

            logger.info(
                "[DeliverableService] Soft-deleted %s %s",
                row.artifact_type, deliverable_id,
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
    # retention — prune old heartbeat reports
    # ------------------------------------------------------------------

    def apply_retention(
        self,
        *,
        source_type: str = "heartbeat",
        keep_per_agent: int = 50,
    ) -> Dict[str, Any]:
        """Soft-delete old outputs beyond *keep_per_agent* most recent per agent.

        Only targets rows matching *source_type* (default: heartbeat) so that
        user-initiated outputs (chat, mission, task) are never auto-pruned.

        After PRD-133b, ``source_type='heartbeat'`` rows live in
        ``agent_reports`` — the only place they're ever written. For other
        source_types (e.g. task-driven code/image retention), we still target
        ``deliverables``.
        """
        target_table = "agent_reports" if source_type == "heartbeat" else "deliverables"
        try:
            if target_table == "agent_reports":
                # agent_reports has its own source_type semantics baked into
                # heartbeat_result_id — we retain any row tied to a heartbeat.
                rank_sql = text("""
                    WITH ranked AS (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY COALESCE(agent_id, 0)
                                   ORDER BY created_at DESC
                               ) AS rn
                          FROM agent_reports
                         WHERE workspace_id        = :workspace_id
                           AND heartbeat_result_id IS NOT NULL
                           AND deleted_at          IS NULL
                    )
                    UPDATE agent_reports
                       SET deleted_at = NOW(), updated_at = NOW()
                     WHERE id IN (SELECT id FROM ranked WHERE rn > :keep)
                    RETURNING id
                """)
                result = self.db.execute(
                    rank_sql,
                    {
                        "workspace_id": str(self.workspace_id),
                        "keep": keep_per_agent,
                    },
                )
            else:
                rank_sql = text("""
                    WITH ranked AS (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY COALESCE(agent_id, 0)
                                   ORDER BY created_at DESC
                               ) AS rn
                          FROM deliverables
                         WHERE workspace_id = :workspace_id
                           AND source_type  = :source_type
                           AND deleted_at   IS NULL
                    )
                    UPDATE deliverables
                       SET deleted_at = NOW(), updated_at = NOW()
                     WHERE id IN (SELECT id FROM ranked WHERE rn > :keep)
                    RETURNING id
                """)
                result = self.db.execute(
                    rank_sql,
                    {
                        "workspace_id": str(self.workspace_id),
                        "source_type": source_type,
                        "keep": keep_per_agent,
                    },
                )

            pruned_ids = [str(r.id) for r in result.fetchall()]
            self.db.commit()

            logger.info(
                "[DeliverableService] Retention pruned %d %s deliverables (kept %d per agent)",
                len(pruned_ids), source_type, keep_per_agent,
            )
            return {
                "success": True,
                "pruned": len(pruned_ids),
                "source_type": source_type,
                "keep_per_agent": keep_per_agent,
            }
        except Exception as exc:
            self.db.rollback()
            logger.error(
                "[DeliverableService] apply_retention() failed: %s",
                exc, exc_info=True,
            )
            return {"success": False, "error": f"retention failed: {exc}"}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        """Convert a SQLAlchemy row to a JSON-serialisable dict."""
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
            "preview_url": (
                _workspace_file_url(row.workspace_id, row.file_path)
                if row.storage_type == "workspace" and row.file_path
                else row.preview_url
            ),
            "preview_type": row.preview_type,
            "extra": row.extra or {},
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
