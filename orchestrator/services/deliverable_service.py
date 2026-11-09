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
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.workspace_client import WorkspaceClient
from modules.documents.thumbnails.eligibility import thumbnail_url_for
from services.deliverable_filters import list_filter
from services.deliverable_tags import card_tags_on_register, tags_in
from modules.documents.thumbnails.schedule import thumbnail_after_register

logger = logging.getLogger(__name__)

# Maximum bytes of inline content returned by get_deliverable(include_content=True).
# Larger files are truncated with a content_truncated=True flag so callers can
# offer a download instead. Prevents OOM on huge files.
MAX_INLINE_CONTENT_BYTES = 1_000_000  # 1 MB


def _workspace_file_url(workspace_id: str | UUID, file_path: str) -> str:
    """Build a URL to serve a workspace file to the browser.

    Uses `/files/raw` for binary formats (images, PDFs, etc.) which returns
    actual bytes with correct MIME types. Uses `/files/content` for text files
    which returns JSON for the code viewer. Every extension whose artifact type
    streams by URL (``STREAMED_ARTIFACT_TYPES``: images and videos) is binary,
    so a .mov or .mkv video plays from the same route as an .mp4.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    _BINARY_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
        ".pdf", ".docx", ".xlsx", ".pptx", ".zip", ".tar", ".gz",
        ".mp4", ".mp3", ".wav", ".ogg", ".webm",
    }
    if ext in _BINARY_EXTENSIONS or EXTENSION_TO_ARTIFACT.get(ext) in STREAMED_ARTIFACT_TYPES:
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


# F244 (night 7): a ticket's run that failed still files its report, which the
# Reports page lists as critical. The outputs view called it "published", and #0177
# showed nine of them, each only the credit message. Its run's status rides in the
# report's metrics (api.board_tasks._auto_create_task_report).
FAILED_RUN = "failed"


def shown_status(row: Any) -> str:
    """The deliverable's status as people read it: a failed run's report is "failed"."""
    extra = getattr(row, "extra", None)
    if getattr(row, "artifact_type", None) == "report" and isinstance(extra, dict) \
            and extra.get("task_status") == FAILED_RUN:
        return FAILED_RUN
    return row.status


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


# Artifact types that should auto-register when agents write files. Archives
# and audio are excluded because agents rarely produce them as primary
# deliverables (usually intermediate/temp files). Video is a primary Deliverable
# (PRD-251 S0.4: rendered MP4s show in Outputs with a player).
AGENT_REGISTERABLE_ARTIFACT_TYPES = frozenset({
    "report", "image", "document", "slide", "spreadsheet", "code", "video",
})

# Artifact types whose bytes stream to the browser by URL (``/files/raw``),
# never inline through ``get_deliverable(include_content=True)``.
STREAMED_ARTIFACT_TYPES = frozenset({"image", "video"})


def _humanize_basename(file_path: str) -> str:
    """Turn 'weekly-sales-report.md' → 'Weekly Sales Report'."""
    base = os.path.basename(file_path or "")
    stem, _ = os.path.splitext(base)
    cleaned = re.sub(r"[-_]+", " ", stem).strip()
    return cleaned.title() if cleaned else base


def _validate_register_artifact_type(
    file_path: str, artifact_type: Optional[str]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Infer the artifact type and refuse blog_post/report (PRD-129/133 guard).

    Blog posts live in ``blog_posts`` (BlogService), reports in
    ``agent_reports`` (ReportService). The Outputs view already surfaces
    both — callers must write to the native service, never register() here.

    Returns (inferred_type, None) on success, or (inferred_type, error_response).
    """
    inferred_type = artifact_type or _infer_artifact_type(file_path)
    if inferred_type in ("blog_post", "report"):
        return inferred_type, {
            "success": False,
            "error": (
                f"register() refuses artifact_type='{inferred_type}'. "
                "Blog posts and reports are written by their native "
                "services and surfaced via v_workspace_outputs."
            ),
        }
    return inferred_type, None


class DeliverableService:
    """Service for managing deliverables (agent outputs) in a workspace."""

    def __init__(self, db: Session, workspace_id: UUID | str):
        self.db = db
        self.workspace_id = workspace_id

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------

    @thumbnail_after_register  # F353: a PDF, Word document or sheet gets its first page drawn
    @card_tags_on_register  # 7 Oct: a board card's Deliverable carries the card's tags
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

        inferred_type, type_error = _validate_register_artifact_type(file_path, artifact_type)
        if type_error:
            return type_error

        fields = self._derive_register_fields(
            file_path, title, file_type, storage_type, preview_url, preview_type, inferred_type
        )

        return self._upsert_and_log_deliverable(
            file_path=file_path,
            source_type=source_type,
            source_id=source_id,
            agent_id=agent_id,
            agent_name=agent_name,
            inferred_type=inferred_type,
            summary=summary,
            storage_type=storage_type,
            file_size_bytes=file_size_bytes,
            extra=extra,
            status=status,
            fields=fields,
        )

    def _upsert_and_log_deliverable(
        self,
        *,
        file_path: str,
        source_type: str,
        source_id: Optional[str],
        agent_id: Optional[int],
        agent_name: Optional[str],
        inferred_type: str,
        summary: Optional[str],
        storage_type: str,
        file_size_bytes: Optional[int],
        extra: Optional[Dict[str, Any]],
        status: str,
        fields: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Run the upsert, log the outcome, and shape register()'s response envelope."""
        try:
            deliverable_id, was_insert = self._upsert_deliverable_row({
                "workspace_id": str(self.workspace_id),
                "source_type": source_type,
                "source_id": source_id,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "artifact_type": inferred_type,
                "title": fields["final_title"],
                "summary": summary,
                "storage_type": storage_type,
                "file_path": file_path,
                "file_name": fields["file_name"],
                "file_type": fields["file_type"],
                "file_size_bytes": file_size_bytes,
                "preview_url": fields["preview_url"],
                "preview_type": fields["preview_type"],
                "extra": json.dumps(extra or {}),
                "status": status,
            })

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
                "title": fields["final_title"],
            }
        except Exception as exc:
            self.db.rollback()
            logger.error(
                "[DeliverableService] register() failed path=%s: %s",
                file_path, exc, exc_info=True,
            )
            # CodeQL py/stack-trace-exposure: exc's text (a DB error, a path, ...) stops
            # at the server log; the caller gets a fixed, generic message.
            return {"success": False, "error": "register failed"}

    def _derive_register_fields(
        self,
        file_path: str,
        title: Optional[str],
        file_type: Optional[str],
        storage_type: str,
        preview_url: Optional[str],
        preview_type: Optional[str],
        inferred_type: str,
    ) -> Dict[str, Optional[str]]:
        """Fill in title/file_name/file_type/preview_url/preview_type defaults."""
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

        return {
            "final_title": final_title,
            "file_name": file_name,
            "file_type": file_type,
            "preview_url": preview_url,
            "preview_type": preview_type,
        }

    def _upsert_deliverable_row(self, params: Dict[str, Any]) -> Tuple[Optional[str], bool]:
        """Run the idempotent INSERT ... ON CONFLICT upsert; return (id, was_insert)."""
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
            params,
        )
        row = result.fetchone()
        self.db.commit()
        deliverable_id = str(row[0]) if row else None
        was_insert = bool(row[1]) if row else False
        return deliverable_id, was_insert

    # ------------------------------------------------------------------
    # list_deliverables
    # ------------------------------------------------------------------

    def list_deliverables(
        self, *, artifact_type: Optional[str] = None, source_type: Optional[str] = None,
        source_type_exclude: Optional[str] = None, source_id: Optional[str] = None, agent_id: Optional[int] = None,
        date_from: Optional[str] = None, date_to: Optional[str] = None, search: Optional[str] = None,
        tag: Optional[str] = None, limit: int = 24, offset: int = 0,
    ) -> Dict[str, Any]:
        """List deliverables with filters. Excludes soft-deleted rows.

        ``source_type_exclude`` is a comma-separated list of source_types to
        exclude (e.g. ``"heartbeat"``). Used by the redesigned Deliverables
        feed to keep agent self-status noise out of the main grid.

        ``source_id`` (PRD-164 S3) scopes to one originating mission/task/
        heartbeat — the mission-page Deliverables tab and the
        platform_list_deliverables tool filter on it.

        ``tag`` (7 Oct) keeps the Deliverables carrying that tag in ``extra.tags``,
        whatever case it was stored in (services/deliverable_filters.py).
        """
        limit = max(1, min(int(limit or 24), 100))
        offset = max(0, int(offset or 0))
        where, params = list_filter(self.workspace_id, {
            "artifact_type": artifact_type, "source_type": source_type, "source_type_exclude": source_type_exclude,
            "source_id": source_id, "agent_id": agent_id, "date_from": date_from, "date_to": date_to,
            "search": search, "tag": tag,
        })

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

            return {"success": True, "deliverables": [self._row_to_dict(r) for r in rows], "total": total,
                    "limit": limit, "offset": offset}
        except Exception as exc:
            logger.error("[DeliverableService] list_deliverables() failed: %s", exc, exc_info=True)
            # CodeQL py/stack-trace-exposure: exc's text stays server-side (logged above).
            return {"success": False, "error": "list failed", "deliverables": [], "total": 0,
                    "limit": limit, "offset": offset}

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
                content_fields = await self._load_deliverable_content(data, deliverable_id)
                data = {**data, **content_fields}

            return {"success": True, "deliverable": data}
        except Exception as exc:
            logger.error(
                "[DeliverableService] get_deliverable(%s) failed: %s",
                deliverable_id, exc, exc_info=True,
            )
            # CodeQL py/stack-trace-exposure: exc's text stays server-side (logged above).
            return {"success": False, "error": "get failed"}

    async def _load_deliverable_content(self, data: Dict[str, Any], deliverable_id: str) -> Dict[str, Any]:
        """Compute the content fields for include_content=True, to merge into the result."""
        if data["artifact_type"] == "blog_post":
            return self._blog_post_content_fields(deliverable_id, data)
        if data["storage_type"] == "workspace" and data["file_path"]:
            return await self._workspace_file_content_fields(data)
        return {}

    def _blog_post_content_fields(self, deliverable_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Blog posts store content inline in blog_posts.content — no workspace round-trip
        needed. Side-effect: also fixes the historic "blog posts render empty" bug caused
        by NULL blog_posts.file_path when BlogService._write_content failed.
        """
        blog_row = self.db.execute(
            text("SELECT content FROM blog_posts WHERE id = :id"),
            {"id": deliverable_id},
        ).fetchone()
        return {
            "content_url": data.get("preview_url"),
            "content": (blog_row[0] if blog_row else "") or "",
            "content_truncated": False,
        }

    async def _workspace_file_content_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Read a workspace-stored file's content (capped), or its stream-only URL."""
        # Always compute content_url fresh — older rows may have preview_url
        # pointing at /files/content which is JSON, not binary.
        content_url = _workspace_file_url(self.workspace_id, data["file_path"])

        if data["artifact_type"] in STREAMED_ARTIFACT_TYPES:
            # Images and videos stream via URL, never inline.
            return {"content_url": content_url, "content": None}

        ws_client = WorkspaceClient(str(self.workspace_id))
        file_result = await ws_client.read_file(data["file_path"])
        if not file_result.get("success"):
            return {
                "content_url": content_url,
                "content": None,
                "content_error": file_result.get("error", "Could not read file"),
            }
        content = file_result.get("content", "") or ""
        return {"content_url": content_url, **self._capped_content_fields(content)}

    @staticmethod
    def _capped_content_fields(content: str) -> Dict[str, Any]:
        """Cap inline content to prevent OOM on huge files — caller can hit content_url."""
        encoded = content.encode("utf-8", errors="ignore")
        if len(encoded) > MAX_INLINE_CONTENT_BYTES:
            truncated = encoded[:MAX_INLINE_CONTENT_BYTES].decode("utf-8", errors="ignore")
            return {"content": truncated, "content_truncated": True}
        return {"content": content, "content_truncated": False}

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate counts for the workspace: total, by_type, by_agent — plus the
        clean-render rate over rendered documents (P2-09 S4).

        ``extra.render`` is written at registration by the document generation
        service (``unresolved_count`` / ``unknown_count`` / ``template_lane``);
        rows without it (blog posts, reports, pre-P2-09 documents) are excluded
        from the render aggregate. Honest empty: no rendered documents →
        ``clean_render_rate`` is ``None``, never a fabricated 100%.
        """
        try:
            total = self._count_outputs()
            by_type_rows = self._count_outputs_by_type()
            by_agent_rows = self._count_outputs_by_agent()
            # P2-09 S4 — clean-render rate + template-lane coverage over the
            # rendered Deliverables (rows carrying `extra.render`).
            render_row = self._render_quality_row()

            return self._build_stats_response(total, by_type_rows, by_agent_rows, render_row)
        except Exception as exc:
            logger.error(
                "[DeliverableService] get_stats() failed: %s",
                exc, exc_info=True,
            )
            # CodeQL py/stack-trace-exposure: exc's text stays server-side (logged above).
            return {
                "success": False,
                "error": "stats failed",
                "total": 0,
                "by_type": {},
                "by_agent": [],
                "clean_render_rate": None,
                "render": {"total": 0, "clean": 0, "by_lane": {"block": 0, "legacy": 0}},
            }

    def _count_outputs(self) -> int:
        """Total non-deleted outputs in the workspace."""
        return self.db.execute(
            text("""
                SELECT COUNT(*)
                FROM v_workspace_outputs
                WHERE workspace_id = :workspace_id
                  AND deleted_at IS NULL
            """),
            {"workspace_id": str(self.workspace_id)},
        ).scalar() or 0

    def _count_outputs_by_type(self) -> Any:
        """Non-deleted output counts grouped by artifact_type, most common first."""
        return self.db.execute(
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

    def _count_outputs_by_agent(self) -> Any:
        """Non-deleted output counts grouped by agent, most prolific first."""
        return self.db.execute(
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

    def _render_quality_row(self) -> Any:
        """Rendered-document totals: clean count and template-lane split."""
        return self.db.execute(
            text("""
                SELECT
                    COUNT(*) AS rendered_total,
                    COUNT(*) FILTER (
                        WHERE COALESCE((extra->'render'->>'unresolved_count')::int, 0) = 0
                          AND COALESCE((extra->'render'->>'unknown_count')::int, 0) = 0
                    ) AS rendered_clean,
                    COUNT(*) FILTER (
                        WHERE extra->'render'->>'template_lane' = 'block'
                    ) AS lane_block,
                    COUNT(*) FILTER (
                        WHERE extra->'render'->>'template_lane' = 'legacy'
                    ) AS lane_legacy
                FROM v_workspace_outputs
                WHERE workspace_id = :workspace_id
                  AND deleted_at IS NULL
                  AND (extra -> 'render') IS NOT NULL
            """),
            {"workspace_id": str(self.workspace_id)},
        ).fetchone()

    @staticmethod
    def _build_stats_response(total: int, by_type_rows: Any, by_agent_rows: Any, render_row: Any) -> Dict[str, Any]:
        """Assemble the get_stats() success envelope from its four query results."""
        rendered_total = int(render_row.rendered_total or 0) if render_row else 0
        rendered_clean = int(render_row.rendered_clean or 0) if render_row else 0
        clean_render_rate = (
            rendered_clean / rendered_total if rendered_total else None
        )

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
            "clean_render_rate": clean_render_rate,
            "render": {
                "total": rendered_total,
                "clean": rendered_clean,
                "by_lane": {
                    "block": int(render_row.lane_block or 0) if render_row else 0,
                    "legacy": int(render_row.lane_legacy or 0) if render_row else 0,
                },
            },
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
            # CodeQL py/stack-trace-exposure: exc's text stays server-side (logged above).
            return {"success": False, "error": "delete failed"}

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
                pruned_ids = self._prune_old_agent_report_heartbeats(keep_per_agent)
            else:
                pruned_ids = self._prune_old_deliverables(source_type, keep_per_agent)

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
            # CodeQL py/stack-trace-exposure: exc's text stays server-side (logged above).
            return {"success": False, "error": "retention failed"}

    def _prune_old_agent_report_heartbeats(self, keep_per_agent: int) -> list[str]:
        """Soft-delete heartbeat agent_reports beyond keep_per_agent per agent.

        agent_reports has its own source_type semantics baked into
        heartbeat_result_id — we retain any row tied to a heartbeat.
        Returns the pruned row ids (not yet committed).
        """
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
        return [str(r.id) for r in result.fetchall()]

    def _prune_old_deliverables(self, source_type: str, keep_per_agent: int) -> list[str]:
        """Soft-delete deliverables of source_type beyond keep_per_agent per agent.

        Returns the pruned row ids (not yet committed).
        """
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
        return [str(r.id) for r in result.fetchall()]

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
            "thumbnail_url": thumbnail_url_for(row),  # F353: the card's first-page picture
            "extra": row.extra or {},
            "tags": tags_in(row.extra),  # 7 Oct: the Deliverable's tags, as its panel shows them
            "status": shown_status(row),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
