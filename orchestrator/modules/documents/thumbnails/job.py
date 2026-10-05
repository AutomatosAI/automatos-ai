"""F353 (issue #947): render one Deliverable's first page and record it.

The Deliverable is read from ``v_workspace_outputs`` (scoped to its workspace), so
the same job serves the ``deliverables`` rows (PDFs, Word documents, sheets) and
the ``agent_reports`` rows (reports). Heartbeat reports are skipped: the feed
hides them, and there are thousands.

Where the result is recorded:

* a ``deliverables`` row: ``extra.thumbnail`` = ``{"file": "thumb_<id>.png",
  "rendered_at": ...}``, or ``{"failed": "<reason>", "at": ...}`` after a render
  error, so the backfill does not retry it on every run;
* a report: nothing. ``agent_reports`` has no column for it (its ``metrics`` are
  shown to people and agents), so a report's card asks for the picture and an
  absent one falls back to the icon.

The database session is held only to read and to record, never during the render.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from sqlalchemy import text

from modules.documents.thumbnails.eligibility import HEARTBEAT, REPORT, THUMBNAIL_ARTIFACT_TYPES
from modules.documents.thumbnails.render import SUPPORTED_EXTENSIONS, ThumbnailError, render_png_isolated
from modules.documents.thumbnails.sources import MAX_SOURCE_BYTES, read_source
from modules.documents.thumbnails.store import save_thumbnail

logger = logging.getLogger(__name__)

MAX_FAILURE_CHARS = 300


@dataclass(frozen=True)
class Output:
    """The fields of a ``v_workspace_outputs`` row the job needs."""

    id: str
    workspace_id: str
    artifact_type: str
    source_type: str
    storage_type: str
    file_path: str
    file_size_bytes: Optional[int]

    @property
    def ext(self) -> str:
        return os.path.splitext(self.file_path or "")[1].lower()


def load_output(db: Any, workspace_id: str | UUID, output_id: str | UUID) -> Optional[Output]:
    """The live Deliverable ``output_id`` of ``workspace_id``; None when there is none."""
    row = db.execute(
        text("""
            SELECT o.id, o.workspace_id, o.artifact_type, o.source_type,
                   o.storage_type, o.file_path, o.file_size_bytes
            FROM v_workspace_outputs o
            WHERE o.id = CAST(:id AS uuid) AND o.workspace_id = CAST(:ws AS uuid)
              AND o.deleted_at IS NULL
        """),
        {"id": str(output_id), "ws": str(workspace_id)},
    ).fetchone()
    if row is None:
        return None
    return Output(str(row.id), str(row.workspace_id), row.artifact_type or "", row.source_type or "",
                  row.storage_type or "", row.file_path or "", row.file_size_bytes)


def skip_reason(output: Output) -> Optional[str]:
    """Why no picture is drawn for ``output``; None when one is."""
    if output.artifact_type not in THUMBNAIL_ARTIFACT_TYPES:
        return f"a {output.artifact_type or 'typeless'} Deliverable has no page to draw"
    if output.artifact_type == REPORT and output.source_type == HEARTBEAT:
        return "heartbeat reports are not drawn"
    if output.ext not in SUPPORTED_EXTENSIONS:
        return f"no preview is drawn for {output.ext or 'extensionless'} files"
    if (output.file_size_bytes or 0) > MAX_SOURCE_BYTES:
        return "the file is too big to draw"
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_result(db: Any, output: Output, result: Dict[str, Any]) -> None:
    """Merge ``{"thumbnail": result}`` into a ``deliverables`` row's ``extra`` (reports: nothing)."""
    if output.artifact_type == REPORT:
        return
    db.execute(
        text("""
            UPDATE deliverables
            SET extra = COALESCE(extra, '{}'::jsonb) || CAST(:patch AS JSONB)
            WHERE id = CAST(:id AS uuid) AND workspace_id = CAST(:ws AS uuid)
        """),
        {"patch": json.dumps({"thumbnail": result}), "id": output.id, "ws": output.workspace_id},
    )
    db.commit()


def _draw(output: Output, render: Callable[[bytes, str], bytes]) -> Dict[str, Any]:
    """Read, render and store; the dict to record, or ``{}`` when the file was not there."""
    data = read_source(output.workspace_id, output.storage_type, output.file_path)
    if data is None:
        logger.info("[F353] no thumbnail for %s: its file %s was not found", output.id, output.file_path)
        return {}
    try:
        png = render(data, output.ext)
    except ThumbnailError as e:
        logger.warning("[F353] thumbnail render failed for %s (%s): %s", output.id, output.file_path, e)
        return {"failed": str(e)[:MAX_FAILURE_CHARS], "at": _now()}
    name = save_thumbnail(output.workspace_id, output.id, png)
    logger.info("[F353] thumbnail %s drawn for %s", name, output.file_path)
    return {"file": name, "rendered_at": _now()}


def render_for_output(
    session_factory: Callable[[], Any],
    workspace_id: str | UUID,
    output_id: str | UUID,
    render: Callable[[bytes, str], bytes] = render_png_isolated,
) -> str:
    """Draw ``output_id``'s first page: "rendered", "failed", "skipped", "missing" or "no-file"."""
    with session_factory() as db:
        output = load_output(db, workspace_id, output_id)
    if output is None:
        return "missing"
    reason = skip_reason(output)
    if reason:
        logger.debug("[F353] no thumbnail for %s: %s", output.id, reason)
        return "skipped"
    result = _draw(output, render)
    if not result:
        return "no-file"
    with session_factory() as db:
        record_result(db, output, result)
    return "rendered" if "file" in result else "failed"
