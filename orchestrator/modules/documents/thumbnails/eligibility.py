"""F353 (issue #947): which Deliverables get a first-page picture, and the URL a card asks.

Light on purpose: ``services/deliverable_service.py`` imports it on every list
(``thumbnail_url`` on each Deliverable) and on every register (the queue check).

* Drawn: documents (.pdf, .docx, .txt, .md), spreadsheets (.xlsx, .csv, .tsv)
  and reports (.md), except heartbeat reports.
* Not drawn: images and videos (their cards already show them), slides, code,
  archives, audio, and the document formats with no reader here (.doc, .odt,
  .rtf, .xls, .ods, .pptx).
"""
from __future__ import annotations

import os
from typing import Any, Optional
from uuid import UUID

from modules.documents.thumbnails.render import SUPPORTED_EXTENSIONS

THUMBNAIL_ARTIFACT_TYPES = frozenset({"document", "spreadsheet", "report"})
REPORT = "report"
HEARTBEAT = "heartbeat"
THUMBNAIL_URL = "/api/deliverables/{id}/thumbnail"


def wants_thumbnail(artifact_type: Optional[str], file_path: Optional[str]) -> bool:
    """A type and an extension whose first page is drawn."""
    ext = os.path.splitext(file_path or "")[1].lower()
    return artifact_type in THUMBNAIL_ARTIFACT_TYPES and ext in SUPPORTED_EXTENSIONS


def _recorded_file(extra: Any) -> Optional[str]:
    """``extra.thumbnail.file`` when a picture was recorded."""
    thumbnail = extra.get("thumbnail") if isinstance(extra, dict) else None
    return thumbnail.get("file") if isinstance(thumbnail, dict) else None


def _is_uuid(value: Any) -> bool:
    try:
        UUID(str(value))
    except ValueError:
        return False
    return True


def thumbnail_url_for(row: Any) -> Optional[str]:
    """The card's picture URL for a ``v_workspace_outputs`` row, or None (the card keeps its icon).

    A ``deliverables`` row has one once ``extra.thumbnail.file`` is recorded
    (``job.record_result``). A report has no column to record it in, so a
    non-heartbeat report always offers one and the card falls back to its icon
    when the route answers 404.
    """
    artifact_type = getattr(row, "artifact_type", None)
    if artifact_type == REPORT:
        drawable = getattr(row, "source_type", None) != HEARTBEAT
    else:
        drawable = bool(_recorded_file(getattr(row, "extra", None)))
    if not drawable or not wants_thumbnail(artifact_type, getattr(row, "file_path", None)):
        return None
    return THUMBNAIL_URL.format(id=row.id) if _is_uuid(row.id) else None
