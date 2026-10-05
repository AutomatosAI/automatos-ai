"""F353 (issue #947): draw the first pages of the documents made before F353, at a set rate.

Run by hand (``scripts/backfill_f353_document_thumbnails.py``), never at boot.
It walks ``v_workspace_outputs`` oldest first and draws each document, sheet and
report (not heartbeat reports) that has no picture yet:

* a ``deliverables`` row with no ``extra.thumbnail`` (one that failed before is
  skipped unless ``retry_failed``);
* a report with no stored picture (``store.load_thumbnail`` finds none).

Each Deliverable goes through the same ``job.render_for_output`` as a new one, and
at most ``per_minute`` renders are started a minute (a sleep between them), so a
large workspace never takes the CPU or the workspace worker from live requests.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from sqlalchemy import text

from modules.documents.thumbnails.eligibility import HEARTBEAT, REPORT, THUMBNAIL_ARTIFACT_TYPES, wants_thumbnail
from modules.documents.thumbnails.job import render_for_output
from modules.documents.thumbnails.store import load_thumbnail

logger = logging.getLogger(__name__)

DEFAULT_PER_MINUTE = 20
SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class Candidate:
    """One Deliverable the backfill may draw."""

    id: str
    workspace_id: str
    artifact_type: str
    file_path: str
    extra: Dict[str, Any]


def find_candidates(db: Any, workspace_id: Optional[str] = None, limit: Optional[int] = None) -> List[Candidate]:
    """Live documents, sheets and non-heartbeat reports, oldest first (optionally one workspace)."""
    rows = db.execute(
        text("""
            SELECT o.id, o.workspace_id, o.artifact_type, o.file_path, o.extra
            FROM v_workspace_outputs o
            WHERE o.deleted_at IS NULL
              AND o.artifact_type = ANY(:types)
              AND o.source_type <> :heartbeat
              AND (CAST(:ws AS uuid) IS NULL OR o.workspace_id = CAST(:ws AS uuid))
            ORDER BY o.created_at ASC
            LIMIT :limit
        """),
        {"types": sorted(THUMBNAIL_ARTIFACT_TYPES), "heartbeat": HEARTBEAT, "ws": workspace_id,
         "limit": limit},
    ).fetchall()
    return [
        Candidate(str(r.id), str(r.workspace_id), r.artifact_type, r.file_path or "",
                  r.extra if isinstance(r.extra, dict) else {})
        for r in rows if wants_thumbnail(r.artifact_type, r.file_path)
    ]


def needs_drawing(candidate: Candidate, retry_failed: bool) -> bool:
    """No picture yet (and not a known failure, unless retrying failures)."""
    if candidate.artifact_type == REPORT:
        return load_thumbnail(candidate.workspace_id, candidate.id) is None
    recorded = candidate.extra.get("thumbnail")
    if not isinstance(recorded, dict):
        return True
    return retry_failed and not recorded.get("file")


def _draw_one(render_one: Callable[..., str], session_factory: Callable[[], Any], candidate: Candidate) -> str:
    """One render; an unexpected error is logged and counted, and the backfill goes on."""
    try:
        return render_one(session_factory, candidate.workspace_id, candidate.id)
    except Exception:
        logger.exception("[F353] backfill could not draw %s (%s)", candidate.id, candidate.file_path)
        return "error"


def backfill(
    session_factory: Callable[[], Any],
    candidates: Iterable[Candidate],
    *,
    per_minute: int = DEFAULT_PER_MINUTE,
    retry_failed: bool = False,
    dry_run: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    render_one: Callable[..., str] = render_for_output,
) -> Dict[str, int]:
    """Draw each candidate that needs it, at most ``per_minute`` a minute. Counts per outcome."""
    gap_s = SECONDS_PER_MINUTE / max(1, per_minute)
    outcomes: Counter = Counter()
    started = 0
    for candidate in candidates:
        if not needs_drawing(candidate, retry_failed):
            outcomes["has-picture"] += 1
            continue
        if dry_run:
            outcomes["would-draw"] += 1
            continue
        if started:
            sleep(gap_s)
        started += 1
        outcomes[_draw_one(render_one, session_factory, candidate)] += 1
    logger.info("[F353] thumbnail backfill: %s", dict(outcomes))
    return dict(outcomes)
