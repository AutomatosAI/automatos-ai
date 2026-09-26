"""PRD-251 S1.6: a post carries the credit its media's music asks for.

Every rendered file records the music it mixed on its Deliverable
(``extra.music``, ``core/music_credit.py``): a Socials render
(``modules/socials/render.py``) and ``generate_document`` with a social format
alike. A post's ``media`` names Deliverables, as the ids a save sets or as the
file records a render writes. :func:`media_credits` reads the credit lines
those Deliverables ask for, in the caller's workspace only, and the API appends
them to the post's copy on every save (``service.with_credits``). So a post
that attaches a video with a CC BY track carries the track's credit, however
the video was made, and an edit that drops the line gets it back.

``deliverables`` has no ORM model: it is read with raw SQL, as
``services/deliverable_service.py`` writes it. A malformed id is skipped before
the query, so an id Postgres cannot parse never reaches it.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

import sqlalchemy as sa

from core.music_credit import deliverable_credit

# A post names a handful of files; more than this is not media to credit.
MAX_MEDIA_IDS = 100

_CREDITS = sa.text(
    """
    SELECT d.id, d.extra
      FROM deliverables d
     WHERE d.workspace_id = :workspace_id
       AND d.deleted_at IS NULL
       AND d.id IN :ids
    """
).bindparams(sa.bindparam("ids", expanding=True))


def _uuid(value: Any) -> Optional[str]:
    if not isinstance(value, (str, UUID)):
        return None
    try:
        return str(UUID(str(value)))
    except ValueError:
        return None


def media_deliverable_ids(media: Any) -> List[str]:
    """The Deliverable ids ``media`` names (``{aspect: [id | file record]}``), in order, each once."""
    ids: List[str] = []
    if not isinstance(media, Mapping):
        return ids
    for entries in media.values():
        for entry in entries if isinstance(entries, (list, tuple)) else ():
            ident = _uuid(entry.get("deliverable_id") if isinstance(entry, Mapping) else entry)
            if ident and ident not in ids:
                ids.append(ident)
    return ids[:MAX_MEDIA_IDS]


def _extra(raw: Any) -> Dict[str, Any]:
    """``deliverables.extra``: a dict on Postgres (JSONB), text on SQLite."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)) and raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def media_credits(db: Any, workspace_id: Any, media: Any) -> List[str]:
    """The credit lines the music of ``media``'s Deliverables asks for, in media order, each once."""
    ids = media_deliverable_ids(media)
    if not ids:
        return []
    rows = db.execute(_CREDITS, {"workspace_id": str(workspace_id), "ids": ids}).fetchall()
    extras = {_uuid(str(row.id)): _extra(row.extra) for row in rows}
    lines: List[str] = []
    for ident in ids:
        line = deliverable_credit(extras.get(ident))
        if line and line not in lines:
            lines.append(line)
    return lines
