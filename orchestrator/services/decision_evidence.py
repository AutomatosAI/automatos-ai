"""PRD-248 tuning (6 Oct): the evidence a shadow judgement reads, gathered off the request path.

``core.llm.decisions.evidence`` shapes the evidence and does no I/O. This module reads
it: a session's deliverables' first lines from the workspace volume, and a report's
linked tickets from the database. Both run in a worker thread inside the shadow task,
after the platform's own decision is made, so neither ever delays a result or a
report. Every read is fail-soft: a file or a row that can't be read is left out.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from config import config

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".csv", ".html", ".htm", ".json"})
FIRST_LINES = 3
READ_MAX_BYTES = 4096
TICKETS_MAX = 5
BRIEF_MAX_CHARS = 600
_TAG = re.compile(r"<[^>]+>")


def day_label(when: Optional[datetime]) -> str:
    """A date as the evidence writes it: "17 Sep 2026"; "" for none."""
    if when is None:
        return ""
    return f"{when.day} {when.strftime('%b')} {when.year}"


def today_label() -> str:
    return day_label(datetime.now(timezone.utc))


def first_lines(path: Path, *, lines: int = FIRST_LINES) -> str:
    """The first non-blank lines of a text file (tags stripped from HTML); "" for any other
    file or a file that can't be read."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(READ_MAX_BYTES)
    except OSError:
        return ""
    if path.suffix.lower() in (".html", ".htm"):
        head = _TAG.sub(" ", head)
    kept = [line.strip() for line in head.splitlines() if line.strip()]
    return "\n".join(kept[:lines])


def session_deliverables(workspace_id: Any, deliverables: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """A session's registered deliverables as name, type and first lines (read from the
    workspace volume; a file in the projects folder is not mounted here, so its name only)."""
    volume = Path(config.WORKSPACE_VOLUME_PATH) / str(workspace_id)
    out: List[Dict[str, str]] = []
    for item in deliverables:
        rel = str(item.get("file_path") or "")
        out.append({
            "name": str(item.get("title") or rel.rsplit("/", 1)[-1]),
            "type": str(item.get("artifact_type") or ""),
            "first_lines": first_lines(volume / rel) if rel else "",
        })
    return out


def linked_tickets(workspace_id: Any, task_ids: Sequence[Any]) -> List[Dict[str, str]]:
    """The tickets a report links to, as title, brief and the day they were asked for.
    Read with its own session, scoped to the report's workspace."""
    ids = [int(i) for i in task_ids if str(i).isdigit()][:TICKETS_MAX]
    if not ids:
        return []
    from core.database.database import get_db_session
    from core.models.core import BoardTask

    with get_db_session() as db:
        rows = (
            db.query(BoardTask.title, BoardTask.description, BoardTask.created_at)
            .filter(BoardTask.workspace_id == workspace_id, BoardTask.id.in_(ids))
            .all()
        )
        return [
            {"title": str(title or ""), "brief": str(description or "")[:BRIEF_MAX_CHARS], "asked_on": day_label(created)}
            for title, description, created in rows
        ]


async def shadow_session_end(engine: Any, *, deliverable_refs: Sequence[Mapping[str, Any]], **values: Any) -> Any:
    """The session_end judgement, with the deliverables' first lines read in a thread first."""
    from core.llm.decisions import judgements

    try:
        made = await asyncio.to_thread(session_deliverables, values.get("workspace_id"), deliverable_refs)
    except Exception:  # noqa: BLE001 — the judgement still runs on what the platform holds
        logger.debug("[decision] session deliverables unreadable", exc_info=True)
        made = [{"name": str(d.get("title") or ""), "type": str(d.get("artifact_type") or "")} for d in deliverable_refs]
    return await judgements.shadow_session_end(engine, deliverables=made, **values)


async def shadow_report_triage(engine: Any, *, linked_task_ids: Sequence[Any], **values: Any) -> Any:
    """The report_triage judgement, with the report's linked tickets read in a thread first."""
    from core.llm.decisions import judgements

    try:
        tickets = await asyncio.to_thread(linked_tickets, values.get("workspace_id"), linked_task_ids)
    except Exception:  # noqa: BLE001 — the judgement still runs without the tickets
        logger.debug("[decision] linked tickets unreadable", exc_info=True)
        tickets = []
    return await judgements.shadow_report_triage(engine, linked_tickets=tickets, **values)
