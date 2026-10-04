"""F305 (night 9): "Add to knowledge" — the owner, not the platform, puts an agent's
approved answer or report into the knowledge base, as the owner's own document.

Agent outputs no longer enter the owner's documents on their own
(services/agent_output_scope.py). When the owner wants one there, this files it the
way an upload is filed: ``source_type`` empty (not agent_output), so every default
search finds it, and the Knowledge Graph reads it. A card must be approved (done);
a report for a ticket must have its ticket done. A report's Execution Metrics stay
in its file and are left out of the document.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

DONE = "done"
OWNER_TAG = "added-by-owner"
NOT_APPROVED = "{ticket} isn't approved yet. Approve it on the board, then add it to knowledge."
NO_ANSWER = "{ticket} has no answer to add."
REPORT_NOT_APPROVED = "This report's ticket isn't approved yet. Approve it on the board, then add the report."
REPORT_EMPTY = "This report has no text to add."
_METRICS = re.compile(r"\n## Execution Metrics\n.*?(?=\n## |\Z)", re.S)
_SLUG = re.compile(r"[^a-z0-9]+")
SLUG_CHARS = 60


class NotFound(Exception):
    """The card or report is not in this workspace."""


class Refused(Exception):
    """The card or report cannot be added; the message says why, for the owner."""


def _slug(title: str, fallback: str) -> str:
    return _SLUG.sub("-", (title or "").lower())[:SLUG_CHARS].strip("-") or fallback


async def _file_as_owners(workspace_id: Any, content: str, filename: str, title: str, tags: list,
                          by: str) -> int:
    """Upload ``content`` through the ingestion manager as the owner's document."""
    from api.documents import get_document_manager

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
            fh.write(content)
            tmp_path = fh.name
        return await get_document_manager(str(workspace_id)).upload_document(
            file_path=tmp_path, filename=filename, tags=[OWNER_TAG, *tags], created_by=by,
            description=f"Added to knowledge by the owner: {title}"[:500], source_type=None)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def approved_card(db: Any, workspace_id: Any, task_id: int) -> Any:
    """The card, when it is in the workspace, approved and has an answer; else raises."""
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_label

    task = db.query(BoardTask).filter(BoardTask.id == task_id,
                                      BoardTask.workspace_id == UUID(str(workspace_id))).first()
    if task is None:
        raise NotFound("Ticket not found")
    if task.status != DONE:
        raise Refused(NOT_APPROVED.format(ticket=ticket_label(task, capital=True)))
    if not (task.result or "").strip():
        raise Refused(NO_ANSWER.format(ticket=ticket_label(task, capital=True)))
    return task


async def add_card(db: Any, workspace_id: Any, task_id: int, *, by: str) -> Dict[str, Any]:
    """File an approved card's answer as the owner's document."""
    task = approved_card(db, workspace_id, task_id)
    content = f"# {task.title}\n\n{str(task.result).strip()}\n"
    doc_id = await _file_as_owners(workspace_id, content, f"{_slug(task.title, f'card-{task.id}')}.md", task.title,
                                   [f"card:{task.id}"], by)
    logger.info("[F305] the owner added card %s to knowledge as document %s", task.id, doc_id)
    return {"success": True, "document_id": doc_id, "task_id": task.id}


async def add_report(db: Any, workspace_id: Any, report_id: str, *, by: str) -> Dict[str, Any]:
    """File a report's text (without its Execution Metrics) as the owner's document. A
    report for a ticket waits for the ticket to be approved."""
    from services.report_knowledge import _any_done, linked_tasks, report_row
    from services.report_service import ReportService

    row = report_row(db, workspace_id, report_id)
    if row is None:
        raise NotFound("Report not found")
    tasks = linked_tasks(row)
    if tasks and not _any_done(db, workspace_id, tasks):
        raise Refused(REPORT_NOT_APPROVED)
    got = await ReportService(db, workspace_id).get_report(str(report_id))
    content = _METRICS.sub("", str(((got or {}).get("report") or {}).get("content") or "")).strip()
    if not content:
        raise Refused(REPORT_EMPTY)
    doc_id = await _file_as_owners(workspace_id, content + "\n", f"{_slug(row.title, 'report')}.md", row.title,
                                   [f"report-added:{row.id}"], by)
    logger.info("[F305] the owner added report %s to knowledge as document %s", row.id, doc_id)
    return {"success": True, "document_id": doc_id, "report_id": str(row.id)}


__all__ = ["NotFound", "Refused", "add_card", "add_report", "approved_card"]
