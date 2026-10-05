"""F305 (night 9): "Add to knowledge" — the owner, not the platform, puts an agent's
approved answer or report into the knowledge base, as the owner's own document.

Agent outputs no longer enter the owner's documents on their own
(services/agent_output_scope.py). When the owner wants one there, this files it the
way an upload is filed: ``source_type`` empty (not agent_output), so every default
search finds it, and the Knowledge Graph reads it. A card must be approved (done);
a report for a ticket must have its ticket done. A report's Execution Metrics stay
in its file and are left out of the document.

F354 (5 Oct): the owner adds a Deliverable too. Gerard: "maybe for invoices, letters, a
user might want to document these, then you can ask an agent later to pull all files
related to customer x, or send invoice to client y… but we don't want all system
reports". So it is one document at a time, the owner's choice, never automatic. The
document's own file (the PDF, DOCX, XLSX, CSV, Markdown or text) goes through the
upload path, which reads its text as it reads any upload; its description and tags
name who it is for (``extra.parties``, modules/documents/deliverable_extra.py) and the
Deliverable's file, so a search by the customer's name lists it. Tagged
``deliverable-added:<id>``: adding it again answers with that copy, and removing it
deletes that copy (the Documents page's own delete, services/document_removal.py).
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Any, Dict, Optional, Sequence, Union
from uuid import UUID

from sqlalchemy import text

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
DESCRIPTION_CHARS = 500
DELIVERABLE_TAG = "deliverable-added:{deliverable_id}"
KNOWLEDGE_DOCUMENT_ID = "knowledge_document_id"
# The kinds of file the upload path reads the text of (modules/rag/ingestion/manager.py).
KNOWLEDGE_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt"})
NOT_A_DOCUMENT = "Only a document can be added to knowledge: a PDF, Word, Excel, CSV, Markdown or text file."
FILE_UNREADABLE = ("This document's file could not be read, so it was not added. "
                   "Try again, or download it and upload it on the Documents page.")
DELIVERABLE_NOT_FOUND = "Deliverable not found"
ALREADY_IN_DOCUMENTS = "This file is already in your Documents."


class NotFound(Exception):
    """The card, report or Deliverable is not in this workspace."""


class Refused(Exception):
    """The card, report or Deliverable cannot be added; the message says why, for the owner."""


def _slug(title: str, fallback: str) -> str:
    return _SLUG.sub("-", (title or "").lower())[:SLUG_CHARS].strip("-") or fallback


async def _file_as_owners(workspace_id: Any, content: Union[str, bytes], filename: str, title: str, tags: list,
                          by: str, details: str = "") -> int:
    """Upload ``content`` (text, or a file's bytes; ``filename``'s extension says which
    kind, as an upload's does) through the ingestion manager as the owner's document."""
    from api.documents import get_document_manager

    tmp_path: Optional[str] = None
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    try:
        with tempfile.NamedTemporaryFile("wb", suffix=os.path.splitext(filename)[1] or ".md", delete=False) as fh:
            fh.write(data)
            tmp_path = fh.name
        return await get_document_manager(str(workspace_id)).upload_document(
            file_path=tmp_path, filename=filename, tags=[OWNER_TAG, *tags], created_by=by,
            description=f"Added to knowledge by the owner: {title}{details}"[:DESCRIPTION_CHARS], source_type=None)
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


# ── a Deliverable (F354) ────────────────────────────────────────────────────

def _deliverable(db: Any, workspace_id: Any, deliverable_id: Any) -> Any:
    """The Deliverable's row when it is in the workspace and not deleted; else NotFound."""
    try:
        did = str(UUID(str(deliverable_id)))
    except (TypeError, ValueError) as exc:
        raise NotFound(DELIVERABLE_NOT_FOUND) from exc
    row = db.execute(text(
        "SELECT id, title, file_path, file_name, storage_type, extra FROM deliverables "
        "WHERE id = CAST(:id AS uuid) AND workspace_id = CAST(:ws AS uuid) AND deleted_at IS NULL"),
        {"id": did, "ws": str(workspace_id)}).fetchone()
    if row is None:
        raise NotFound(DELIVERABLE_NOT_FOUND)
    return row


def knowledge_copies(db: Any, workspace_id: Any, deliverable_ids: Sequence[Any]) -> Dict[str, int]:
    """The owner's document each Deliverable was added as, by Deliverable id (absent when
    it was not added)."""
    by_tag = {DELIVERABLE_TAG.format(deliverable_id=d): str(d) for d in deliverable_ids}
    if not by_tag:
        return {}
    rows = db.execute(text(
        "SELECT id, tags FROM documents WHERE workspace_id = CAST(:ws AS uuid) "
        "AND CAST(tags AS text[]) && CAST(:tags AS text[]) ORDER BY id"),
        {"ws": str(workspace_id), "tags": list(by_tag)}).fetchall()
    return {by_tag[tag]: row.id for row in rows for tag in (row.tags or ()) if tag in by_tag}


def with_knowledge_state(db: Any, workspace_id: Any, answer: Dict[str, Any]) -> Dict[str, Any]:
    """The Deliverables list's (or one Deliverable's) answer, each Deliverable with its
    ``knowledge_document_id``: the owner's document it was added as, or None."""
    many, one = answer.get("deliverables"), answer.get("deliverable")
    shown = [*(many or []), *([one] if isinstance(one, dict) else [])]
    if not shown:
        return answer
    copies = knowledge_copies(db, workspace_id, [d["id"] for d in shown])

    def marked(item: Dict[str, Any]) -> Dict[str, Any]:
        return {**item, KNOWLEDGE_DOCUMENT_ID: copies.get(str(item["id"]))}

    updated = dict(answer)
    if many is not None:
        updated["deliverables"] = [marked(d) for d in many]
    if isinstance(one, dict):
        updated["deliverable"] = marked(one)
    return updated


def _about(row: Any) -> str:
    """The description's tail: who the document is for, and where its file is."""
    from modules.documents.deliverable_extra import party_lines

    return "".join(f". {line}" for line in [*party_lines(row.extra), f"Deliverable file: {row.file_path}"])


async def add_deliverable(db: Any, workspace_id: Any, deliverable_id: Any, *, by: str) -> Dict[str, Any]:
    """File a Deliverable's document (its own file) as the owner's document, once: adding
    it again answers with the copy already filed."""
    from modules.documents.deliverable_extra import party_tags
    from services.deliverable_file import deliverable_bytes

    row = _deliverable(db, workspace_id, deliverable_id)
    did = str(row.id)
    filed = knowledge_copies(db, workspace_id, [did]).get(did)
    if filed is not None:
        return {"success": True, "document_id": filed, "deliverable_id": did, "already_added": True}
    extension = os.path.splitext(row.file_name or row.file_path or "")[1].lower()
    if extension not in KNOWLEDGE_EXTENSIONS:
        raise Refused(NOT_A_DOCUMENT)
    data = await deliverable_bytes(workspace_id, row.storage_type, row.file_path)
    if not data:
        raise Refused(FILE_UNREADABLE)
    tags = [DELIVERABLE_TAG.format(deliverable_id=did), *party_tags(row.extra)]
    doc_id = await _file_as_owners(workspace_id, data, f"{_slug(row.title, 'document')}{extension}", row.title,
                                   tags, by, _about(row))
    if knowledge_copies(db, workspace_id, [did]).get(did) != doc_id:
        # The upload path answers with an earlier document of the same bytes, untagged.
        logger.warning("[F354] Deliverable %s matched existing document %s; not added again", did, doc_id)
        raise Refused(ALREADY_IN_DOCUMENTS)
    logger.info("[F354] the owner added Deliverable %s to knowledge as document %s", did, doc_id)
    return {"success": True, "document_id": doc_id, "deliverable_id": did, "already_added": False}


def remove_deliverable(db: Any, workspace_id: Any, deliverable_id: Any) -> Dict[str, Any]:
    """Delete the owner's document a Deliverable was added as (its file, chunks and
    vectors). The Deliverable itself stays. Nothing to remove is not an error."""
    from core.models.core import Document
    from services.document_removal import remove_document

    row = _deliverable(db, workspace_id, deliverable_id)
    tag = DELIVERABLE_TAG.format(deliverable_id=str(row.id))
    copies = db.query(Document).filter(Document.workspace_id == UUID(str(workspace_id)),
                                       Document.tags.any(tag)).all()
    for document in copies:
        remove_document(document, str(workspace_id))
    logger.info("[F354] the owner removed Deliverable %s from knowledge (%d document(s))", row.id, len(copies))
    return {"success": True, "deliverable_id": str(row.id), "removed": len(copies)}


__all__ = ["NotFound", "Refused", "add_card", "add_deliverable", "add_report", "approved_card",
           "knowledge_copies", "remove_deliverable", "with_knowledge_state"]
