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

PRE-11 (Gerard, 7 Oct): a card and a report work the same way. Each copy carries its
source's tag (``card:<id>``, ``report-added:<id>``, ``deliverable-added:<id>``) beside
``added-by-owner``; adding again answers with that copy (``already_added``), removing
deletes it and leaves the source, and the board's and the Reports page's answers carry
``knowledge_document_id``. Who may: the routes' gate (core/auth/workspace_admin.py).
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union
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
# The tag that marks the owner's copy of each kind of source (F305's card and report
# tags kept as they were, so a copy added before PRE-11 shows as added).
CARD_TAG = "card:{source_id}"
REPORT_TAG = "report-added:{source_id}"
DELIVERABLE_TAG = "deliverable-added:{source_id}"
KNOWLEDGE_DOCUMENT_ID = "knowledge_document_id"
# The kinds of file the upload path reads the text of (modules/rag/ingestion/manager.py).
KNOWLEDGE_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt"})
NOT_A_DOCUMENT = "Only a document can be added to knowledge: a PDF, Word, Excel, CSV, Markdown or text file."
FILE_UNREADABLE = ("This document's file could not be read, so it was not added. "
                   "Try again, or download it and upload it on the Documents page.")
CARD_NOT_FOUND = "Ticket not found"
REPORT_NOT_FOUND = "Report not found"
DELIVERABLE_NOT_FOUND = "Deliverable not found"
ALREADY_IN_DOCUMENTS = "This file is already in your Documents."


class NotFound(Exception):
    """The card, report or Deliverable is not in this workspace."""


class Refused(Exception):
    """The card, report or Deliverable cannot be added; the message says why, for the owner."""


def _slug(title: str, fallback: str) -> str:
    return _SLUG.sub("-", (title or "").lower())[:SLUG_CHARS].strip("-") or fallback


def _tag(template: str, source_id: Any) -> str:
    return template.format(source_id=source_id)


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


# ── the owner's copies, whatever their source ─────────────────────────────

def knowledge_copies(db: Any, workspace_id: Any, source_ids: Sequence[Any],
                     template: str = DELIVERABLE_TAG) -> Dict[str, int]:
    """The owner's document each source (a Deliverable by default; a card or a report
    with its ``template``) was added as, by source id (absent when it was not added).
    Only a document the owner added counts: it carries ``added-by-owner`` too."""
    by_tag = {_tag(template, s): str(s) for s in source_ids}
    if not by_tag:
        return {}
    rows = db.execute(text(
        "SELECT id, tags FROM documents WHERE workspace_id = CAST(:ws AS uuid) "
        "AND CAST(tags AS text[]) && CAST(:tags AS text[]) "
        "AND CAST(:owner AS text) = ANY(CAST(tags AS text[])) ORDER BY id"),
        {"ws": str(workspace_id), "tags": list(by_tag), "owner": OWNER_TAG}).fetchall()
    return {by_tag[tag]: row.id for row in rows for tag in (row.tags or ()) if tag in by_tag}


def with_knowledge_ids(db: Any, workspace_id: Any, items: Sequence[Dict[str, Any]],
                       template: str) -> List[Dict[str, Any]]:
    """Each item (an answer's dict with an ``id``) with its ``knowledge_document_id``:
    the owner's document it was added as, or None. One query for all of them."""
    copies = knowledge_copies(db, workspace_id, [item["id"] for item in items], template) if items else {}
    return [{**item, KNOWLEDGE_DOCUMENT_ID: copies.get(str(item["id"]))} for item in items]


def with_knowledge_state(db: Any, workspace_id: Any, answer: Dict[str, Any], *, many: str = "deliverables",
                         one: str = "deliverable", template: str = DELIVERABLE_TAG) -> Dict[str, Any]:
    """A list answer (``answer[many]``) or a detail answer (``answer[one]``), each
    item with its ``knowledge_document_id``. Deliverables by default; the Reports
    page passes ``reports`` / ``report`` and ``REPORT_TAG``."""
    listed, single = answer.get(many), answer.get(one)
    shown = [*(listed or []), *([single] if isinstance(single, dict) else [])]
    if not shown:
        return answer
    marked = with_knowledge_ids(db, workspace_id, shown, template)
    updated = dict(answer)
    if listed is not None:
        updated[many] = marked[:len(listed)]
    if isinstance(single, dict):
        updated[one] = marked[-1]
    return updated


async def _file_once(db: Any, workspace_id: Any, template: str, source_id: Any,
                     file_it: Callable[[], Awaitable[int]]) -> Dict[str, Any]:
    """Add a source once: the copy already filed (``already_added``), else the new
    document ``file_it`` files, which must be the one carrying the source's tag."""
    sid = str(source_id)
    filed = knowledge_copies(db, workspace_id, [sid], template).get(sid)
    if filed is not None:
        return {"document_id": filed, "already_added": True}
    doc_id = await file_it()
    if knowledge_copies(db, workspace_id, [sid], template).get(sid) != doc_id:
        # The upload path answers with an earlier document of the same bytes, untagged.
        logger.warning("[F305] %s matched existing document %s; not added again", _tag(template, sid), doc_id)
        raise Refused(ALREADY_IN_DOCUMENTS)
    return {"document_id": doc_id, "already_added": False}


def _remove_copies(db: Any, workspace_id: Any, template: str, source_id: Any) -> int:
    """Delete the owner's copies of one source (file, chunks and vectors, as the
    Documents page deletes). Returns how many were removed; none is not an error."""
    from core.models.core import Document
    from services.document_removal import remove_document

    ws = UUID(str(workspace_id))
    copies = db.query(Document).filter(Document.workspace_id == ws, Document.tags.any(_tag(template, source_id)),
                                       Document.tags.any(OWNER_TAG)).all()
    for document in copies:
        remove_document(document, str(ws))
    return len(copies)


# ── a card (F305) ───────────────────────────────────────────────────────────

def _card(db: Any, workspace_id: Any, task_id: int) -> Any:
    from core.models.core import BoardTask

    task = db.query(BoardTask).filter(BoardTask.id == task_id,
                                      BoardTask.workspace_id == UUID(str(workspace_id))).first()
    if task is None:
        raise NotFound(CARD_NOT_FOUND)
    return task


def _approved_answer(task: Any) -> str:
    """The card's answer, when it is approved and has one; else Refused."""
    from services.ticket_numbers import ticket_label

    if task.status != DONE:
        raise Refused(NOT_APPROVED.format(ticket=ticket_label(task, capital=True)))
    if not (task.result or "").strip():
        raise Refused(NO_ANSWER.format(ticket=ticket_label(task, capital=True)))
    return str(task.result).strip()


async def add_card(db: Any, workspace_id: Any, task_id: int, *, by: str) -> Dict[str, Any]:
    """File an approved card's answer as the owner's document, once."""
    task = _card(db, workspace_id, task_id)

    async def file_it() -> int:
        content = f"# {task.title}\n\n{_approved_answer(task)}\n"
        return await _file_as_owners(workspace_id, content, f"{_slug(task.title, f'card-{task.id}')}.md",
                                     task.title, [_tag(CARD_TAG, task.id)], by)

    added = await _file_once(db, workspace_id, CARD_TAG, task.id, file_it)
    logger.info("[F305] the owner added card %s to knowledge as document %s", task.id, added["document_id"])
    return {"success": True, "task_id": task.id, **added}


def remove_card(db: Any, workspace_id: Any, task_id: int) -> Dict[str, Any]:
    """Delete the owner's copy of a card's answer; the card stays."""
    task = _card(db, workspace_id, task_id)
    removed = _remove_copies(db, workspace_id, CARD_TAG, task.id)
    logger.info("[F305] the owner removed card %s from knowledge (%d document(s))", task.id, removed)
    return {"success": True, "task_id": task.id, "removed": removed}


# ── a report (F305) ─────────────────────────────────────────────────────────

def _report(db: Any, workspace_id: Any, report_id: Any) -> Any:
    from services.report_knowledge import report_row

    row = report_row(db, workspace_id, report_id)
    if row is None:
        raise NotFound(REPORT_NOT_FOUND)
    return row


async def _report_text(db: Any, workspace_id: Any, row: Any) -> str:
    """The report's text without its Execution Metrics. A report for a ticket waits
    for the ticket to be approved."""
    from services.report_knowledge import _any_done, linked_tasks
    from services.report_service import ReportService

    tasks = linked_tasks(row)
    if tasks and not _any_done(db, workspace_id, tasks):
        raise Refused(REPORT_NOT_APPROVED)
    got = await ReportService(db, workspace_id).get_report(str(row.id))
    content = _METRICS.sub("", str(((got or {}).get("report") or {}).get("content") or "")).strip()
    if not content:
        raise Refused(REPORT_EMPTY)
    return content


async def add_report(db: Any, workspace_id: Any, report_id: str, *, by: str) -> Dict[str, Any]:
    """File a report's text as the owner's document, once."""
    row = _report(db, workspace_id, report_id)

    async def file_it() -> int:
        content = await _report_text(db, workspace_id, row)
        return await _file_as_owners(workspace_id, content + "\n", f"{_slug(row.title, 'report')}.md", row.title,
                                     [_tag(REPORT_TAG, row.id)], by)

    added = await _file_once(db, workspace_id, REPORT_TAG, row.id, file_it)
    logger.info("[F305] the owner added report %s to knowledge as document %s", row.id, added["document_id"])
    return {"success": True, "report_id": str(row.id), **added}


def remove_report(db: Any, workspace_id: Any, report_id: Any) -> Dict[str, Any]:
    """Delete the owner's copy of a report; the report stays."""
    row = _report(db, workspace_id, report_id)
    removed = _remove_copies(db, workspace_id, REPORT_TAG, row.id)
    logger.info("[F305] the owner removed report %s from knowledge (%d document(s))", row.id, removed)
    return {"success": True, "report_id": str(row.id), "removed": removed}


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

    async def file_it() -> int:
        extension = os.path.splitext(row.file_name or row.file_path or "")[1].lower()
        if extension not in KNOWLEDGE_EXTENSIONS:
            raise Refused(NOT_A_DOCUMENT)
        data = await deliverable_bytes(workspace_id, row.storage_type, row.file_path)
        if not data:
            raise Refused(FILE_UNREADABLE)
        tags = [_tag(DELIVERABLE_TAG, did), *party_tags(row.extra)]
        return await _file_as_owners(workspace_id, data, f"{_slug(row.title, 'document')}{extension}", row.title,
                                     tags, by, _about(row))

    added = await _file_once(db, workspace_id, DELIVERABLE_TAG, did, file_it)
    logger.info("[F354] the owner added Deliverable %s to knowledge as document %s", did, added["document_id"])
    return {"success": True, "deliverable_id": did, **added}


def remove_deliverable(db: Any, workspace_id: Any, deliverable_id: Any) -> Dict[str, Any]:
    """Delete the owner's document a Deliverable was added as (its file, chunks and
    vectors). The Deliverable itself stays. Nothing to remove is not an error."""
    row = _deliverable(db, workspace_id, deliverable_id)
    removed = _remove_copies(db, workspace_id, DELIVERABLE_TAG, row.id)
    logger.info("[F354] the owner removed Deliverable %s from knowledge (%d document(s))", row.id, removed)
    return {"success": True, "deliverable_id": str(row.id), "removed": removed}


__all__ = ["CARD_TAG", "DELIVERABLE_TAG", "NotFound", "REPORT_TAG", "Refused", "add_card", "add_deliverable",
           "add_report", "knowledge_copies", "remove_card", "remove_deliverable", "remove_report",
           "with_knowledge_ids", "with_knowledge_state"]
