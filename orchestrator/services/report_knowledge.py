"""F235 (night 6, the persona's #4; Gerard, 2 Oct: "approved work only"): a job
report becomes a Document, the knowledge Auto searches and quotes, only once its
work is approved.

Night 6: 122 of the workspace's 129 Documents were the product's own job reports.
Each was filed the moment its run finished, rejected rounds, failures and empty
answers included, each headed "Status: review" whether approved or not, and Auto
quoted them back as the owner's facts (friction 45, B38).

- A report for a ticket (``linked_task_ids``) waits for that ticket to be done:
  approved, or finished with review off. Then its current round is filed, headed
  approved, and an earlier round of the same ticket that was filed is removed, so
  the approved answer is the only copy.
- A playbook run's report (its "**Execution:**" line names the run) is filed when
  the run's card is done as the run ends; a run that failed or stopped for review
  is not.
- A report for no ticket (a heartbeat's) waits for the owner: a grade of 4 or 5,
  or an acknowledgement.

The wait never touches the report itself: it is on the Reports page from the start.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import text

logger = logging.getLogger(__name__)

APPROVING_GRADE = 4
REPORT_TAG = "report:{report_id}"
APPROVED_STATUS = "**Status:** approved"
_REVIEW_STATUS = re.compile(r"^\*\*Status:\*\* review[ \t]*$", re.M)
_EXECUTION_LINE = re.compile(r"^\*\*Execution:\*\* (?P<execution>[A-Za-z0-9_.:-]+)[ \t]*$", re.M)


# ── whether a report waits ───────────────────────────────────────────────────

def _uuid(value: Any) -> Optional[str]:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError):
        return None


def report_row(db: Any, workspace_id: Any, report_id: Any) -> Optional[Any]:
    """The report's row (the columns the wait reads), or None."""
    rid = _uuid(report_id)
    if rid is None:
        return None
    return db.execute(text(
        "SELECT id, title, agent_name, report_type, file_path, linked_task_ids, grade, acknowledged_at "
        "FROM agent_reports WHERE id = CAST(:id AS uuid) AND workspace_id = CAST(:ws AS uuid)"),
        {"id": rid, "ws": str(workspace_id)}).fetchone()


def linked_tasks(row: Any) -> List[int]:
    raw = row.linked_task_ids
    ids = raw if isinstance(raw, list) else json.loads(raw or "[]")
    return [int(i) for i in ids if str(i).isdigit()]


def _any_done(db: Any, workspace_id: Any, task_ids: Sequence[int]) -> bool:
    return db.execute(text(
        "SELECT 1 FROM board_tasks WHERE workspace_id = CAST(:ws AS uuid) AND id = ANY(:ids) "
        "AND status = 'done' LIMIT 1"), {"ws": str(workspace_id), "ids": list(task_ids)}).fetchone() is not None


def _playbook_card_done(db: Any, workspace_id: Any, content: str) -> Optional[bool]:
    """For a playbook run's report: whether the run's card is done. None for any
    other report."""
    found = _EXECUTION_LINE.search(content or "")
    if not found:
        return None
    return db.execute(text(
        "SELECT 1 FROM board_tasks WHERE workspace_id = CAST(:ws AS uuid) AND source_type = 'recipe' "
        "AND source_id = :execution AND status = 'done' LIMIT 1"),
        {"ws": str(workspace_id), "execution": found.group("execution")}).fetchone() is not None


def owner_approved(row: Any) -> bool:
    """A report for no ticket: graded 4 or 5, or acknowledged."""
    return row.acknowledged_at is not None or (row.grade or 0) >= APPROVING_GRADE


def held_for_approval(db: Any, workspace_id: Any, report_id: Any, content: str) -> bool:
    """Whether the report must wait before it becomes a Document. Read-only. A row
    this module does not know is filed as it always was."""
    row = report_row(db, workspace_id, report_id)
    if row is None:
        return False
    tasks = linked_tasks(row)
    if tasks:
        return not _any_done(db, workspace_id, tasks)
    card_done = _playbook_card_done(db, workspace_id, content)
    if card_done is not None:
        return not card_done
    return not owner_approved(row)


# ── filing it once approved ─────────────────────────────────────────────────

def filed_documents(db: Any, workspace_id: Any, report_ids: Sequence[Any]) -> Dict[str, List[int]]:
    """The Documents each report was filed as, by report id."""
    tags = [REPORT_TAG.format(report_id=r) for r in report_ids]
    if not tags:
        return {}
    rows = db.execute(text(
        "SELECT id, tags FROM documents WHERE workspace_id = CAST(:ws AS uuid) AND source_type = 'agent_output' "
        "AND CAST(tags AS text[]) && CAST(:tags AS text[])"), {"ws": str(workspace_id), "tags": tags}).fetchall()
    filed: Dict[str, List[int]] = {}
    for row in rows:
        for tag in set(row.tags or ()) & set(tags):
            filed.setdefault(tag.split(":", 1)[1], []).append(row.id)
    return filed


async def file_report(db: Any, workspace_id: Any, report_id: Any) -> Optional[int]:
    """File one report as a Document now that its work is approved, headed approved
    (its file on the Reports page too), through the flywheel's own ingest."""
    from services.knowledge_flywheel import ingest_agent_output
    from services.report_service import ReportService

    got = await ReportService(db, workspace_id).get_report(str(report_id))
    report = got.get("report") if got.get("success") else None
    if not report or report.get("content") is None:
        return None
    if held_for_approval(db, workspace_id, report_id, report["content"]):
        return None
    content = _REVIEW_STATUS.sub(APPROVED_STATUS, report["content"])
    if content != report["content"]:
        await _rewrite(workspace_id, report["file_path"], content)
    title, kind = report["title"], report.get("report_type") or "report"
    return await ingest_agent_output(
        db, workspace_id, content=content, filename=report["file_path"].rsplit("/", 1)[-1], source="report",
        source_id=str(report_id), title=title, description=f"Agent report ({kind}): {title}"[:500],
        agent_name=report.get("agent_name"), created_by=report.get("agent_name") or "agent",
        extra_tags=[REPORT_TAG.format(report_id=report_id)], report_type=kind)


async def _rewrite(workspace_id: Any, file_path: str, content: str) -> None:
    from core.workspace_client import WorkspaceClient

    written = await WorkspaceClient(str(workspace_id)).write_file(file_path, content)
    if not written.get("success", False):
        logger.warning("[F235] report file %s kept its review header: %s", file_path, written.get("error"))


async def file_report_if_ready(db: Any, workspace_id: Any, report_id: Any) -> Optional[int]:
    """The owner approved a report directly (a grade of 4 or 5, an acknowledgement):
    filed now, unless it is filed already or its ticket is not done."""
    if filed_documents(db, workspace_id, [report_id]):
        return None
    return await file_report(db, workspace_id, report_id)


def ticket_rounds(db: Any, workspace_id: Any, task: Any) -> Dict[str, List[str]]:
    """The reports of a ticket, newest first, split at its current run's start:
    ``current`` (this run's) and ``earlier`` (rounds before it)."""
    rows = db.execute(text(
        "SELECT id, created_at FROM agent_reports WHERE workspace_id = CAST(:ws AS uuid) "
        "AND linked_task_ids @> CAST(:one AS jsonb) ORDER BY created_at DESC, id DESC"),
        {"ws": str(workspace_id), "one": json.dumps([int(task.id)])}).fetchall()
    since = getattr(task, "started_at", None)
    current = [str(r.id) for r in rows if since is None or r.created_at >= since]
    return {"current": current, "earlier": [str(r.id) for r in rows if str(r.id) not in current]}


async def file_ticket_report(db: Any, workspace_id: Any, task: Any) -> Optional[int]:
    """The ticket is done, approved or finished with review off: its current round's
    newest report is filed, and every other round of it that was filed is removed.
    F305 (night 9): only in a workspace that opted in to filing; otherwise nothing is
    filed and a round filed before is left as it is."""
    from services.knowledge_flywheel import flywheel_enabled

    if not flywheel_enabled(db, workspace_id):
        return None
    rounds = ticket_rounds(db, workspace_id, task)
    keep = rounds["current"][:1]
    filed = filed_documents(db, workspace_id, rounds["current"] + rounds["earlier"])
    stale = [doc for report_id, docs in filed.items() if report_id not in keep for doc in docs]
    if stale:
        await asyncio.to_thread(remove_documents, db, workspace_id, stale)
    if not keep or keep[0] in filed:
        return None            # not written yet (finalize files it at creation), or filed then
    return await file_report(db, workspace_id, keep[0])


async def file_done_ticket(db: Any, workspace_id: Any, task: Any) -> None:
    """``file_ticket_report`` for the board's done paths, inside a savepoint: a
    failure is logged, never raised into the ticket's completion, and the caller's
    transaction stays usable."""
    try:
        with db.begin_nested():
            await file_ticket_report(db, workspace_id, task)
    except Exception:
        logger.exception("[F235] ticket %s is done; its report was not filed", getattr(task, "id", "?"))


async def file_owner_approved(db: Any, workspace_id: Any, report_id: Any) -> None:
    """``file_report_if_ready`` for the owner's own approval of a report (a grade of
    4 or 5, an acknowledgement): a failure is logged, never raised into the action."""
    try:
        await file_report_if_ready(db, workspace_id, report_id)
    except Exception:
        logger.exception("[F235] report %s was approved; it was not filed", report_id)


def remove_documents(db: Any, workspace_id: Any, document_ids: Sequence[int]) -> List[int]:
    """Remove Documents in the workspace as the Documents page's delete does
    (``services.document_removal``). Returns the ids removed."""
    from core.models.core import Document
    from services.document_removal import remove_document

    ws = UUID(str(workspace_id))
    removed = []
    for doc in db.query(Document).filter(Document.workspace_id == ws, Document.id.in_(list(document_ids))).all():
        remove_document(doc, str(ws))
        removed.append(doc.id)
    return removed


__all__ = [
    "APPROVING_GRADE", "file_done_ticket", "file_owner_approved", "file_report", "file_report_if_ready",
    "file_ticket_report",
    "filed_documents", "held_for_approval", "linked_tasks", "owner_approved", "remove_documents", "report_row",
    "ticket_rounds",
]
