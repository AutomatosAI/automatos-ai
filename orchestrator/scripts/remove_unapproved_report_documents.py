"""
F235 one-off (Gerard, 2 Oct): remove the job-report Documents that "approved work
only" would never have filed.

Before F235 every job report became a Document the moment its run finished. Night 6
left 122 of the workspace's 129 Documents as the product's own reports: rejected
rounds, failures and empty answers included. From F235 on, a report is filed only
once its work is approved (``services.report_knowledge``). This removes, for one
workspace, the report Documents that rule keeps out:

- a report whose ticket is not done (in review, failed, sent back, cancelled);
- an earlier round of a done ticket (the approved round is the only copy);
- a playbook run's report whose card is not done;
- a report for no ticket that the owner has not graded 4 or 5 nor acknowledged.

Only Documents filed from a report are looked at (``source_type`` agent_output with a
``report:<id>`` tag); the owner's own uploads are never touched. A report Document
with no report row behind it is kept and listed.

Dry run by default: it lists what it would remove. ``--apply`` removes them through
the Documents page's own delete (file, chunks, row and vectors). Run per workspace by
TESTER, never at boot:

    docker exec -i -w /app automatos_backend python - --workspace-id <uuid> [--apply] \\
        < orchestrator/scripts/remove_unapproved_report_documents.py

Prints one JSON line: what was kept, removed (or would be) and why, and what is unknown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, List, Tuple
from uuid import UUID

KEEP, REMOVE, UNKNOWN = "keep", "remove", "unknown"
EXIT_OK, EXIT_ERROR = 0, 1
REPORT_TAG_PREFIX = "report:"


def _ticket_verdict(db: Any, workspace_id: UUID, report_id: str, task_ids: List[int]) -> Tuple[str, str]:
    """A ticket's report stays only as the newest report of a done ticket's current run."""
    from core.models.core import BoardTask
    from services.report_knowledge import ticket_rounds

    tasks = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.id.in_(task_ids)).all()
    for task in tasks:
        if task.status != "done":
            continue
        if report_id in ticket_rounds(db, workspace_id, task)["current"][:1]:
            return KEEP, f"the approved round of ticket {task.id}"
        return REMOVE, f"an earlier round of ticket {task.id}"
    return REMOVE, "its ticket is not done"


async def verdict(db: Any, workspace_id: UUID, report_id: str) -> Tuple[str, str]:
    """Whether "approved work only" keeps a filed report, and why."""
    from services.report_knowledge import held_for_approval, linked_tasks, report_row
    from services.report_service import ReportService

    row = report_row(db, workspace_id, report_id)
    if row is None:
        return UNKNOWN, "no report row behind it"
    tasks = linked_tasks(row)
    if tasks:
        return _ticket_verdict(db, workspace_id, str(row.id), tasks)
    got = await ReportService(db, workspace_id).get_report(str(row.id))
    content = ((got.get("report") or {}).get("content")) or ""
    if held_for_approval(db, workspace_id, str(row.id), content):
        return REMOVE, "not approved: no done card, grade of 4 or 5, or acknowledgement"
    return KEEP, "approved"


async def review(db: Any, workspace_id: UUID) -> Dict[str, List[Dict[str, Any]]]:
    """Every report Document in the workspace, sorted into keep, remove and unknown."""
    from core.models.core import Document

    plan: Dict[str, List[Dict[str, Any]]] = {KEEP: [], REMOVE: [], UNKNOWN: []}
    documents = db.query(Document).filter(Document.workspace_id == workspace_id,
                                          Document.source_type == "agent_output").all()
    for doc in documents:
        reports = [t[len(REPORT_TAG_PREFIX):] for t in (doc.tags or ()) if str(t).startswith(REPORT_TAG_PREFIX)]
        if not reports:
            continue
        kind, why = await verdict(db, workspace_id, reports[0])
        plan[kind].append({"document": doc.id, "file": doc.filename, "report": reports[0], "why": why})
    return plan


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="remove the report Documents approved-work-only keeps out")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--apply", action="store_true", help="remove them (default: list them only)")
    args = parser.parse_args(argv)
    try:
        workspace_id = UUID(args.workspace_id)
    except ValueError:
        print(json.dumps({"error": f"not a workspace id: {args.workspace_id}"}))
        return EXIT_ERROR

    from core.database.database import SessionLocal
    from services.report_knowledge import remove_documents

    db = SessionLocal()
    try:
        plan = asyncio.run(review(db, workspace_id))
        removed = remove_documents(db, workspace_id, [d["document"] for d in plan[REMOVE]]) if args.apply else []
        db.rollback()                  # the review only reads; removals commit through the manager
        print(json.dumps({"workspace": str(workspace_id), "applied": args.apply, "kept": len(plan[KEEP]),
                          "removed" if args.apply else "would_remove": plan[REMOVE], "removed_ids": removed,
                          "unknown": plan[UNKNOWN]}, default=str))
        return EXIT_OK
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
