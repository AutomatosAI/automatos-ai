"""
PRD-129 US-005: Backfill existing agent_reports into deliverables
==================================================================

Iterates every row in ``agent_reports`` and calls
``DeliverableService.register()`` so that historical reports show up in the
new Workspace Outputs gallery without re-running any agents.

Idempotent: relies on the unique partial index
``uq_deliverables_workspace_path ON (workspace_id, file_path) WHERE deleted_at IS NULL``
from the PRD-129 migration. Re-running the script updates existing rows in
place (via ON CONFLICT) — it never creates duplicates.

Usage::

    cd orchestrator
    python scripts/backfill_prd129_deliverables.py --dry-run
    python scripts/backfill_prd129_deliverables.py
    python scripts/backfill_prd129_deliverables.py --workspace-id <uuid>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from config import config
from services.deliverable_service import DeliverableService

logger = logging.getLogger("backfill_prd129_deliverables")


def _source_type_for(report_type: Optional[str], heartbeat_result_id: Optional[int]) -> str:
    """Mirror ReportService logic: heartbeat standups vs task reports."""
    if heartbeat_result_id is not None or (report_type or "") == "standup":
        return "heartbeat"
    return "task"


def backfill(db: Session, *, workspace_id: Optional[str], dry_run: bool) -> dict:
    """Iterate agent_reports and register each as a deliverable.

    Returns a summary dict: ``{processed, inserted, updated, skipped, errors}``.
    """
    conditions = []
    params: dict = {}
    if workspace_id:
        conditions.append("workspace_id = :workspace_id")
        params["workspace_id"] = workspace_id

    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    rows = db.execute(
        text(f"""
            SELECT
                r.id,
                r.workspace_id,
                r.agent_id,
                r.heartbeat_result_id,
                r.report_type,
                r.title,
                r.summary,
                r.file_path,
                r.file_type,
                r.file_size_bytes,
                a.name AS agent_name
            FROM agent_reports r
            LEFT JOIN agents a ON a.id = r.agent_id
            {where_sql}
            ORDER BY r.created_at ASC
        """),
        params,
    ).fetchall()

    summary = {"processed": 0, "inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    # Group by workspace so we only build one DeliverableService per workspace.
    services: dict[str, DeliverableService] = {}

    for row in rows:
        summary["processed"] += 1
        ws_id = str(row.workspace_id)

        if not row.file_path:
            logger.warning("Skipping report %s — no file_path", row.id)
            summary["skipped"] += 1
            continue

        source_type = _source_type_for(row.report_type, row.heartbeat_result_id)
        source_id = (
            str(row.heartbeat_result_id)
            if row.heartbeat_result_id is not None
            else str(row.id)
        )

        if dry_run:
            print(
                f"[DRY-RUN] would register: workspace={ws_id} "
                f"agent={row.agent_name!r}({row.agent_id}) "
                f"path={row.file_path} title={row.title!r} source={source_type}"
            )
            summary["skipped"] += 1
            continue

        svc = services.get(ws_id)
        if svc is None:
            svc = DeliverableService(db=db, workspace_id=ws_id)
            services[ws_id] = svc

        try:
            result = svc.register(
                file_path=row.file_path,
                title=row.title,
                source_type=source_type,
                source_id=source_id,
                agent_id=row.agent_id,
                agent_name=row.agent_name,
                artifact_type="report",
                summary=row.summary,
                storage_type="workspace",
                file_type=row.file_type or "md",
                file_size_bytes=row.file_size_bytes,
                extra={"report_id": str(row.id), "report_type": row.report_type},
            )
        except Exception as exc:  # defensive — register() should not raise
            logger.error(
                "register() raised for report %s: %s", row.id, exc, exc_info=True,
            )
            summary["errors"] += 1
            continue

        if not result.get("success"):
            logger.error(
                "register() failed for report %s: %s",
                row.id, result.get("error"),
            )
            summary["errors"] += 1
            continue

        if result.get("created"):
            summary["inserted"] += 1
        else:
            summary["updated"] += 1

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill agent_reports into deliverables (PRD-129 US-005)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be inserted without touching the database",
    )
    parser.add_argument(
        "--workspace-id",
        type=str,
        default=None,
        help="Limit backfill to a single workspace (UUID)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine = create_engine(config.DATABASE_URL)

    print("=" * 72)
    print("PRD-129 Deliverables Backfill")
    print(f"  dry-run:      {args.dry_run}")
    print(f"  workspace-id: {args.workspace_id or '<all>'}")
    print("=" * 72)

    with Session(engine) as db:
        summary = backfill(
            db,
            workspace_id=args.workspace_id,
            dry_run=args.dry_run,
        )

    print()
    print("-" * 72)
    print(
        f"Processed: {summary['processed']}  "
        f"Inserted: {summary['inserted']}  "
        f"Updated: {summary['updated']}  "
        f"Skipped: {summary['skipped']}  "
        f"Errors: {summary['errors']}"
    )
    print("-" * 72)

    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
