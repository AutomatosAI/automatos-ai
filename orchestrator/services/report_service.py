"""
Report Service (PRD-76)
========================

Handles agent report creation, listing, grading, and stats.
Reports combine DB metadata (for discovery/filtering) with
workspace files (for full content).
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import desc, func, text
from sqlalchemy.orm import Session

from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    """Convert string to kebab-case slug for file naming."""
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:80]


def _agent_slug(agent_name: str) -> str:
    """Consistent agent directory name."""
    return _slugify(agent_name)


class ReportService:
    """Service for managing agent reports."""

    def __init__(self, db: Session, workspace_id: UUID):
        self.db = db
        self.workspace_id = workspace_id

    async def create_report(
        self,
        agent_id: Optional[int],
        agent_name: str,
        title: str,
        content: str,
        report_type: str = "standup",
        status: str = "ok",
        summary: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        heartbeat_result_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a report: write file to workspace + insert DB row."""

        import uuid as _uuid
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d_%H%M%S") + "_" + _uuid.uuid4().hex[:6]
        title_slug = _slugify(title)
        agent_dir = _agent_slug(agent_name)

        file_path = f"reports/{agent_dir}/{date_str}_{title_slug}.md"

        # Auto-generate summary from content if not provided
        if not summary and content:
            # First non-empty line after stripping markdown headers
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            summary = (lines[0][:497] + "...") if lines and len(lines[0]) > 497 else (lines[0] if lines else None)

        # 1. Write file to workspace
        ws_client = WorkspaceClient(str(self.workspace_id))
        write_result = await ws_client.write_file(file_path, content)

        if not write_result.get("success", False):
            logger.error(
                "[ReportService] Failed to write report file %s: %s",
                file_path,
                write_result.get("error", "unknown"),
            )
            return {
                "success": False,
                "error": f"Failed to write report file: {write_result.get('error', 'unknown')}",
            }

        # 2. Get file size
        file_size = len(content.encode("utf-8"))

        # 3. Insert DB row
        try:
            result = self.db.execute(
                text("""
                    INSERT INTO agent_reports
                        (workspace_id, agent_id, agent_name, heartbeat_result_id,
                         report_type, title, summary, status,
                         file_path, file_type, file_size_bytes,
                         metrics, attachments, created_at, updated_at)
                    VALUES
                        (:workspace_id, :agent_id, :agent_name, :heartbeat_result_id,
                         :report_type, :title, :summary, :status,
                         :file_path, :file_type, :file_size_bytes,
                         :metrics, :attachments, NOW(), NOW())
                    RETURNING id
                """),
                {
                    "workspace_id": str(self.workspace_id),
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "heartbeat_result_id": heartbeat_result_id,
                    "report_type": report_type,
                    "title": title,
                    "summary": summary,
                    "status": status,
                    "file_path": file_path,
                    "file_type": "markdown",
                    "file_size_bytes": file_size,
                    "metrics": json.dumps(metrics or {}),
                    "attachments": json.dumps(attachments or []),
                },
            )
            row = result.fetchone()
            report_id = str(row[0]) if row else None

            # PRD-128: dispatch report_submitted before commit so the
            # notification row joins the same transaction as the report
            # insert. Never blocks the report flow.
            try:
                from core.services.notification_dispatcher import NotificationDispatcher

                dispatcher = NotificationDispatcher(self.db, str(self.workspace_id))
                await dispatcher.dispatch(
                    event_type="report_submitted",
                    title=f"Report: {title}",
                    message=(summary or "")[:500] or None,
                    link_type="report",
                    link_id=report_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    status=status,
                )
            except Exception:
                logger.error(
                    "[ReportService] report_submitted dispatch failed for report %s",
                    report_id,
                    exc_info=True,
                )

            self.db.commit()

            logger.info(
                "[ReportService] Created report %s for agent %s (%s): %s",
                report_id, agent_name, report_type, title,
            )

            return {
                "success": True,
                "report_id": report_id,
                "file_path": file_path,
                "title": title,
            }

        except Exception as e:
            self.db.rollback()
            logger.error("[ReportService] DB insert failed: %s", e, exc_info=True)
            return {"success": False, "error": f"Database insert failed: {e}"}

    async def list_reports(
        self,
        agent_id: Optional[int] = None,
        report_type: Optional[str] = None,
        status: Optional[str] = None,
        graded: Optional[bool] = None,
        period: str = "30d",
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List reports for workspace with filters."""

        conditions = ["r.workspace_id = :workspace_id"]
        params: Dict[str, Any] = {"workspace_id": str(self.workspace_id)}

        if agent_id:
            conditions.append("r.agent_id = :agent_id")
            params["agent_id"] = agent_id

        if report_type:
            conditions.append("r.report_type = :report_type")
            params["report_type"] = report_type

        if status:
            conditions.append("r.status = :status")
            params["status"] = status

        if graded is True:
            conditions.append("r.grade IS NOT NULL")
        elif graded is False:
            conditions.append("r.grade IS NULL")

        # Period filter
        days = _parse_period(period)
        if days:
            conditions.append("r.created_at >= NOW() - INTERVAL ':days days'")
            # Use string interpolation for interval (parameterized intervals not supported)
            conditions[-1] = f"r.created_at >= NOW() - INTERVAL '{days} days'"

        where = " AND ".join(conditions)

        # Count total
        count_result = self.db.execute(
            text(f"SELECT COUNT(*) FROM agent_reports r WHERE {where}"),
            params,
        )
        total = count_result.scalar() or 0

        # Fetch reports with agent info
        query = text(f"""
            SELECT
                r.id, r.agent_id, COALESCE(r.agent_name, a.name, 'Orchestrator') AS agent_name,
                r.heartbeat_result_id,
                r.report_type, r.title, r.summary, r.status,
                r.file_path, r.file_type, r.file_size_bytes,
                r.metrics, r.attachments,
                r.grade, r.grade_notes, r.graded_by, r.graded_at,
                r.created_at
            FROM agent_reports r
            LEFT JOIN agents a ON a.id = r.agent_id
            WHERE {where}
            ORDER BY r.created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        params["limit"] = limit
        params["offset"] = offset

        rows = self.db.execute(query, params).fetchall()

        reports = []
        for row in rows:
            reports.append({
                "id": str(row.id),
                "agent_id": row.agent_id,
                "agent_name": row.agent_name,
                "heartbeat_result_id": row.heartbeat_result_id,
                "report_type": row.report_type,
                "title": row.title,
                "summary": row.summary,
                "status": row.status,
                "file_path": row.file_path,
                "file_type": row.file_type,
                "file_size_bytes": row.file_size_bytes,
                "metrics": row.metrics if isinstance(row.metrics, dict) else json.loads(row.metrics or "{}"),
                "attachments": row.attachments if isinstance(row.attachments, list) else json.loads(row.attachments or "[]"),
                "grade": row.grade,
                "grade_notes": row.grade_notes,
                "graded_by": row.graded_by,
                "graded_at": row.graded_at.isoformat() if row.graded_at else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })

        return {
            "success": True,
            "reports": reports,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def get_report(self, report_id: str, include_content: bool = True) -> Dict[str, Any]:
        """Get a single report with optional file content."""

        row = self.db.execute(
            text("""
                SELECT
                    r.id, r.workspace_id, r.agent_id,
                    COALESCE(r.agent_name, a.name, 'Orchestrator') AS agent_name,
                    r.heartbeat_result_id,
                    r.report_type, r.title, r.summary, r.status,
                    r.file_path, r.file_type, r.file_size_bytes,
                    r.metrics, r.attachments,
                    r.grade, r.grade_notes, r.graded_by, r.graded_at,
                    r.created_at
                FROM agent_reports r
                LEFT JOIN agents a ON a.id = r.agent_id
                WHERE r.id = :report_id AND r.workspace_id = :workspace_id
            """),
            {"report_id": report_id, "workspace_id": str(self.workspace_id)},
        ).fetchone()

        if not row:
            return {"success": False, "error": "Report not found"}

        report = {
            "id": str(row.id),
            "workspace_id": str(row.workspace_id),
            "agent_id": row.agent_id,
            "agent_name": row.agent_name,
            "heartbeat_result_id": row.heartbeat_result_id,
            "report_type": row.report_type,
            "title": row.title,
            "summary": row.summary,
            "status": row.status,
            "file_path": row.file_path,
            "file_type": row.file_type,
            "file_size_bytes": row.file_size_bytes,
            "metrics": row.metrics if isinstance(row.metrics, dict) else json.loads(row.metrics or "{}"),
            "attachments": row.attachments if isinstance(row.attachments, list) else json.loads(row.attachments or "[]"),
            "grade": row.grade,
            "grade_notes": row.grade_notes,
            "graded_by": row.graded_by,
            "graded_at": row.graded_at.isoformat() if row.graded_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

        # Fetch file content from workspace
        if include_content:
            ws_client = WorkspaceClient(str(self.workspace_id))
            file_result = await ws_client.read_file(row.file_path)
            if file_result.get("success", False):
                report["content"] = file_result.get("content", "")
            else:
                report["content"] = None
                report["content_error"] = file_result.get("error", "Could not read file")

        return {"success": True, "report": report}

    async def grade_report(
        self,
        report_id: str,
        grade: int,
        grade_notes: Optional[str] = None,
        graded_by: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Submit a grade for a report."""

        if grade < 1 or grade > 5:
            return {"success": False, "error": "Grade must be between 1 and 5"}

        result = self.db.execute(
            text("""
                UPDATE agent_reports
                SET grade = :grade,
                    grade_notes = :grade_notes,
                    graded_by = :graded_by,
                    graded_at = NOW(),
                    updated_at = NOW()
                WHERE id = :report_id AND workspace_id = :workspace_id
                RETURNING id
            """),
            {
                "report_id": report_id,
                "workspace_id": str(self.workspace_id),
                "grade": grade,
                "grade_notes": grade_notes,
                "graded_by": graded_by,
            },
        )
        row = result.fetchone()
        self.db.commit()

        if not row:
            return {"success": False, "error": "Report not found"}

        return {"success": True, "report_id": report_id, "grade": grade}

    async def get_stats(self, period: str = "7d") -> Dict[str, Any]:
        """Aggregate report stats for the workspace."""

        days = _parse_period(period) or 7
        interval = f"{days} days"

        row = self.db.execute(
            text(f"""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE grade IS NULL) AS ungraded_count,
                    AVG(grade) FILTER (WHERE grade IS NOT NULL) AS avg_grade
                FROM agent_reports
                WHERE workspace_id = :workspace_id
                  AND created_at >= NOW() - INTERVAL '{interval}'
            """),
            {"workspace_id": str(self.workspace_id)},
        ).fetchone()

        by_type = self.db.execute(
            text(f"""
                SELECT report_type, COUNT(*) AS cnt
                FROM agent_reports
                WHERE workspace_id = :workspace_id
                  AND created_at >= NOW() - INTERVAL '{interval}'
                GROUP BY report_type
            """),
            {"workspace_id": str(self.workspace_id)},
        ).fetchall()

        by_status = self.db.execute(
            text(f"""
                SELECT status, COUNT(*) AS cnt
                FROM agent_reports
                WHERE workspace_id = :workspace_id
                  AND created_at >= NOW() - INTERVAL '{interval}'
                GROUP BY status
            """),
            {"workspace_id": str(self.workspace_id)},
        ).fetchall()

        return {
            "success": True,
            "total": row.total if row else 0,
            "ungraded_count": row.ungraded_count if row else 0,
            "avg_grade": round(float(row.avg_grade), 2) if row and row.avg_grade else None,
            "by_type": {r.report_type: r.cnt for r in by_type},
            "by_status": {r.status: r.cnt for r in by_status},
            "period": period,
        }

    async def get_latest_report(
        self,
        agent_name: Optional[str] = None,
        agent_id: Optional[int] = None,
        report_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the most recent report from a specific agent."""

        conditions = ["r.workspace_id = :workspace_id"]
        params: Dict[str, Any] = {"workspace_id": str(self.workspace_id)}

        if agent_id:
            conditions.append("r.agent_id = :agent_id")
            params["agent_id"] = agent_id
        elif agent_name:
            conditions.append("LOWER(a.name) LIKE LOWER(:agent_name)")
            params["agent_name"] = f"%{agent_name}%"
        else:
            return {"success": False, "error": "Provide agent_name or agent_id"}

        if report_type:
            conditions.append("r.report_type = :report_type")
            params["report_type"] = report_type

        where = " AND ".join(conditions)

        row = self.db.execute(
            text(f"""
                SELECT
                    r.id, r.agent_id,
                    COALESCE(r.agent_name, a.name, 'Orchestrator') AS agent_name,
                    r.report_type, r.title, r.summary, r.status,
                    r.file_path, r.metrics, r.created_at
                FROM agent_reports r
                LEFT JOIN agents a ON a.id = r.agent_id
                WHERE {where}
                ORDER BY r.created_at DESC
                LIMIT 1
            """),
            params,
        ).fetchone()

        if not row:
            return {"success": False, "error": "No reports found for this agent"}

        # Fetch content
        ws_client = WorkspaceClient(str(self.workspace_id))
        file_result = await ws_client.read_file(row.file_path)

        report_data = {
            "id": str(row.id),
            "agent_id": row.agent_id,
            "agent_name": row.agent_name,
            "report_type": row.report_type,
            "title": row.title,
            "summary": row.summary,
            "status": row.status,
            "file_path": row.file_path,
            "metrics": row.metrics if isinstance(row.metrics, dict) else json.loads(row.metrics or "{}"),
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

        if file_result.get("success"):
            report_data["content"] = file_result.get("content", "")
        else:
            report_data["content"] = None
            report_data["content_error"] = file_result.get("error", "Could not read report file")
            logger.warning(
                "[ReportService] Could not read report file %s: %s",
                row.file_path, file_result.get("error"),
            )

        return {"success": True, "report": report_data}


def _parse_period(period: str) -> Optional[int]:
    """Parse period string like '7d', '30d', '1d' into days."""
    match = re.match(r"(\d+)d", period)
    return int(match.group(1)) if match else None
