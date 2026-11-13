"""``report_to: auto``: Auto's review ticket for a heartbeat (PRD-72).

Moved out of ``HeartbeatService._route_heartbeat_to_auto`` unchanged in what it files, so the
method fits the 50-line rule; it runs on a worker thread on a session of its own. Since
13 Nov 2026 a Claude Code agent's heartbeat reaches it when its session ends, not when its
ticket is queued (``services.heartbeats.session_ticks``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

HEARTBEAT_SOURCE = "heartbeat"
SUMMARY_LINE_MAX_CHARS = 200
FINDINGS_MAX_CHARS = 2000
TITLE_MAX_CHARS = 500
URGENT_STATUSES = ("error", "critical")


def _review_fields(*, source_agent_id: Optional[int], source_agent_name: Optional[str], message: str,
                   link_id: Optional[Any], status: str) -> Dict[str, Any]:
    """The review ticket's own words: what the heartbeat found, and where its report is."""
    first = (message or "").strip().splitlines()[0] if (message or "").strip() else ""
    summary_line = first[:SUMMARY_LINE_MAX_CHARS] or "no findings"
    description = (
        f"{source_agent_name or 'Agent'} heartbeat completed (status={status}).\n\n"
        f"Findings:\n{(message or '(no findings)')[:FINDINGS_MAX_CHARS]}\n\n"
        f"Read the full report with `platform_get_latest_report` "
        f"(agent_name=\"{source_agent_name}\")."
    )
    return {
        "title": f"Review {source_agent_name or 'agent'} heartbeat — {summary_line}"[:TITLE_MAX_CHARS],
        "description": description,
        "priority": "high" if status in URGENT_STATUSES else "medium",
        "created_by_id": str(source_agent_id) if source_agent_id else None,
        "source_id": str(link_id) if link_id is not None else None,
    }


def file_auto_review(*, workspace_id: str, source_agent_id: Optional[int], source_agent_name: Optional[str],
                     message: str, link_id: Optional[Any], status: str) -> Optional[int]:
    """A BoardTask for the workspace's Auto to pick up on its next tick; its id, or None when
    the workspace has no Auto or the write failed (logged). Auto's heartbeat already scans its
    assigned tickets, so a ticket is enough: no scheduler, no notification."""
    from core.database.database import SessionLocal
    from core.models import Agent
    from core.models.core import BoardTask
    from services.board_consent import consent_for_lane_ticket

    db = SessionLocal()
    try:
        auto = db.query(Agent).filter(Agent.workspace_id == workspace_id, Agent.is_system_agent.is_(True),
                                      Agent.name == "Auto").first()
        if not auto:
            logger.warning("[Heartbeat] report_to=auto but no canonical Auto found for ws=%s — "
                           "falling back to silent (DB only).", workspace_id)
            return None
        fields = _review_fields(source_agent_id=source_agent_id, source_agent_name=source_agent_name,
                                message=message, link_id=link_id, status=status)
        task = BoardTask(workspace_id=workspace_id, status="assigned", assigned_agent_id=auto.id,
                         created_by_type="agent", source_type=HEARTBEAT_SOURCE, **fields)
        db.add(task)
        db.commit()
        # Local edition: the operator's heartbeat schedule is the approval
        # (PRD-234 board_consent) — Auto's review ticket runs on the next tick.
        consent_for_lane_ticket(db, workspace_id=workspace_id, task=task, source_type=HEARTBEAT_SOURCE)
        logger.info("[Heartbeat] report_to=auto: created BoardTask id=%s assigned to Auto (agent_id=%s) "
                    "for source_agent=%s", task.id, auto.id, source_agent_id)
        return task.id
    except Exception:  # noqa: BLE001 — a heartbeat never fails on its review ticket; logged
        db.rollback()
        logger.exception("[Heartbeat] the review ticket for Auto was not filed for ws=%s", workspace_id)
        return None
    finally:
        db.close()


__all__ = ["file_auto_review"]
