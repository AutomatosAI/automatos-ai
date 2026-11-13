"""``report_to: auto``: Auto's review ticket for a heartbeat (PRD-72), and when one is due.

Moved out of ``HeartbeatService._route_heartbeat_to_auto`` unchanged in what it files, so a
Claude Code agent's heartbeat can file it when its SESSION ends (13 Nov 2026).

A ``runtime: cli`` agent's heartbeat files a board ticket for the paired host (PRD-234 S3)
and returns at once with one finding, "queued for your Claude Code session as ticket #0330",
which was handled like a result. On the Automatos HQ install the News & Research agent's
hourly heartbeat filed, every hour, a "standup" report holding only that line and a "Review
News & Research heartbeat — queued for …" ticket that ran Auto on nothing: 24 cards a day,
and the empty report hid the agent's real digest from ``get_latest_report``. A tick that only
queued a session now files neither (``only_queued_a_session``). When the session's ticket
finishes, ``services.task_report`` files its report and calls ``review_finished_session``,
which files Auto's review with what the session found.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

REPORT_TO_AUTO = "auto"
HEARTBEAT_SOURCE = "heartbeat"
# What a Claude Code agent's tick records when it files its session ticket
# (``HeartbeatService._agent_tick``: ``{"action": "file_cli_ticket", "task_id": …}``).
FILE_CLI_TICKET = "file_cli_ticket"
SUMMARY_LINE_MAX_CHARS = 200
FINDINGS_MAX_CHARS = 2000
TITLE_MAX_CHARS = 500
URGENT_STATUSES = ("error", "critical")
FAILED_TICKET_STATUSES = ("failed", "cancelled")


def only_queued_a_session(actions: Any) -> bool:
    """True when a heartbeat tick's actions are the session ticket(s) it filed and nothing else:
    the tick has no result yet, the session will."""
    try:
        items = json.loads(actions) if isinstance(actions, str) else actions
    except ValueError:
        return False
    if not isinstance(items, list) or not items:
        return False
    return all(isinstance(a, dict) and a.get("action") == FILE_CLI_TICKET for a in items)


def tick_only_queued_a_session(workspace_id: Any, link_id: Any) -> bool:
    """``only_queued_a_session`` for the stored tick ``link_id`` (a ``heartbeat_results`` row of
    this workspace). False when there is no such row: the review is filed as before."""
    from core.database.database import SessionLocal

    try:
        tick_id = int(link_id)
    except (TypeError, ValueError):
        return False
    db = SessionLocal()
    try:
        row = db.execute(
            text("SELECT actions_taken FROM heartbeat_results WHERE id = :id AND workspace_id = :workspace_id"),
            {"id": tick_id, "workspace_id": str(workspace_id)},
        ).fetchone()
        return row is not None and only_queued_a_session(row[0])
    except Exception:  # noqa: BLE001 — unknown means "file the review", as before this check existed
        logger.exception("[Heartbeat] tick %s could not be read; its review is filed as before", link_id)
        return False
    finally:
        db.close()


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


def is_session_heartbeat_ticket(task: Any) -> bool:
    """A Claude Code agent's heartbeat ticket (``cli_ticket_lane``: source ``heartbeat``,
    ``agent:<id>``), never Auto's review ticket (whose source is the tick's own id)."""
    from services.cli_ticket_lane import source_id_for

    source_id = str(getattr(task, "source_id", None) or "")
    return getattr(task, "source_type", None) == HEARTBEAT_SOURCE and source_id.startswith(source_id_for("agent", ""))


def _reports_to_auto(db: Session, workspace_id: Any, agent_id: Optional[int]) -> bool:
    from core.models import Agent

    if agent_id is None:
        return False
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    heartbeat = ((agent.configuration or {}).get("heartbeat") or {}) if agent is not None else {}
    return str(heartbeat.get("report_to") or "").strip().lower() == REPORT_TO_AUTO


async def review_finished_session(db: Session, workspace_id: str, task: Any, *, agent_name: str,
                                  answer: str) -> Optional[int]:
    """When a Claude Code agent's heartbeat ticket finishes and its agent reports to Auto,
    Auto's review ticket, carrying what the session found; its id, else None. The ticket is
    written on a worker thread, on a session of its own."""
    if not is_session_heartbeat_ticket(task) or not _reports_to_auto(db, workspace_id, task.assigned_agent_id):
        return None
    failed = getattr(task, "status", None) in FAILED_TICKET_STATUSES or bool(getattr(task, "error_message", None))
    message = answer or str(getattr(task, "error_message", None) or "")
    return await asyncio.to_thread(
        functools.partial(file_auto_review, workspace_id=str(workspace_id), source_agent_id=task.assigned_agent_id,
                          source_agent_name=agent_name, message=message, link_id=None,
                          status="error" if failed else "ok"),
    )


__all__ = ["file_auto_review", "is_session_heartbeat_ticket", "only_queued_a_session", "review_finished_session",
           "tick_only_queued_a_session"]
