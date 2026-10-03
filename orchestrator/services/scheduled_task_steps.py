"""The scheduler's steps that need no service object (PRD-252 split them out of
``scheduled_task_service``): what a scheduling request may ask for, what
scheduling says it did, and what firing a row does to it and to the board.

PRD-252 R4: a schedule's id never follows a '#', which now means a ticket's
number; the ticket a board-delivery row files is named by its own number.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from services.schedule_util import is_valid_cron

logger = logging.getLogger(__name__)

DELIVER_CHAT = "chat"              # PRD-77: open a chat with the target agent
DELIVER_BOARD_TASK = "board_task"  # file a board ticket (assigned, or Inbox)
DELIVERY_MODES = (DELIVER_CHAT, DELIVER_BOARD_TASK)
REVIEW_MODES = ("auto", "human", "llm")


def request_error(
    task_type: str, deliver_as: str, created_by_agent_id: Optional[int], created_by_user_id: Optional[str],
    target_agent_id: Optional[int], payload: Dict[str, Any],
) -> Optional[str]:
    """Why a scheduling request cannot be taken as asked, or None."""
    if task_type not in ("one_shot", "recurring"):
        return "task_type must be 'one_shot' or 'recurring'"
    if deliver_as not in DELIVERY_MODES:
        return f"deliver_as must be one of: {', '.join(DELIVERY_MODES)}"
    if created_by_agent_id is None and not created_by_user_id:
        return "A creator is required (an agent or a user)"
    if deliver_as == DELIVER_CHAT and target_agent_id is None:
        return "Chat delivery needs a target agent"
    if deliver_as == DELIVER_BOARD_TASK and not str(payload.get("title") or "").strip():
        return "A board task needs a title"
    return None


def one_shot_time(schedule: str) -> datetime:
    return datetime.fromisoformat(schedule.replace("Z", "+00:00"))


def schedule_error(task_type: str, schedule: str) -> Optional[str]:
    """Why ``schedule`` is not a time this task can fire at, or None."""
    if task_type == "one_shot":
        try:
            if one_shot_time(schedule) <= datetime.now(timezone.utc):
                return "Schedule datetime must be in the future"
        except (ValueError, TypeError):
            return f"Invalid ISO datetime: {schedule}. Use format: 2026-03-11T09:00:00Z"
        return None
    # Validate with the shared cron util (croniter — standard crontab
    # semantics, the same the calendar and firing use post-PRD-162).
    if not is_valid_cron(schedule):
        return f"Invalid cron expression: {schedule}. Use 5-field format: '0 9 * * 1' (minute hour dom month dow)"
    return None


def scheduled_message(
    task_id: int, task_type: str, deliver_as: str, payload: Dict[str, Any], target_name: Optional[str],
) -> str:
    """What scheduling says it did. PRD-252 R4: a schedule's id never follows a
    '#', which now means a ticket's number (the ticket it files gets its own)."""
    if deliver_as == DELIVER_BOARD_TASK:
        where = f"assigned to '{target_name}'" if target_name else "in the Inbox"
        return f"Scheduled {task_type} board task (schedule {task_id}) '{payload['title']}' — filed {where} when it fires"
    return f"Scheduled {task_type} task (schedule {task_id}) for agent '{target_name or 'unknown'}'"


def skipped_for_trial(db: Session, task: Any, task_id: int) -> bool:
    """PRD-222 US-005 — no background burn: trial workspaces get no scheduled
    execution until converted. Visible skip, never silent."""
    try:
        from core.models.workspaces import Workspace
        from services.trial_ledger import is_trial_active_workspace
        # PRD-234 S3: the local edition has no platform-paid trial credit to
        # protect (operator keys / the user's own Claude subscription) — the
        # onboarding trial record must not switch scheduled work off there.
        from config import config as _config
        _local_edition = getattr(_config, "AUTH_EDITION", "saas") == "local"

        _ws = db.query(Workspace).get(task.workspace_id)
        if not _local_edition and is_trial_active_workspace(_ws):
            logger.warning(
                "[ScheduledTask] Skipping task %d — trial workspace %s "
                "(no background burn until converted)",
                task_id, task.workspace_id,
            )
            return True
    except Exception:  # noqa: BLE001 — a failed trial check never stops a schedule firing
        logger.warning("[ScheduledTask] trial-skip check failed for task %d; firing it", task_id, exc_info=True)
    return False


def count_the_run(db: Session, task: Any, task_id: int) -> None:
    """Record the run; a row that reached max_runs, or a one-shot, is completed."""
    db.execute(
        text("""
            UPDATE agent_scheduled_tasks
            SET run_count = run_count + 1,
                last_run_at = NOW(),
                updated_at = NOW()
            WHERE id = :id
        """),
        {"id": task_id},
    )
    if task.max_runs and (task.run_count + 1) >= task.max_runs:
        _complete(db, task_id)
        _unschedule(task_id)
    if task.task_type == "one_shot":
        _complete(db, task_id)


def _complete(db: Session, task_id: int) -> None:
    db.execute(
        text("""
            UPDATE agent_scheduled_tasks
            SET status = 'completed', updated_at = NOW()
            WHERE id = :id
        """),
        {"id": task_id},
    )


def _unschedule(task_id: int) -> None:
    """Remove a completed row's job from the scheduler."""
    try:
        from services.scheduler import get_unified_scheduler
        sched = get_unified_scheduler()
        if sched.apscheduler:
            sched.apscheduler.remove_job(f"scheduled_task_{task_id}")
    except Exception:  # noqa: BLE001 — the reconcile tick drops a completed row's job too
        logger.debug("[ScheduledTask] could not remove the job for task %d", task_id, exc_info=True)


def ticket_for(task: Any, task_id: int) -> Any:
    """The board ticket a fired board-delivery row files (not yet added)."""
    from core.models.core import BoardTask
    from services.board_sla import PRIORITY_SLA_HOURS, sla_deadline_for
    from services.cli_ticket_lane import source_id_for

    payload = task.payload if isinstance(getattr(task, "payload", None), dict) else {}
    description = str(task.description or "").strip()
    first_line = description.splitlines()[0] if description else "Scheduled task"
    title = (str(payload.get("title") or "").strip() or first_line)[:255]
    priority = payload.get("priority") if payload.get("priority") in PRIORITY_SLA_HOURS else "medium"
    review_mode = payload.get("review_mode") if payload.get("review_mode") in REVIEW_MODES else "auto"
    assigned_agent_id = getattr(task, "target_agent_id", None)
    created_by_user_id = getattr(task, "created_by_user_id", None)
    now = datetime.now(timezone.utc)
    return BoardTask(
        workspace_id=task.workspace_id,
        title=title,
        description=description or title,
        priority=priority,
        review_mode=review_mode,
        assigned_agent_id=assigned_agent_id,
        status="assigned" if assigned_agent_id else "inbox",
        created_by_type="user" if created_by_user_id else "agent",
        created_by_id=str(created_by_user_id or getattr(task, "created_by_agent_id", None) or ""),
        source_type="scheduled_task",
        source_id=source_id_for("task", task_id, now),
        tags=[str(t) for t in (payload.get("tags") or []) if t],
        sla_deadline=sla_deadline_for(priority, now=now),
    )


def tell_the_chat(db: Session, task: Any, ticket: Any) -> None:
    """Say in the chat that scheduled it that the ticket was filed (PRD-252 R4: by its number)."""
    origin_chat_id = getattr(task, "origin_chat_id", None)
    if not origin_chat_id:
        return
    from services.chat_messenger import deliver_background_message
    from services.ticket_numbers import ticket_label

    deliver_background_message(
        db, workspace_id=str(task.workspace_id),
        text=f"Filed board {ticket_label(ticket)}: {ticket.title}",
        source={"origin": "scheduled_task"}, chat_id=str(origin_chat_id),
        link_type="board_task", link_id=str(ticket.id),
    )
