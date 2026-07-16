"""
Escalation Service — auto-escalate blocked and repeatedly-stalled tasks.

Creates board tasks for human attention when:
- A board task has been blocked > 24 hours
- A task has stalled repeatedly (2+ stalls)

(Budget-paused missions are surfaced by the coordinator's
``mission_budget_paused`` notification — PRD-204 S4 — not by a board card.)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from core.models.core import BoardTask

logger = logging.getLogger(__name__)

# Blocked tasks escalate after this many hours
BLOCKED_ESCALATION_HOURS = 24


def check_blocked_escalations(db: Session, workspace_id) -> int:
    """
    Find board tasks blocked longer than BLOCKED_ESCALATION_HOURS and
    create escalation tasks for human review.

    Returns the number of escalations created.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=BLOCKED_ESCALATION_HOURS)

    blocked_tasks = (
        db.query(BoardTask)
        .filter(
            BoardTask.workspace_id == workspace_id,
            BoardTask.status == "blocked",
            BoardTask.blocked_at.isnot(None),
            BoardTask.blocked_at <= cutoff,
        )
        .all()
    )

    created = 0
    for task in blocked_tasks:
        # Check if an escalation already exists for this task
        existing = (
            db.query(BoardTask.id)
            .filter(
                BoardTask.workspace_id == workspace_id,
                BoardTask.tags.contains(["escalation", f"blocked:{task.id}"]),
                BoardTask.status.notin_(["done"]),
            )
            .first()
        )
        if existing:
            continue

        hours_blocked = (datetime.now(timezone.utc) - task.blocked_at).total_seconds() / 3600

        escalation = BoardTask(
            workspace_id=workspace_id,
            title=f"Escalation: '{task.title[:100]}' blocked {int(hours_blocked)}h",
            description=(
                f"Task #{task.id} has been blocked for {int(hours_blocked)} hours.\n\n"
                f"**Reason:** {task.blocked_reason or 'Unknown'}\n\n"
                f"**Original task:** {task.title}\n\n"
                f"Please investigate and unblock or reassign."
            ),
            status="inbox",
            priority="high",
            review_mode="human",
            created_by_type="system",
            created_by_id="escalation_service",
            tags=["escalation", f"blocked:{task.id}"],
        )
        db.add(escalation)
        created += 1

        logger.warning(
            "Created escalation task for blocked board task %d (blocked %dh) in workspace %s",
            task.id,
            int(hours_blocked),
            workspace_id,
        )

    if created > 0:
        db.flush()

    return created


# PRD-204 S4: notify_budget_exceeded was removed. It was dead code (zero
# non-test callers since introduction) and duplicated the budget-pause
# surface. The single owner of that boundary is now the coordinator's
# ``notify_mission_budget_paused`` (a real ``mission_budget_paused``
# notification dispatched when the dispatcher's budget gate pauses a run);
# the paused mission's board card already flips to "blocked" via
# orchestration_board_bridge, so a second escalation card added nothing.


def escalate_stalled_task(
    db: Session,
    workspace_id,
    task_id: int,
    task_title: str,
    stall_count: int,
    agent_name: str = "Unknown",
) -> Optional[BoardTask]:
    """
    Escalate a task that has stalled repeatedly (2+ times).
    Creates a high-priority inbox task for human review.
    """
    if stall_count < 2:
        return None

    # Check for existing escalation
    existing = (
        db.query(BoardTask.id)
        .filter(
            BoardTask.workspace_id == workspace_id,
            BoardTask.tags.contains(["escalation", f"stalled:{task_id}"]),
            BoardTask.status.notin_(["done"]),
        )
        .first()
    )
    if existing:
        return None

    escalation = BoardTask(
        workspace_id=workspace_id,
        title=f"Repeated stall: '{task_title[:100]}' ({stall_count}x)",
        description=(
            f"Task #{task_id} has stalled {stall_count} times.\n\n"
            f"**Agent:** {agent_name}\n"
            f"**Task:** {task_title}\n\n"
            f"Consider reassigning to a different agent or investigating the root cause."
        ),
        status="inbox",
        priority="high",
        review_mode="human",
        created_by_type="system",
        created_by_id="escalation_service",
        tags=["escalation", f"stalled:{task_id}"],
    )
    db.add(escalation)
    db.flush()

    logger.warning(
        "Created stall escalation for task %d (stalled %dx) in workspace %s",
        task_id,
        stall_count,
        workspace_id,
    )
    return escalation
