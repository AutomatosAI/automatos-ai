"""PRD-234 S3 — every lane that targets a Claude Code agent files a board ticket.

Heartbeats, scheduled tasks, channel mentions, webhooks and Composio triggers all
used to call ``AgentFactory.execute_with_prompt``; for a ``runtime: cli`` agent the
factory refuses (by design — the user's own ``claude`` runs the work), and each
lane then either logged the refusal, replied nothing, or worse, recorded it as a
success. The honest shape is one ticket on the board: the paired CLI host claims
it, the result lands like any other ticket's, and the lane can say so.

One writer, one shape: the ticket carries the lane's prompt as its description,
``source_type`` names the lane, and ``source_id`` makes the lane's re-fires
idempotent (an open ticket from the same source is reused, never duplicated).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from sqlalchemy.orm import Session

from core.cli_runtime import RUNTIME_CLI, runtime_kind_of
from core.models.core import Agent, BoardTask

logger = logging.getLogger(__name__)

OPEN_STATUSES: Sequence[str] = ("inbox", "assigned", "in_progress", "blocked", "review")
QUEUED_LINE = "queued for your Claude Code session as ticket #{task_id}"
NO_HOST_REASON = (
    "Waiting for a CLI host — none is online. Start it with `make cli-host`; "
    "the ticket is claimed on its first poll."
)


def is_cli_agent(db: Session, agent_id: Optional[int]) -> bool:
    """True when the agent's configuration says ``runtime: cli``. Lookup failure → False
    (the caller then takes the API path, whose own guard still refuses cli agents)."""
    if agent_id is None:
        return False
    try:
        agent = db.query(Agent).filter(Agent.id == int(agent_id)).first()
    except Exception:  # noqa: BLE001
        return False
    if agent is None:
        return False
    return runtime_kind_of(getattr(agent, "configuration", None) or {}) == RUNTIME_CLI


def open_ticket_for_source(db: Session, workspace_id: Any, source_type: str, source_id: str) -> Optional[BoardTask]:
    """The still-open ticket a lane already filed for this source, if any."""
    try:
        return (
            db.query(BoardTask)
            .filter(
                BoardTask.workspace_id == workspace_id,
                BoardTask.source_type == source_type,
                BoardTask.source_id == source_id,
                BoardTask.status.in_(list(OPEN_STATUSES)),
            )
            .order_by(BoardTask.id.desc())
            .first()
        )
    except Exception:  # noqa: BLE001
        return None


def host_online(db: Session, workspace_id: Any) -> bool:
    """Is any paired CLI host of this workspace currently heartbeating?"""
    try:
        from core.models.cli_hosts import CliHost, CliHostStatus
        hosts = db.query(CliHost).filter(
            CliHost.workspace_id == workspace_id,
            CliHost.status == CliHostStatus.PAIRED.value,
        ).all()
        return any(h.is_online() for h in hosts)
    except Exception:  # noqa: BLE001
        return False


def file_cli_ticket(
    db: Session,
    *,
    workspace_id: Any,
    agent_id: int,
    title: str,
    prompt: str,
    source_type: str,
    source_id: str,
    priority: str = "medium",
    review_mode: str = "auto",
    tags: Optional[list] = None,
) -> BoardTask:
    """File (or reuse) the board ticket a lane owes a Claude Code agent.

    Returns the ticket. ``task.blocked_reason`` carries the no-host warning when
    no paired host is online — the status stays ``assigned`` so the host claims
    it the moment it is back; nothing needs re-dispatching.
    """
    existing = open_ticket_for_source(db, workspace_id, source_type, source_id)
    if existing is not None:
        logger.info("[CliTicketLane] %s/%s already has open ticket #%s — reusing", source_type, source_id, existing.id)
        return existing
    task = BoardTask(
        workspace_id=workspace_id,
        title=title[:255],
        description=prompt,
        priority=priority if priority in ("low", "medium", "high", "urgent") else "medium",
        assigned_agent_id=agent_id,
        status="assigned",
        created_by_type="system",
        created_by_id=source_type,
        source_type=source_type,
        source_id=source_id,
        review_mode=review_mode,
        tags=list(tags or []),
    )
    if not host_online(db, workspace_id):
        task.blocked_reason = NO_HOST_REASON
    db.add(task)
    db.commit()
    db.refresh(task)
    _notify(db, workspace_id, task)
    logger.info("[CliTicketLane] filed ticket #%s for agent %s from %s/%s", task.id, agent_id, source_type, source_id)
    return task


def queued_line(task: BoardTask) -> str:
    """The one line a lane replies with."""
    line = QUEUED_LINE.format(task_id=task.id)
    if getattr(task, "blocked_reason", None):
        line += " (no CLI host is online yet — start it with `make cli-host`)"
    return line


def _notify(db: Session, workspace_id: Any, task: BoardTask) -> None:
    """Board SSE + dispatcher wake, both fail-soft (the same two calls the HTTP
    create path makes)."""
    try:
        from services.board_events import notify_board_event
        notify_board_event(db, workspace_id=str(workspace_id), task_id=task.id, status=task.status, event="task_created")
    except Exception:  # noqa: BLE001
        logger.debug("[CliTicketLane] board notify skipped", exc_info=True)
    try:
        from services.board_dispatcher import notify_task_available
        notify_task_available(db, workspace_id=str(workspace_id), task_id=task.id)
    except Exception:  # noqa: BLE001
        logger.debug("[CliTicketLane] dispatch notify skipped", exc_info=True)


def source_id_for(prefix: str, key: Any, at: Optional[datetime] = None) -> str:
    """A stable per-fire source id: ``heartbeat`` wants ONE open ticket per agent
    (no timestamp); a scheduled task wants one per run (timestamp)."""
    if at is None:
        return f"{prefix}:{key}"
    return f"{prefix}:{key}:{at.astimezone(timezone.utc):%Y%m%dT%H%M}"
