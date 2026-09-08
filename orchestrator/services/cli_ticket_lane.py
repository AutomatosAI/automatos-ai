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
    actor: Optional[str] = None,
    created_by_type: Optional[str] = None,
    created_by_id: Optional[str] = None,
    resume_session_id: Optional[str] = None,
    resume_host_id: Optional[str] = None,
    orchestration_run_id: Any = None,
    orchestration_task_id: Any = None,
) -> BoardTask:
    """File (or reuse) the board ticket a lane owes a Claude Code agent.

    Returns the ticket. ``task.blocked_reason`` carries the no-host warning when
    no paired host is online — the status stays ``assigned`` so the host claims
    it the moment it is back; nothing needs re-dispatching.

    PRD-239: ``actor`` (``user:<id>``) marks a ticket a human asked for in a live
    chat turn — its consent is the operator's own (D16), not the lane's standing
    grant. ``resume_session_id`` + ``resume_host_id`` ask the host that ran the
    previous session of the same conversation to continue it (``claude --resume``).
    ``orchestration_run_id`` / ``orchestration_task_id`` tie a mission task's
    ticket to its run.
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
        created_by_type=created_by_type or "system",
        created_by_id=created_by_id or source_type,
        source_type=source_type,
        source_id=source_id,
        review_mode=review_mode,
        tags=list(tags or []),
        orchestration_run_id=orchestration_run_id,
        orchestration_task_id=orchestration_task_id,
    )
    if resume_session_id:
        # Read at claim (cli_host_service) and honoured only by the same host —
        # a transcript lives on one machine. The claim rebuilds runtime_ref.
        task.runtime_ref = {
            "resume_session_id": str(resume_session_id),
            "resume_host_id": str(resume_host_id) if resume_host_id else None,
        }
    if not host_online(db, workspace_id):
        task.blocked_reason = NO_HOST_REASON
    db.add(task)
    db.commit()
    db.refresh(task)
    # PRD-234 (2026-09-07): the operator's standing schedule is their approval on
    # the local edition — the dispatcher's claim finds an active grant instead of
    # parking the heartbeat behind 'always_ask'. No-op on SaaS. PRD-239: a ticket
    # a human typed for in chat carries that human's consent instead (D16).
    from services.board_consent import WHY_ASKED_IN_CHAT, consent_for_created_ticket, consent_for_lane_ticket
    if actor:
        consent_for_created_ticket(db, workspace_id=workspace_id, task=task, actor=actor, why=WHY_ASKED_IN_CHAT)
    else:
        consent_for_lane_ticket(db, workspace_id=workspace_id, task=task, source_type=source_type)
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


# ── PRD-239: the chat lane, and lanes that wait for the session to end ───────

CHAT_SOURCE_TYPE = "chat"
RECIPE_SOURCE_TYPE = "recipe"
MISSION_SOURCE_TYPE = "mission"
TERMINAL_STATUSES: Sequence[str] = ("done", "review", "failed", "cancelled")
RUNNING_STATUSES: Sequence[str] = ("assigned", "in_progress", "blocked")
DEFAULT_LANE_POLL_SECONDS = 5


def chat_source_id(chat_id: Any, message_key: Any) -> str:
    """``chat:<chat id>:<message key>`` — one ticket per chat message."""
    return f"{CHAT_SOURCE_TYPE}:{chat_id}:{message_key}"


def chat_origin_of(task: Any) -> Optional[str]:
    """The conversation a chat ticket belongs to (from its source id), else None."""
    if getattr(task, "source_type", None) != CHAT_SOURCE_TYPE:
        return None
    raw = str(getattr(task, "source_id", "") or "")
    parts = raw.split(":", 2)
    if len(parts) < 2 or parts[0] != CHAT_SOURCE_TYPE or not parts[1]:
        return None
    return parts[1]


def chat_tickets(db: Session, workspace_id: Any, chat_id: Any, agent_id: int, statuses: Sequence[str]):
    """This conversation's tickets for this agent, newest first."""
    try:
        return (
            db.query(BoardTask)
            .filter(
                BoardTask.workspace_id == workspace_id,
                BoardTask.assigned_agent_id == int(agent_id),
                BoardTask.source_type == CHAT_SOURCE_TYPE,
                BoardTask.source_id.like(f"{CHAT_SOURCE_TYPE}:{chat_id}:%"),
                BoardTask.status.in_(list(statuses)),
            )
            .order_by(BoardTask.id.desc())
            .all()
        )
    except Exception:  # noqa: BLE001 — a broken query means "no history", never a failed turn
        logger.debug("[CliTicketLane] chat ticket lookup failed", exc_info=True)
        return []


def previous_session_of(db: Session, workspace_id: Any, chat_id: Any, agent_id: int):
    """``(session id, host id)`` of the newest ENDED chat ticket of this
    conversation and agent, or ``None``. The session id the hooks reported wins
    over the pre-assigned one (a resumed session keeps its own id)."""
    for task in chat_tickets(db, workspace_id, chat_id, agent_id, TERMINAL_STATUSES):
        ref = getattr(task, "runtime_ref", None)
        if not isinstance(ref, dict):
            continue
        session_id = ref.get("cli_session_id") or ref.get("session_id")
        host_id = ref.get("host_id")
        if session_id and host_id:
            return str(session_id), str(host_id)
    return None


def running_predecessor_of(db: Session, task: Any) -> Optional[BoardTask]:
    """An OLDER chat ticket of the same conversation + agent that is still
    running (``in_progress``), or None. One session takes one turn at a time."""
    chat_id = chat_origin_of(task)
    if not chat_id or not getattr(task, "assigned_agent_id", None):
        return None
    for other in chat_tickets(db, task.workspace_id, chat_id, task.assigned_agent_id, ("in_progress",)):
        if other.id != task.id and other.id < task.id:
            return other
    return None


def exec_result_for(task: Any) -> dict:
    """A finished ticket as the ``execute_with_prompt`` result shape the playbook
    executor and the mission dispatcher already read (status/result/error/tokens)."""
    ref = getattr(task, "runtime_ref", None)
    ref = ref if isinstance(ref, dict) else {}
    usage = ref.get("usage") if isinstance(ref.get("usage"), dict) else {}
    try:
        tokens = int(usage.get("total_tokens") or 0) or (
            int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        )
    except (TypeError, ValueError):
        tokens = 0
    status = getattr(task, "status", None)
    base = {
        "runtime": RUNTIME_CLI,
        "task_id": task.id,
        "board_status": status,
        "tokens_used": tokens,
        "execution": {"tokens_used": tokens, "tool_calls": [], "messages": []},
    }
    if status in ("done", "review"):
        result = getattr(task, "result", None) or ""
        if status == "review":
            result = (result + "\n\n" if result else "") + (
                f"(ticket #{task.id} is held for review on the board)"
            )
        return {"status": "success", "result": result, **base}
    if status == "cancelled":
        return {"status": "cancelled", "result": "", "error": f"ticket #{task.id} was cancelled", **base}
    error = getattr(task, "error_message", None) or f"ticket #{task.id} failed"
    return {"status": "error", "result": "", "error": error, **base}


async def run_cli_ticket_and_wait(
    db: Session,
    *,
    workspace_id: Any,
    agent_id: int,
    title: str,
    prompt: str,
    source_type: str,
    source_id: str,
    timeout_s: Optional[float] = None,
    poll_s: Optional[float] = None,
    **file_kwargs: Any,
) -> dict:
    """File the ticket a step or mission task owes a session agent and wait for
    it to end (PRD-239 S3). Returns ``exec_result_for`` the ended ticket, or an
    error result that names the still-running ticket when ``timeout_s`` passes
    — the session carries on; its result lands on the board.
    """
    import asyncio
    import time

    task = file_cli_ticket(
        db, workspace_id=workspace_id, agent_id=agent_id, title=title, prompt=prompt,
        source_type=source_type, source_id=source_id, **file_kwargs,
    )
    task_id = task.id
    if poll_s is None:
        try:
            from config import config

            poll_s = float(getattr(config, "CLI_LANE_POLL_SECONDS", DEFAULT_LANE_POLL_SECONDS))
        except Exception:  # noqa: BLE001
            poll_s = float(DEFAULT_LANE_POLL_SECONDS)
    poll_s = max(0.5, float(poll_s))
    started = time.monotonic()
    while True:
        db.expire_all()  # see the host's writes, not this session's cache
        current = db.query(BoardTask).filter(BoardTask.id == task_id).first()
        if current is None:
            return {"status": "error", "error": f"ticket #{task_id} disappeared while the session ran",
                    "runtime": RUNTIME_CLI, "task_id": task_id}
        if current.status in TERMINAL_STATUSES:
            return exec_result_for(current)
        waited = time.monotonic() - started
        if timeout_s is not None and waited >= timeout_s:
            return {
                "status": "error",
                "error": (
                    f"ticket #{task_id} is still running after {int(waited)} s — the Claude Code "
                    "session carries on and its result lands on the board"
                ),
                "runtime": RUNTIME_CLI,
                "task_id": task_id,
                "timed_out": True,
            }
        await asyncio.sleep(poll_s)
