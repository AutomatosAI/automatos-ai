"""
Board Tasks API
===============

CRUD + planning endpoints for the lightweight task board (PRD-72).
Tasks follow a Kanban lifecycle: inbox -> assigned -> in_progress -> review -> blocked -> done.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, text
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import Session

from config import config
from core.auth.hybrid import get_request_context_hybrid, require_task_context
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext
from core.auth.scopes import TASKS_READ
from core.database.database import get_db
from core.models.core import BoardTask
from core.models import Agent
from core.utils.exception_telemetry import record_error
from core.utils.background_tasks import launch_guarded
from core.cli_runtime import RUNTIME_API, RUNTIME_CLI, runtime_kind_of  # PRD-234 S1a
from services.session_report import session_report_lines  # PRD-234 S2
from services.board_consent import (  # PRD-234: a human's board action is the approval
    WHY_CREATED_AND_ASSIGNED, WHY_MOVED_TO_IN_PROGRESS, WHY_RUN_NOW, actor_ref as _operator_ref,
    consent_for_created_ticket, record_operator_consent,
)
from services.board_dispatcher import notify_task_available
from services.ticket_redo import SENT_BACK, SENT_BACK_WITHOUT_A_NOTE, with_correction
from services.board_sla import PRIORITY_SLA_HOURS
from services.board_events import board_event_stream, notify_board_event

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/tasks", tags=["board-tasks"])

# "closed" (night 1, 2026-09-18): tidying a board meant marking work "done" or
# "cancelled" when neither was true — a ticket superseded by another, or simply
# no longer wanted. Closed is terminal, claims nothing about the work, and keeps
# the board honest.
VALID_STATUSES = {"inbox", "assigned", "in_progress", "review", "blocked", "done", "failed", "cancelled", "closed"}
# F194: the source kinds a request may file: the board's own create ('user') and a
# Command Centre follow-up ('activity'). Every other kind is the platform's.
USER_CREATABLE_SOURCE_TYPES = frozenset({"user", "activity"})
# #1094: a ticket with no agent cannot run, so it is never in progress. The
# board's PATCHes and platform_update_task_status refuse it in these words.
NO_AGENT_NO_PROGRESS = "Assign an agent first: a ticket with no agent cannot be in progress."
# An operator note is read on a card, not in a document.
MAX_TASK_NOTE_CHARS = 1000
# A reviewer's verdict is folded into the next attempt's prompt, so it is read
# by a model as well as a person — long enough to be specific, bounded so it
# cannot crowd out the ticket itself.
MAX_REVIEW_FEEDBACK_CHARS = 4000
# Exactly what PATCH /api/v1/tasks/{id} stores. Anything else is a 422 rather
# than a silent 200 (F060).
PATCHABLE_TASK_FIELDS = frozenset({
    "title", "description", "status", "blocked_reason", "priority", "review_mode",
    "assigned_agent_id", "result", "error_message", "tags", "planning_data",
    "note", "review_feedback",
})
VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
VALID_REVIEW_MODES = {"human", "llm", "auto"}
# F180: PRD-234's platform_create_task said 'manual' for a person's review; the
# board says 'human'. The tools take either and keep the board's word.
REVIEW_MODE_ALIASES = {"manual": "human"}


def _one_of(value: Any, allowed: Any) -> bool:
    """A body field names one of ``allowed``: a JSON list or object is not a name,
    and a 422 rather than the 500 its lookup raised (F195)."""
    return isinstance(value, str) and value in allowed


AGENT_ID_REFUSAL = "assigned_agent_id must be an agent's id, or null"


def _agent_id_of(raw: Any) -> Optional[int]:
    """A sent assigned_agent_id as an agent's id (None for null); anything else is
    a 422 (F195's candidates: 'abc' or a list was a 500). true is not agent 1."""
    if raw is None:
        return None
    try:
        if isinstance(raw, bool):
            raise TypeError("a boolean is not an id")
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=AGENT_ID_REFUSAL)


def _text_of(value: Any, field: str) -> str:
    """A sent text field, trimmed ('' for null); a number or a list is a 422, not
    the 500 its .strip() raised (F195's candidates)."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail=f"{field} must be text")
    return value.strip()


def board_review_mode(value: Any) -> Optional[str]:
    """The board's review_mode for ``value`` ('manual' is 'human'), or None when it is not one."""
    mode = REVIEW_MODE_ALIASES.get(value, value) if isinstance(value, str) else None
    return mode if mode in VALID_REVIEW_MODES else None

# PRD-171 F025: source_types the board must NOT self-execute on drag/PATCH.
# 'recipe' runs through the recipe executor; 'orchestration'/'orchestration_task'
# are mission mirrors the mission engine already owns (orchestration_board_bridge)
# — firing a board execution on them double-runs work the mission drives.
_NON_EXECUTABLE_SOURCE_TYPES = frozenset(
    {"recipe", "orchestration", "orchestration_task"}
)

# Priority → SLA deadline hours: the shared table (services.board_sla), so the
# scheduled-task lane files tickets with the same deadlines this route stamps.
_PRIORITY_SLA_HOURS = PRIORITY_SLA_HOURS


# ── Auto-report creation (mirrors heartbeat_service._auto_create_report) ───
async def _auto_create_task_report(
    db: Session,
    workspace_id: str,
    task: BoardTask,
    exec_result: Dict[str, Any],
) -> None:
    """
    Persist an agent_reports row for a completed task so it shows up in
    Reports / Deliverables / Activity Feed — same pattern heartbeats use.
    Always non-blocking: never raises, just warns on failure.
    """
    try:
        from services.report_service import ReportService, compute_execution_metrics

        agent_name = "Unknown Agent"
        if task.assigned_agent_id:
            agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
            if agent:
                agent_name = agent.name

        # Source the body from the agent's actual response, falling back to whatever
        # text was captured in task.result.
        llm_text = (
            exec_result.get("result")
            or exec_result.get("response")
            or exec_result.get("output")
            or exec_result.get("content")
            or task.result
            or ""
        )

        # Pull cost/model/duration rollup from llm_usage (window = task started→completed)
        exec_metrics = compute_execution_metrics(
            db,
            workspace_id,
            agent_id=task.assigned_agent_id,
            execution_id=getattr(task, "execution_id", None),
            started_at=getattr(task, "started_at", None),
            completed_at=getattr(task, "completed_at", None),
            extra={
                "task_id": task.id,
                "task_status": task.status,
                "trigger": "task",
            },
        )

        # Honour upstream-supplied tokens if the rollup found nothing
        if not exec_metrics.get("tokens_used"):
            usage = exec_result.get("usage") or {}
            fallback_tokens = (
                usage.get("total_tokens")
                or exec_result.get("tokens_used")
                or 0
            )
            if fallback_tokens:
                exec_metrics["tokens_used"] = fallback_tokens

        report_status = "ok" if task.status in ("done", "review") else "warning"
        if task.error_message:
            report_status = "critical"

        # Render the same shape heartbeat reports use so consumers stay uniform.
        lines = [
            f"# {agent_name} — Task Report",
            f"**Task:** {task.title}",
            f"**Status:** {task.status}",
            "",
        ]
        if task.error_message:
            lines.append("## Error")
            lines.append(str(task.error_message))
            lines.append("")
        if llm_text:
            lines.append("## Result")
            lines.append(str(llm_text))
            lines.append("")
        lines.extend(session_report_lines(exec_result))  # PRD-234 S2 (empty for API runs)
        lines.append("## Execution Metrics")
        lines.append(f"- Model: {exec_metrics.get('model') or 'unknown'}")
        lines.append(f"- LLM calls: {exec_metrics.get('llm_calls', 0)}")
        lines.append(f"- Tokens (in/out/total): "
                     f"{exec_metrics.get('input_tokens', 0)} / "
                     f"{exec_metrics.get('output_tokens', 0)} / "
                     f"{exec_metrics.get('tokens_used', 0)}")
        if exec_result.get("runtime") == RUNTIME_CLI:
            lines.append("- Cost: plan usage (subscription) — no dollar figure")
        else:
            lines.append(f"- Cost: ${exec_metrics.get('cost_usd', 0):.4f}")
        if exec_metrics.get("duration_ms") is not None:
            lines.append(f"- Duration: {exec_metrics['duration_ms']} ms")
        content = "\n".join(lines)

        # Summary is the first non-empty body line — same convention as heartbeat reports.
        summary = None
        for line in str(llm_text).split("\n"):
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                summary = (stripped[:497] + "...") if len(stripped) > 497 else stripped
                break

        svc = ReportService(db, workspace_id)
        report_result = await svc.create_report(
            agent_id=task.assigned_agent_id,
            agent_name=agent_name,
            title=f"Task: {task.title}",
            content=content,
            report_type="task",
            status=report_status,
            summary=summary,
            metrics=exec_metrics,
            linked_task_ids=[task.id],
        )
        if not report_result.get("success"):
            logger.warning(
                "[BoardTasks] Auto-report creation failed for task=%s: %s",
                task.id, report_result.get("error"),
            )
    except Exception:
        logger.error(
            "[BoardTasks] _auto_create_task_report raised for task=%s",
            getattr(task, "id", "?"),
            exc_info=True,
        )


# ── PRD-128: Unified notification dispatch ─────────────────────────
async def _dispatch_task_complete(db: Session, workspace_id, task: BoardTask) -> None:
    """Fire a ``task_complete`` event through NotificationDispatcher.

    Uses the caller's DB session so the notification row joins the
    existing transaction (no extra commits). Dispatcher never raises on
    delivery failures, but we still wrap in try/except so any programming
    error cannot block the task-completion flow.
    """
    try:
        from core.services.notification_dispatcher import NotificationDispatcher

        agent_name = None
        if task.assigned_agent_id:
            agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
            agent_name = agent.name if agent else f"agent-{task.assigned_agent_id}"

        message = None
        if task.result:
            message = str(task.result)[:500]
        elif task.description:
            message = task.description[:500]

        dispatcher = NotificationDispatcher(db, str(workspace_id))
        await dispatcher.dispatch(
            event_type="task_complete",
            title=f"Task: {task.title}",
            message=message,
            link_type="task",
            link_id=str(task.id),
            agent_id=task.assigned_agent_id,
            agent_name=agent_name,
            status="ok",
        )
    except Exception:
        logger.error(
            "[BoardTasks] task_complete dispatch failed for task %s",
            getattr(task, "id", "?"),
            exc_info=True,
        )

    # PRD-204 S3: board-task terminal choke point (success) -- every
    # completion path funnels through this helper. Fail-soft.
    from services.watch_hooks import watch_ingest_terminal

    watch_ingest_terminal(
        db,
        workspace_id=workspace_id,
        target_type="board_task",
        target_id=str(task.id),
        terminal_state="completed",
        summary=(str(task.result)[:500] if task.result else None),
    )


async def _dispatch_task_failed(db: Session, workspace_id, task: BoardTask) -> None:
    """Fire a ``task_failed`` event (PRD-161 S3).

    Mirrors ``_dispatch_task_complete`` but signals an error terminal state, so
    the user is told the task did NOT succeed — previously a crashed execution
    closed silently as 'done'. Never raises into the execution flow.
    """
    try:
        from core.services.notification_dispatcher import NotificationDispatcher

        agent_name = None
        if task.assigned_agent_id:
            agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
            agent_name = agent.name if agent else f"agent-{task.assigned_agent_id}"

        dispatcher = NotificationDispatcher(db, str(workspace_id))
        await dispatcher.dispatch(
            event_type="task_failed",
            title=f"Task failed: {task.title}",
            message=(task.error_message or "Execution failed")[:500],
            link_type="task",
            link_id=str(task.id),
            agent_id=task.assigned_agent_id,
            agent_name=agent_name,
            status="error",
        )
    except Exception:
        logger.error(
            "[BoardTasks] task_failed dispatch failed for task %s",
            getattr(task, "id", "?"),
            exc_info=True,
        )

    # PRD-204 S3: board-task terminal choke point (failure). Fail-soft.
    from services.watch_hooks import watch_ingest_terminal

    watch_ingest_terminal(
        db,
        workspace_id=workspace_id,
        target_type="board_task",
        target_id=str(task.id),
        terminal_state="failed",
        summary=(task.error_message or "Execution failed")[:500],
    )


# ── Helpers ──────────────────────────────────────────────────────────

def _enrich_with_agents(tasks: list, db: Session, workspace_id) -> list:
    """Join agent info onto task dicts.

    Agents are resolved within ``workspace_id`` only: a task whose
    ``assigned_agent_id`` points at another workspace's agent yields no ``agent``
    block rather than leaking that agent's name/icon (defense-in-depth tenant
    isolation — board reads are now reachable by per-workspace SDK keys).
    """
    agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
    if not agent_ids:
        return [t.to_dict() for t in tasks]

    agents = {
        a.id: a
        for a in db.query(Agent)
        .filter(Agent.id.in_(agent_ids), Agent.workspace_id == workspace_id)
        .all()
    }

    result = []
    for t in tasks:
        d = t.to_dict()
        agent = agents.get(t.assigned_agent_id)
        if agent:
            d["agent"] = {
                "id": agent.id,
                "name": agent.name,
                "agent_icon": getattr(agent, "premium_icon", None),
            }
        result.append(d)
    return result


# ── CRUD ─────────────────────────────────────────────────────────────

@router.post("", dependencies=[Depends(require_workspace_permission("missions:create"))])
async def create_task(
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new board task."""
    body = await request.json()

    title = _text_of(body.get("title"), "title")
    if not title:
        raise HTTPException(status_code=422, detail="title is required")

    assigned_agent_id = _agent_id_of(body.get("assigned_agent_id"))
    if assigned_agent_id is not None:
        agent = db.query(Agent).filter(
            Agent.id == assigned_agent_id,
            Agent.workspace_id == ctx.workspace_id,
        ).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Assigned agent not found in workspace")

    priority = body.get("priority", "medium")
    if not _one_of(priority, VALID_PRIORITIES):
        raise HTTPException(status_code=422, detail=f"Invalid priority: {priority}")

    review_mode = body.get("review_mode", "auto")
    if not _one_of(review_mode, VALID_REVIEW_MODES):
        raise HTTPException(status_code=422, detail=f"Invalid review_mode: {review_mode}")

    planning_data = body.get("planning_data")
    # Auto-set review status when approval_action is present
    if planning_data and isinstance(planning_data, dict) and planning_data.get("approval_action"):
        status = "review"
    else:
        status = "assigned" if assigned_agent_id else "inbox"

    # PRD-127: ephemeral attachments
    attachment_ids = body.get("attachment_ids", [])
    if attachment_ids and not isinstance(attachment_ids, list):
        raise HTTPException(status_code=422, detail="attachment_ids must be a list")

    # PRD-221 S14: a caller (e.g. a Command Centre activity card) may attach
    # the source it was created from, so the board card links back. Defaults
    # to 'user' when absent — unchanged for every existing caller.
    # F194: only a person's kinds. A mission's or playbook's step, a session or
    # a lane's ticket is filed by the platform; a request claiming one made a
    # ticket the dispatcher and the host treat as the platform's own.
    source_type = body.get("source_type") or "user"
    if not isinstance(source_type, str):
        raise HTTPException(status_code=422, detail=(
            f"source_type is one of {sorted(USER_CREATABLE_SOURCE_TYPES)}, or left out."))
    if source_type not in USER_CREATABLE_SOURCE_TYPES:
        raise HTTPException(status_code=422, detail=(
            f"source_type '{source_type}' is filed by the platform, not by a request. "
            f"Use one of {sorted(USER_CREATABLE_SOURCE_TYPES)}, or leave it out."))
    source_id = body.get("source_id")

    task = BoardTask(
        workspace_id=ctx.workspace_id,
        title=title,
        description=body.get("description"),
        raw_prompt=body.get("raw_prompt"),
        status=status,
        priority=priority,
        review_mode=review_mode,
        assigned_agent_id=assigned_agent_id,
        created_by_type="user",
        created_by_id=ctx.user.clerk_user_id or ctx.user.id,
        parent_task_id=body.get("parent_task_id"),
        source_type=source_type,
        source_id=source_id,
        tags=body.get("tags", []),
        planning_data=planning_data,
        attachment_ids=attachment_ids,  # PRD-127
        sla_deadline=datetime.now(timezone.utc) + timedelta(hours=_PRIORITY_SLA_HOURS.get(priority, 24)),
    )
    db.add(task)
    db.flush()  # the id the notice carries

    # PRD-180 S1 (F090): push the new card to subscribed Command Centres.
    # F118: a NOTIFY is delivered when its transaction commits — issue it before the commit
    # (after it, the request's closing rollback drops it).
    notify_board_event(
        db, workspace_id=ctx.workspace_id, task_id=task.id, status=task.status,
        event="task_created",
    )
    db.commit()
    db.refresh(task)

    # PRD-161: assignment = immediate dispatch via the board loop (Q39/Q40).
    # A created-as-assigned task notifies the claimant; the dispatch loop claims
    # it (FOR UPDATE SKIP LOCKED) and runs it — no inline launch, no heartbeat wait.
    if task.status == "assigned" and task.assigned_agent_id and task.source_type != "recipe":
        # PRD-234 D16: the operator created AND assigned it — on the local edition
        # that is the approval; recorded before the dispatcher can claim the row.
        consent_for_created_ticket(
            db, workspace_id=ctx.workspace_id, task=task,
            actor=_operator_ref(ctx), why=WHY_CREATED_AND_ASSIGNED,
        )
        _note_no_host_for_cli(db, task)
        notify_task_available(db, workspace_id=ctx.workspace_id, task_id=task.id)
        db.commit()  # F118: the wake rides this commit

    logger.info("[BoardTasks] Created task %d in workspace %s", task.id, ctx.workspace_id)
    return task.to_dict()


@router.get("")
async def list_tasks(
    ctx: RequestContext = Depends(require_task_context(TASKS_READ)),
    db: Session = Depends(get_db),
    status: Optional[str] = Query(None, description="Comma-separated statuses"),
    agent_id: Optional[int] = Query(None),
    priority: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    parent_task_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List board tasks with optional filters."""
    query = db.query(BoardTask).filter(BoardTask.workspace_id == ctx.workspace_id)

    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        invalid = [s for s in statuses if s not in VALID_STATUSES]
        if invalid:
            raise HTTPException(status_code=422, detail=f"Invalid status values: {invalid}")
        query = query.filter(BoardTask.status.in_(statuses))

    if agent_id is not None:
        query = query.filter(BoardTask.assigned_agent_id == agent_id)

    if priority:
        if priority not in VALID_PRIORITIES:
            raise HTTPException(status_code=422, detail=f"Invalid priority: {priority}")
        query = query.filter(BoardTask.priority == priority)

    if parent_task_id is not None:
        query = query.filter(BoardTask.parent_task_id == parent_task_id)

    if search:
        like_term = f"%{search}%"
        query = query.filter(BoardTask.title.ilike(like_term))

    # PRD-161 S5: archive — done tasks completed longer ago than the configured
    # window drop off the active board (retained in the DB, just not surfaced).
    archive_before = datetime.now(timezone.utc) - timedelta(days=config.BOARD_ARCHIVE_DONE_DAYS)
    query = query.filter(
        ~(
            (BoardTask.status == "done")
            & BoardTask.completed_at.isnot(None)
            & (BoardTask.completed_at < archive_before)
        )
    )

    total = query.count()
    tasks = (
        query.order_by(BoardTask.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "tasks": _enrich_with_agents(tasks, db, ctx.workspace_id),
        "total": total,
    }


@router.get("/stream")
async def stream_board_events(
    ctx: RequestContext = Depends(require_task_context(TASKS_READ)),
):
    """Real-time board SSE via Postgres ``LISTEN/NOTIFY`` (PRD-180 S1, F090).

    Replaces the old timed ping: the stream ``LISTEN``s the ``board_events``
    channel and forwards each board-task mutation (insert / status change /
    claim / requeue) to this client sub-second, scoped to the caller's
    workspace. A heartbeat comment keeps the connection alive but does not drive
    refreshes — real NOTIFY events do. Rides the read-only ``TASKS_READ`` scope
    (Q42): the shared hybrid auth is untouched and no write scope is introduced.
    """
    workspace_id = str(ctx.workspace_id)

    return StreamingResponse(
        board_event_stream(
            workspace_id, heartbeat_seconds=config.BOARD_SSE_HEARTBEAT_SECONDS
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{task_id}")
async def get_task(
    task_id: int,
    ctx: RequestContext = Depends(require_task_context(TASKS_READ)),
    db: Session = Depends(get_db),
):
    """Get a single board task."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    enriched = _enrich_with_agents([task], db, ctx.workspace_id)
    detail = dict(enriched[0])
    detail["cost"] = ticket_cost(db, task.id)
    return detail


def ticket_cost(db: Session, task_id: int) -> Dict[str, Any]:
    """What this ticket actually cost, from the ledger (F034).

    Every model call already writes ``llm_usage`` with
    ``execution_id = 'board_task:<id>'`` — the number existed and was simply
    never shown on the thing that spent it. A subscription (CLI session) run
    costs no API money and is labelled as such rather than shown as a bare
    $0.00 with no explanation.
    """
    try:
        row = db.execute(text("""
            SELECT COALESCE(SUM(total_cost), 0) AS usd,
                   COALESCE(SUM(total_tokens), 0) AS tokens,
                   COUNT(*) AS calls,
                   -- 'tier' is where a subscription run is actually marked;
                   -- request_type stays 'board_task' for both lanes.
                   COALESCE(SUM(total_tokens) FILTER (WHERE tier = 'subscription'), 0) AS sub_tokens,
                   COUNT(*) FILTER (WHERE tier = 'subscription') AS sub_calls
            FROM llm_usage
            WHERE execution_id = :execution_id
        """), {"execution_id": f"board_task:{task_id}"}).fetchone()
    except Exception:  # noqa: BLE001 — a cost read-out must not break the ticket
        logger.warning("[board] cost unavailable for ticket %s", task_id, exc_info=True)
        return {"available": False}

    usd = float(row.usd or 0.0) if row else 0.0
    tokens = int(row.tokens or 0) if row else 0
    calls = int(row.calls or 0) if row else 0
    sub_tokens = int(row.sub_tokens or 0) if row else 0
    sub_calls = int(row.sub_calls or 0) if row else 0
    api_tokens = tokens - sub_tokens
    api_calls = calls - sub_calls

    if sub_calls and not api_calls:
        billing, label = "subscription", (
            f"{sub_tokens:,} tokens on your CLI subscription — no API charge "
            f"({sub_calls} call{'s' if sub_calls != 1 else ''})"
        )
    elif sub_calls:
        billing, label = "mixed", (
            f"${usd:.2f} across {api_calls} API call{'s' if api_calls != 1 else ''} "
            f"({api_tokens:,} tokens), plus {sub_tokens:,} tokens on your subscription"
        )
    else:
        billing, label = "api", f"${usd:.2f} · {tokens:,} tokens · {calls} calls"

    return {
        "available": True,
        "usd": round(usd, 4),
        "tokens": tokens,
        "calls": calls,
        "subscription_tokens": sub_tokens,
        "billing": billing,
        # "$0.00" on a session run is true and misleading: it cost a seat, not
        # a dollar, and the card should say which.
        "label": label,
    }


@router.patch("/{task_id}", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def update_task(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update a board task (partial)."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    body = await request.json()
    status_before = task.status  # F190 review: a repeat of in_progress launches nothing

    # F060: this route accepted any key, returned 200 and echoed the task back,
    # while storing only the eleven fields below. `review_feedback` — the field
    # a reviewer's verdict travels in — went in and vanished, so a rejection
    # looked accepted and nothing was rejected. A PATCH that silently drops
    # what it was given is worse than one that refuses: refuse.
    unknown = sorted(set(body) - PATCHABLE_TASK_FIELDS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown field(s) for a board task: {unknown}. "
                f"This route stores exactly: {sorted(PATCHABLE_TASK_FIELDS)}. "
                "Nothing was changed."
            ),
        )

    # #1094: refused before anything changes; an agent this body assigns counts,
    # read as an id first (review LOW), never by the truth of what was sent.
    agent_after = task.assigned_agent_id
    if "assigned_agent_id" in body:
        agent_after = _agent_id_of(body["assigned_agent_id"])
    # F195's candidates: a status is checked before anything compares it (a list was a 500).
    if "status" in body and not _one_of(body["status"], VALID_STATUSES):
        raise HTTPException(status_code=422, detail=f"Invalid status: {body['status']}")
    if body.get("status") == "in_progress" and not agent_after:
        raise HTTPException(status_code=409, detail=NO_AGENT_NO_PROGRESS)
    owned = mission_runs_it(db, task) if body.get("status") in STARTING_STATUSES else None
    if owned:
        raise HTTPException(status_code=409, detail=owned)

    if "review_feedback" in body:
        # The reviewer's verdict. The dispatcher folds it into the prompt of the
        # next attempt (see _ticket_prompt) and clears it once consumed.
        feedback = body["review_feedback"]
        task.review_feedback = str(feedback)[:MAX_REVIEW_FEEDBACK_CHARS] if feedback else None

    if "title" in body:
        title = _text_of(body["title"], "title")
        if not title:
            raise HTTPException(status_code=422, detail="title cannot be empty")
        task.title = title

    if "description" in body:
        task.description = body["description"]

    if "status" in body:
        new_status = body["status"]
        old_status = task.status
        if new_status == "in_progress" and old_status != "in_progress":
            # F190: a new run starts clean, as PATCH /status does; the last run's
            # outcome goes on record (keep_previous_run) and off the card.
            keep_previous_run(task, why="moved to in progress", by=_operator_ref(ctx))
            task.started_at = datetime.now(timezone.utc)
            task.completed_at = None
            task.error_message = None
            task.result = None
        task.status = new_status
        end_session_claim(task, old_status, new_status)
        if new_status in ("done", "review", "closed"):
            task.completed_at = datetime.now(timezone.utc)
        if new_status == "blocked":
            # A person blocking a ticket that a machine had ALREADY parked used to
            # record nothing — the `blocked_at is None` guard kept the park's
            # reason, so the person's intent was invisible to everything after.
            if task.blocked_at is None:
                task.blocked_at = datetime.now(timezone.utc)
            if body.get("blocked_reason") or old_status != "blocked":
                task.blocked_reason = body.get("blocked_reason")
        if new_status != "blocked" and old_status == "blocked":
            task.blocked_at = None
            task.blocked_reason = None
        # F036: an explicit status change through this route is a person's
        # decision. A stop is recorded so no answer or grant can quietly undo
        # it; any other status lifts it.
        from services.operator_stop import apply_explicit_status

        apply_explicit_status(task, old_status, new_status, body.get("blocked_reason"), by="operator")

    if "priority" in body:
        if not _one_of(body["priority"], VALID_PRIORITIES):
            raise HTTPException(status_code=422, detail=f"Invalid priority: {body['priority']}")
        task.priority = body["priority"]

    if "review_mode" in body:
        if not _one_of(body["review_mode"], VALID_REVIEW_MODES):
            raise HTTPException(status_code=422, detail=f"Invalid review_mode: {body['review_mode']}")
        task.review_mode = body["review_mode"]

    if "assigned_agent_id" in body:
        agent_id_val = body["assigned_agent_id"]
        if agent_id_val is not None:
            agent_id_val = int(agent_id_val)
            agent = db.query(Agent).filter(
                Agent.id == agent_id_val,
                Agent.workspace_id == ctx.workspace_id,
            ).first()
            if not agent:
                raise HTTPException(status_code=404, detail="Assigned agent not found in workspace")
        task.assigned_agent_id = agent_id_val
        # Auto-transition from inbox to assigned when an agent is set
        if agent_id_val and task.status == "inbox":
            task.status = "assigned"

    if "result" in body:
        task.result = body["result"]

    if "error_message" in body:
        task.error_message = body["error_message"]

    if "tags" in body:
        task.tags = body["tags"]

    if "planning_data" in body:
        task.planning_data = body["planning_data"]

    # An operator note — a remark on the ticket that is NOT a rejection.
    # Night 1 (2026-09-18): the only way to say anything to a ticket was to
    # reject it into a redo, so a correction and a comment were the same gesture.
    # Notes land beside the session's own progress notes, in the same list the
    # card already renders.
    if body.get("note"):
        note_text = str(body["note"]).strip()[:MAX_TASK_NOTE_CHARS]
        if note_text:
            ref = dict(task.runtime_ref or {})
            ref["session_notes"] = (ref.get("session_notes") or []) + [{
                "note": note_text,
                "at": datetime.now(timezone.utc).isoformat(),
                "by": "you",
            }]
            task.runtime_ref = ref   # rebuilt, never mutated in place (JSONB)

    # Check if we need to trigger execution
    # PRD-171 F025: only user-owned board tasks self-execute on a status flip.
    # Recipe + mission-mirror rows are driven by their own engines.
    trigger_execution = (
        "status" in body
        and body["status"] == "in_progress"
        and status_before != "in_progress"
        and task.assigned_agent_id
        and task.source_type not in _NON_EXECUTABLE_SOURCE_TYPES
    )

    # PRD-128: dispatch task_complete on terminal transition
    if "status" in body and body["status"] == "done":
        await _dispatch_task_complete(db, ctx.workspace_id, task)

    # PRD-180 S1 (F090): push the mutation to subscribed Command Centres.
    # F118: a NOTIFY is delivered when its transaction commits — issue it before the commit
    notify_board_event(
        db, workspace_id=ctx.workspace_id, task_id=task.id, status=task.status,
        event="task_updated",
    )
    if (
        not trigger_execution
        and "assigned_agent_id" in body
        and task.status == "assigned"
        and task.assigned_agent_id
        and task.source_type != "recipe"
    ):
        # PRD-161: assigning notifies the dispatch loop (single spine); the loop
        # claims 'assigned' tasks only, so re-assigning a running task is a no-op.
        notify_task_available(db, workspace_id=ctx.workspace_id, task_id=task.id)
    db.commit()
    db.refresh(task)

    if trigger_execution:
        _launch_task_execution(
            task_id=task.id,
            agent_id=task.assigned_agent_id,
            workspace_id=str(ctx.workspace_id),
            prompt=task.raw_prompt or task.description or task.title,
            review_mode=task.review_mode or "auto",
            attachment_ids=task.attachment_ids,  # PRD-127
        )

    logger.info("[BoardTasks] Updated task %d", task.id)
    return task.to_dict()


@router.delete("/{task_id}", dependencies=[Depends(require_workspace_permission("missions:delete"))])
async def delete_task(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a board task."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()

    logger.info("[BoardTasks] Deleted task %d", task.id)
    return {"success": True, "deleted_id": task_id}


# ── Status shortcut (drag-and-drop) ─────────────────────────────────

def _decide(db: Session, task: BoardTask, *, seen: str, values: Dict[str, Any]) -> bool:
    """Apply ``values`` to ``task`` only while its status is still ``seen``, the
    status the deciding request read (F195). The request that wins acts on the
    ticket; one that lost matches 0 rows, its read is refreshed, and the caller
    answers ``already_decided``. The in-memory row is synced either way."""
    flipped = (
        db.query(BoardTask)
        .filter(BoardTask.id == task.id, BoardTask.workspace_id == task.workspace_id,
                BoardTask.status == seen)
        .update({getattr(BoardTask, key): value for key, value in values.items()},
                synchronize_session=False)
    )
    if not flipped:
        db.rollback()
        try:
            db.refresh(task)
        except InvalidRequestError:  # deleted while it was being decided
            raise HTTPException(status_code=404, detail="Task not found")
        return False
    for key, value in values.items():
        setattr(task, key, value)
    return True


def _refreshed(db: Session, task: BoardTask, task_id: int) -> None:
    """Re-read a ticket after its decision commits; one deleted in that instant is
    a 404, not a 500 (F195's candidates)."""
    try:
        db.refresh(task)
    except InvalidRequestError:
        raise HTTPException(status_code=404, detail=f"Ticket #{task_id} was deleted as it was decided.")


def already_decided(task: BoardTask) -> HTTPException:
    return HTTPException(status_code=422, detail=(
        f"Ticket #{task.id} was already decided (status: {task.status}); nothing ran again."))


def _record_approval(db: Session, task_id: int, *, decided_at: datetime,
                     action_result: Optional[Dict[str, Any]]) -> bool:
    """Put the approval's result on the ticket only while it is still this
    approval's 'done' (F195): a send-back that landed while the action ran keeps
    the ticket as it sent it back. True when the ticket is still this approval's."""
    kept = (func.coalesce(func.nullif(BoardTask.result, ""), json.dumps(action_result))
            if action_result else BoardTask.result)
    return bool(
        db.query(BoardTask)
        .filter(BoardTask.id == task_id, BoardTask.status == "done", BoardTask.completed_at == decided_at)
        .update({BoardTask.result: kept}, synchronize_session=False)
    )


async def _announce_approval(db: Session, workspace_id: Any, task: BoardTask) -> None:
    """PRD-128's task_complete for an approved ticket. The approval is already on
    record (F195 commits it first), so a notice that fails is logged, never a 500."""
    try:
        await _dispatch_task_complete(db, workspace_id, task)
        db.commit()
    except Exception as exc:  # noqa: BLE001 — the approval stands either way
        db.rollback()
        logger.warning("[BoardTasks] Task %d approved; its task_complete notice failed: %s", task.id, exc,
                       exc_info=True)


def _reopen_review(db: Session, task_id: int, *, decided_at: datetime,
                   finished_before: Optional[datetime]) -> None:
    """The approval's action failed: the ticket goes back to review as it was, so
    it can be approved again, unless something else has moved it since this
    approval's ``decided_at``."""
    db.rollback()
    db.query(BoardTask).filter(
        BoardTask.id == task_id, BoardTask.status == "done", BoardTask.completed_at == decided_at,
    ).update({BoardTask.status: "review", BoardTask.completed_at: finished_before}, synchronize_session=False)
    db.commit()


async def _run_approval_action(db: Session, ctx: RequestContext, approval_action: Dict[str, Any]) -> Dict[str, Any]:
    """Run a review ticket's ``approval_action`` (publish a blog post, start a blog
    mission) and say what it did; an HTTPException says why it could not."""
    action_type = approval_action.get("type")
    try:
        if action_type == "publish_blog":
            from core.services.blog_service import BlogService
            post_id = approval_action.get("post_id")
            if not post_id:
                raise HTTPException(status_code=422, detail="approval_action missing post_id")
            svc = BlogService(db, ctx.workspace_id)
            post = svc.publish_post(UUID(post_id))
            if not post:
                raise HTTPException(status_code=404, detail=f"Blog post {post_id} not found")
            logger.info("[BoardTasks] Approved: published blog post %s (%s)", post.id, post.title)
            return {
                "type": "publish_blog",
                "post_id": str(post.id),
                "title": post.title,
                "slug": post.slug,
                "status": post.status,
                "url": f"/api/widgets/blog/posts/{post.slug}?workspace_id={ctx.workspace_id}",
            }
        if action_type == "create_blog":
            # Used by VECTOR (and any agent) to suggest a blog topic for
            # founder approval. On approve, fire the standard blog mission.
            from modules.tools.discovery.handlers_blog import (
                create_blog_post_from_topic,
            )
            topic = approval_action.get("topic")
            category = approval_action.get("category") or "AI & Automation"
            if not topic:
                raise HTTPException(status_code=422, detail="approval_action missing topic")
            user_id = ctx.user.clerk_user_id if ctx.user else None
            result = await create_blog_post_from_topic(
                db,
                ctx.workspace_id,
                {"topic": topic, "category": category, "_user_id": user_id},
            )
            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Blog mission start failed: {result.get('error', 'unknown')}",
                )
            logger.info(
                "[BoardTasks] Approved: created blog mission %s for topic '%s'",
                result.get("mission_id"), topic,
            )
            return {
                "type": "create_blog",
                "mission_id": result.get("mission_id"),
                "topic": topic,
                "category": category,
                "task_count": result.get("task_count", 0),
            }
        logger.warning("[BoardTasks] Unknown approval_action type: %s", action_type)
        return {"type": action_type, "warning": "Unknown action type, task approved without side-effect"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[BoardTasks] Approval action failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Approval action failed: {e}")


@router.post("/{task_id}/approve", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def approve_task(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Approve a board task in review status.

    If the task has an approval_action in planning_data, execute it
    (e.g., publish a blog post). Then move the task to done.
    """
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "review":
        raise HTTPException(status_code=422, detail=f"Task must be in review status (currently: {task.status})")

    body = await request.json()
    action_result = None
    approval_action = (task.planning_data or {}).get("approval_action")

    # F195: the approval that moves the ticket out of review is the one that acts
    # on it. The move is committed before the action runs (the action can await an
    # LLM), so a second click finds the ticket decided instead of acting again.
    task_id, finished_before, decided_at = task.id, task.completed_at, datetime.now(timezone.utc)
    if not _decide(db, task, seen="review", values={"status": "done", "completed_at": decided_at}):
        raise already_decided(task)
    db.commit()

    try:
        if approval_action:
            action_result = await _run_approval_action(db, ctx, approval_action)
        still_approved = _record_approval(db, task_id, decided_at=decided_at, action_result=action_result)
        db.commit()  # the action's own writes and the ticket's result, together
    except Exception:
        _reopen_review(db, task_id, decided_at=decided_at, finished_before=finished_before)
        raise
    _refreshed(db, task, task_id)

    if not still_approved:
        logger.warning("[BoardTasks] Task %d: the approval's action ran, but the ticket was moved while it "
                       "ran (now: %s); its result was left as it is", task.id, task.status)
    else:
        await _announce_approval(db, ctx.workspace_id, task)
        logger.info("[BoardTasks] Task %d approved and moved to done", task.id)
    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "action_result": action_result,
    }


# F092 (night 3): a finished ticket can be sent back, and a re-run says what it
# replaces. Night 3's #484 re-run overwrote the only evidence of F093.
SENDABLE_BACK = ("review", "done")
FINISHED = ("done", "failed", "cancelled", "review")
PREVIOUS_RUNS_KEPT = 5
PREVIOUS_RESULT_CHARS = 4000


def keep_previous_run(task: Any, *, why: str, by: str, now: Optional[datetime] = None) -> None:
    """Put a finished ticket's status, result and finish time on record in
    ``planning_data.previous_runs`` (rebuilt, never mutated in place) before
    it runs again. Keeps the newest ``PREVIOUS_RUNS_KEPT``."""
    if task.status not in FINISHED and not task.result:
        return
    data = dict(task.planning_data or {})
    runs = list(data.get("previous_runs") or [])
    runs.append({
        "status": task.status,
        "result": (task.result or "")[:PREVIOUS_RESULT_CHARS],
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "why": why,
        "by": by,
        "at": (now or datetime.now(timezone.utc)).isoformat(),
    })
    data["previous_runs"] = runs[-PREVIOUS_RUNS_KEPT:]
    task.planning_data = data


@router.post("/{task_id}/reject", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def reject_task(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Reject a board task in review status with reviewer feedback (PRD-161 Q44).

    Returns the task to the SAME agent as 'assigned' — not dumped back to inbox —
    with the feedback carried into the next execution's context (review_feedback),
    so the agent redoes the work with the correction. The dispatch loop picks the
    re-assigned task up immediately. F092: a DONE ticket can be sent back the
    same way; what it had finished with is kept in its history first.
    """
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in SENDABLE_BACK:
        raise HTTPException(
            status_code=422,
            detail=f"Only a ticket in review or done can be sent back (currently: {task.status})",
        )
    if not task.assigned_agent_id:
        raise HTTPException(status_code=422, detail="Cannot reject a task with no assigned agent")

    body = await request.json()
    feedback = str(body.get("feedback") or "").strip()[:MAX_REVIEW_FEEDBACK_CHARS]
    seen = task.status
    # F198: the redo corrects this draft with every note the ticket has had, so
    # both are kept: the draft from review too (F190), the note beside the others.
    by = _operator_ref(ctx)
    keep_previous_run(task, why=SENT_BACK, by=by)
    if feedback:
        task.planning_data = with_correction(task.planning_data, feedback, by=by,
                                             at=datetime.now(timezone.utc).isoformat())

    # Q44: back to the same agent for another attempt, feedback in context.
    # F195: only from the status this request saw, so a second click (or an
    # approval that landed first) finds the ticket already decided.
    if not _decide(db, task, seen=seen, values={"status": "assigned"}):
        raise already_decided(task)
    task.started_at = None
    task.completed_at = None
    task.result = None
    task.lease_until = None
    task.attempts = 0  # a human-driven redo is a fresh attempt cycle
    task.review_feedback = feedback or SENT_BACK_WITHOUT_A_NOTE

    # Wake the dispatch loop so the redo starts immediately (single spine).
    # F118: a NOTIFY is delivered when its transaction commits — issue it before the commit
    if task.source_type != "recipe":
        notify_task_available(db, workspace_id=ctx.workspace_id, task_id=task.id)
    db.commit()
    _refreshed(db, task, task_id)

    logger.info("[BoardTasks] Task %d rejected → re-assigned to agent %s%s",
                task.id, task.assigned_agent_id,
                f" with feedback: {feedback}" if feedback else "")
    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "assigned_agent_id": task.assigned_agent_id,
        "feedback": feedback or None,
    }


def _recipe_execution_of(source_id: str) -> str:
    """A playbook step's run id: ``recipe:<execution_id>:<step>``, or the bare id."""
    parts = str(source_id or "").split(":")
    return parts[1] if len(parts) >= 3 and parts[0] == "recipe" else str(source_id or "")


# The mission's own ticket, its steps' mirrors, and a CLI agent's mission step.
MISSION_TICKET_TYPES = frozenset({"orchestration", "orchestration_task", "mission"})
GOAL_ON_A_REFUSAL_CHARS = 80
# Statuses that start work: the board never moves a mission's ticket into them.
STARTING_STATUSES = frozenset({"assigned", "in_progress"})


def mission_runs_it(db: Session, task: Any) -> Optional[str]:
    """Why the board never runs a mission's ticket (the mission's own, or one of its
    steps), naming the mission; None for any other ticket. The mission engine runs
    its steps (PRD-171 F025): Run Now re-dispatched an assigned or blocked step
    through the board, so it ran outside its mission."""
    if getattr(task, "source_type", None) not in MISSION_TICKET_TYPES:
        return None
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    run_id = getattr(task, "orchestration_run_id", None)
    step_id = getattr(task, "orchestration_task_id", None)
    if run_id is None and step_id is not None:
        step = db.get(OrchestrationTask, step_id)
        run_id = step.run_id if step is not None else None
    # Only this workspace's mission is ever named (review LOW).
    run = db.query(OrchestrationRun).filter(
        OrchestrationRun.id == run_id, OrchestrationRun.workspace_id == task.workspace_id,
    ).first() if run_id is not None else None
    if run is None:
        run_id = None
    goal = (run.goal or "").strip()[:GOAL_ON_A_REFUSAL_CHARS] if run is not None else ""
    what = "the mission" if task.source_type == "orchestration" else "a step of the mission"
    return (
        f"Ticket #{task.id} is {what}{f' “{goal}”' if goal else ''}: the mission runs its steps, "
        f"not the board. Retry or change it from the mission"
        f"{f' (/missions/{run_id})' if run_id is not None else ''}."
    )


def _running_now(db: Session, task: BoardTask) -> bool:
    """F176 (night 6): whether a run really holds this ticket — a live claim (the
    dispatch lease every run renews from its first moment, and a CLI host renews
    for its session) or, for a playbook step, its playbook run still going. The
    status word alone is not a run: #1094 read 'in_progress' with no claim and no
    execution, and Run Now answered "already running"."""
    if task.status != "in_progress":
        return False
    if task.source_type in _NON_EXECUTABLE_SOURCE_TYPES and task.source_type != "recipe":
        # A mission's step: the mission engine runs it, never the board (PRD-171
        # F025). It holds no board lease, so the lease cannot speak for it, and
        # re-dispatching it would run the step twice (review HIGH).
        return True
    lease = task.lease_until
    if lease is not None:
        lease = lease if lease.tzinfo else lease.replace(tzinfo=timezone.utc)
        if lease > datetime.now(timezone.utc):
            return True
    if task.source_type == "recipe" and task.source_id:
        from sqlalchemy import text as sa_text

        return db.execute(
            sa_text("SELECT 1 FROM recipe_executions WHERE execution_id = :e AND status IN ('pending', 'running')"),
            {"e": _recipe_execution_of(task.source_id)},
        ).first() is not None
    return False


def _redispatch_task(db: Session, task: BoardTask) -> None:
    """Reset a task to a fresh ``assigned`` claim and wake the dispatch loop.

    The shared core of the Run-Now route and the PRD-224 US-003 watch
    corrective re-run: clears the lease, attempt count, and lifecycle
    timestamps, commits, then NOTIFYs the committed row so the dispatch loop
    claims it. Recipe-mirror rows are driven by the recipe executor, never
    board-dispatched. Caller guarantees no run holds the task (``_running_now``).
    """
    task.status = "assigned"
    task.lease_until = None
    task.attempts = 0
    task.completed_at = None
    task.started_at = None
    _note_no_host_for_cli(db, task)  # a Claude Code agent's ticket says who it waits for
    # F118: a NOTIFY is delivered when its transaction commits — issue it before the commit
    if task.source_type != "recipe":
        notify_task_available(db, workspace_id=task.workspace_id, task_id=task.id)
    db.commit()
    db.refresh(task)


@router.post("/{task_id}/run-now", dependencies=[Depends(require_workspace_permission("missions:execute"))])
async def run_task_now(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-161 S5: dispatch a task immediately.

    Resets the task to a fresh ``assigned`` claim (clears lease + attempts) and
    notifies the dispatch loop, so a failed, idle, or just-created task can be
    re-run on demand from the board. A task already in_progress is left alone.
    """
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    owned = mission_runs_it(db, task)
    if owned:
        raise HTTPException(status_code=409, detail=owned)
    if not task.assigned_agent_id:
        raise HTTPException(status_code=422, detail="Assign an agent before running the task")
    # #1115: an agent whose model cannot run is never handed the task (F141's rule).
    from modules.tools.discovery.handlers_board_tasks import _cannot_take_tasks

    agent = db.query(Agent).filter(
        Agent.id == task.assigned_agent_id, Agent.workspace_id == ctx.workspace_id,
    ).first()
    unable = _cannot_take_tasks(db, agent) if agent is not None else None
    if unable:
        raise HTTPException(status_code=409, detail=unable)
    if _running_now(db, task):
        raise HTTPException(
            status_code=409,
            detail=f"Ticket #{task.id} is already running — nothing to start; it reports when it finishes.",
        )

    # PRD-234: pressing Run Now is the operator's approval — record it so the
    # gate lets the ticket through instead of parking it behind a grant.
    record_operator_consent(
        db, workspace_id=ctx.workspace_id, task_id=task.id, agent_id=task.assigned_agent_id,
        actor=_operator_ref(ctx), why=WHY_RUN_NOW,
    )
    was = task.status
    stale = was == "in_progress"  # F176: the word said running, but no run held it
    rerun = was in FINISHED
    if rerun:
        keep_previous_run(task, why="run now", by=_operator_ref(ctx))
    _redispatch_task(db, task)

    logger.info("[BoardTasks] Run Now → task %d re-dispatched to agent %s%s",
                task.id, task.assigned_agent_id, f" (was {was})" if rerun else "")
    # #1115: a Claude Code agent's ticket waits for a host that serves this
    # workspace; queued, it is claimed the moment one is back, but it has not started.
    waiting = _waiting_for_a_host(task)
    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "rerun_of": was if rerun else None,
        "started": not waiting,
        "message": (
            f"Ticket #{task.id} is queued, but nothing can start it yet: {task.blocked_reason}" if waiting
            else f"Re-running ticket #{task.id} — it was {was}; its previous result is kept in the "
            "ticket's history." if rerun
            else f"Ticket #{task.id} said in progress, but nothing was running it — started it now." if stale
            else f"Ticket #{task.id} started."
        ),
    }


def end_session_claim(task: Any, old_status: Any, new_status: Any) -> None:
    """A ticket leaving ``in_progress`` by hand ends the run that was claimed on it.

    Drop the lease and the session credential together (PRD-245). Without this a
    board gesture — drag to Blocked, drag back to In Progress — left the claim's
    lease live and the token hash on the row, so a credential whose plaintext is
    still in that session's transcript and config file kept working with no
    session running. Nothing here touches a ticket that was not running.
    """
    from services.cli_host_service import SESSION_TOKEN_HASH_KEY

    if str(old_status) != "in_progress" or str(new_status) == "in_progress":
        return
    task.lease_until = None
    ref = dict(task.runtime_ref or {})
    if SESSION_TOKEN_HASH_KEY in ref:
        ref.pop(SESSION_TOKEN_HASH_KEY, None)
        task.runtime_ref = ref   # rebuild, never mutate in place (JSONB)


@router.patch("/{task_id}/status", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def update_task_status(
    task_id: int,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update only the status of a task (for drag-and-drop on the board)."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    body = await request.json()
    new_status = _text_of(body.get("status"), "status")
    if new_status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status: {new_status}")
    if new_status == "in_progress" and not task.assigned_agent_id:
        raise HTTPException(status_code=409, detail=NO_AGENT_NO_PROGRESS)  # #1094
    owned = mission_runs_it(db, task) if new_status in STARTING_STATUSES else None
    if owned:
        raise HTTPException(status_code=409, detail=owned)

    old_status = task.status
    # F190 review: only a move INTO in_progress starts a run. Repeating it on a
    # running ticket (a double drag) wiped the live run's result and launched the
    # agent a second time; it now changes nothing. A stuck ticket has Run Now.
    starting = new_status == "in_progress" and old_status != "in_progress"
    if starting:
        keep_previous_run(task, why="moved to in progress", by=_operator_ref(ctx))  # its result is cleared below
    task.status = new_status
    end_session_claim(task, old_status, new_status)
    if starting:
        task.started_at = datetime.now(timezone.utc)
        task.completed_at = None
        task.error_message = None
        task.result = None
    if new_status in ("done", "review") and not task.completed_at:
        task.completed_at = datetime.now(timezone.utc)
    if new_status == "blocked" and task.blocked_at is None:
        task.blocked_at = datetime.now(timezone.utc)
    if new_status != "blocked" and old_status == "blocked":
        task.blocked_at = None
        task.blocked_reason = None
    # F036: the dedicated status route is a person's decision too (the board's
    # drag-and-drop lands here) — same stop rule as PATCH /{task_id}.
    from services.operator_stop import apply_explicit_status

    apply_explicit_status(task, old_status, new_status, body.get("blocked_reason"), by="operator")

    # PRD-128: dispatch task_complete on drag-to-done transitions
    if new_status == "done":
        await _dispatch_task_complete(db, ctx.workspace_id, task)

    # PRD-180 S1 (F090): push the drag-and-drop status change to Command Centres.
    # F118: a NOTIFY is delivered when its transaction commits — issue it before the commit
    notify_board_event(
        db, workspace_id=ctx.workspace_id, task_id=task.id, status=task.status,
        event="status_changed",
    )
    db.commit()
    db.refresh(task)

    # Fire-and-forget: trigger agent execution when moved to in_progress.
    # PRD-171 F025: exclude recipe + mission-mirror rows — dragging a mission
    # mirror to in_progress must not re-run work the mission engine owns.
    if (
        starting
        and task.assigned_agent_id
        and task.source_type not in _NON_EXECUTABLE_SOURCE_TYPES
    ):
        # PRD-234: dragging a ticket to In Progress is the operator's approval.
        record_operator_consent(
            db, workspace_id=ctx.workspace_id, task_id=task.id, agent_id=task.assigned_agent_id,
            actor=_operator_ref(ctx), why=WHY_MOVED_TO_IN_PROGRESS,
        )
        _launch_task_execution(
            task_id=task.id,
            agent_id=task.assigned_agent_id,
            workspace_id=str(ctx.workspace_id),
            prompt=task.raw_prompt or task.description or task.title,
            review_mode=task.review_mode or "auto",
            attachment_ids=task.attachment_ids,  # PRD-127
        )

    return {"id": task.id, "status": task.status}


class SessionDecisionBody(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=64)
    approved: bool


@router.post("/{task_id}/session-decision", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def decide_session_permission_route(
    task_id: int,
    body: SessionDecisionBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-235 W2 S3: answer a Claude Code session's permission question (the card on
    the ticket's Canvas). The host receives the answer on its next event flush."""
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    from services.cli_host_service import decide_session_permission
    try:
        return decide_session_permission(db, task, body.request_id, body.approved, _operator_ref(ctx))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{task_id}/cancel", dependencies=[Depends(require_workspace_permission("missions:update"))])
async def cancel_task(
    task_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-234 S1a: cancel a task that has not finished.

    Terminal ``cancelled`` at once, whatever lane owns it: a queued/blocked task
    simply stops being claimable; a CLI-host session is told to stop on its next
    event batch (``control: ["cancel"]``) and its late result is a no-op; an API
    run still finishes in the background but its result is dropped honestly —
    the completion writer only writes ``in_progress`` rows.
    """
    task = db.query(BoardTask).filter(
        BoardTask.id == task_id,
        BoardTask.workspace_id == ctx.workspace_id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status in ("done", "failed", "cancelled", "closed"):
        return {"id": task.id, "status": task.status, "applied": False}

    # F116: one cancel for the board and for a cancelled playbook run's step
    # tickets — it now also records who cancelled and why.
    from services.board_cancel import cancel_board_ticket

    previous = task.status
    cancel_board_ticket(db, task, by=_operator_ref(ctx), reason="cancelled on the board")
    return {"id": task.id, "status": "cancelled", "applied": True, "previous_status": previous}


# ── Immediate execution (fire-and-forget) ────────────────────────────

async def _lease_heartbeat(task_id: int) -> None:
    """PRD-171 F024: keep a long-running task's dispatch lease alive.

    Runs concurrently with the execution and renews ``lease_until`` every half
    the lease window on its OWN short-lived session, so a legitimately long run
    (> ``BOARD_DISPATCH_LEASE_SECONDS``) is never swept back to ``assigned`` and
    re-claimed. Cancelled the moment the run finishes; if the process crashes the
    heartbeat dies with it, the lease truly lapses, and the sweeper requeues the
    dead run — exactly the intended behaviour. Best-effort throughout: a failed
    renewal is logged and retried, never propagated into the run.
    """
    from core.database.database import SessionLocal
    from services.board_dispatcher import renew_lease

    lease_seconds = config.BOARD_DISPATCH_LEASE_SECONDS
    # Renew well within the window so a slow tick never lets the lease lapse.
    interval = max(1.0, lease_seconds / 2)
    try:
        while True:
            # F176: renew FIRST, so a directly launched run (a drag to In Progress,
            # the status tool) holds a live claim from its first moment, not after
            # half a lease window. A live lease is what "already running" means.
            hb = SessionLocal()
            try:
                if not renew_lease(hb, task_id, lease_seconds=lease_seconds):
                    # Row is no longer in_progress (finished/failed/requeued) —
                    # nothing more to renew.
                    break
            except Exception:
                logger.warning(
                    "[BoardTasks] lease heartbeat failed for task %d", task_id,
                    exc_info=True,
                )
                hb.rollback()
            finally:
                hb.close()
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise


def _estimate_board_task_cost_usd(db, task_id: int, agent_id: int) -> float:
    """PRD-192 S3: a REAL dollar estimate for the pending board task.

    Before this the approval gate always received the ``estimated_cost_usd=0.0``
    default, so an ``auto_below_budget`` policy auto-approved every task (C.5 —
    the ceiling could never bind). Estimator: prompt tokens of the task's text
    (``raw_prompt`` else title+description) + the agent's configured output cap
    (``model_config.max_tokens``, else the model registry's own ceiling),
    priced by ``modules.policy.pricing`` against the task agent's model — the
    flat rate applies only inside pricing as the registry-miss last resort.

    Never raises: 0.0 when nothing is resolvable (the gate then behaves as
    before for that task).
    """
    try:
        from core.context_guard import count_tokens
        from modules.policy import pricing as _pricing

        task = db.query(BoardTask).get(task_id)
        if task is None:
            return 0.0
        prompt_text = task.raw_prompt or " ".join(
            p for p in (task.title, task.description) if p
        )
        est_in = count_tokens(prompt_text or "")

        model_id = None
        est_out = 0
        agent = db.query(Agent).get(agent_id) if agent_id else None
        mc = getattr(agent, "model_config", None) or {}
        if isinstance(mc, dict):
            model_id = mc.get("model_id")
            try:
                est_out = int(mc.get("max_tokens") or 0)
            except (TypeError, ValueError):
                est_out = 0
        if model_id and not est_out:
            try:
                from core.models import LLMModel

                m = db.query(LLMModel).filter_by(model_id=model_id).first()
                est_out = int(m.max_output_tokens or 0) if m else 0
            except Exception:
                est_out = 0

        if model_id:
            priced = _pricing.estimate_cost_usd(db, model_id, est_in, est_out)
            if priced is not None:
                return float(priced)
        # No model / registry miss ⇒ the ONE flat last-resort, inside pricing.
        return _pricing.price_total_tokens_usd(db, model_id, est_in + est_out)
    except Exception:
        logger.warning(
            "[BoardTasks] cost estimate failed for task %s — passing 0.0",
            task_id, exc_info=True,
        )
        return 0.0


def _board_task_blocked_pending_approval(
    db, task_id: int, agent_id: int, workspace_id: str
) -> bool:
    """PRD-181 S2: return True if the board task must wait for human approval.

    - An active (granted, unexpired) grant for this task ⇒ proceed (False).
    - Otherwise evaluate the workspace approval policy. If it asks, block the
      task (status → ``blocked``, ``blocked_reason`` referencing the grant) and
      return True so the caller does NOT execute it.

    Fail posture (PRD-192 S1, locked #4): under the policy plane's ENFORCE
    stages (``destructive`` | ``on``) an approval-gate ERROR blocks the task
    pending approval — an errored governance gate must never launch autonomous
    work. In ``off``/``shadow`` the historical fail-open stands (the per-tool
    PolicyGate still applies to every tool the task's agent invokes).
    """
    try:
        from core.services.approval_grants import find_active_grant
        from core.models.approval_grants import SUBJECT_BOARD_TASK
        from services.board_approval import evaluate_board_task_approval

        # Already authorised? Proceed.
        if find_active_grant(
            db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=str(task_id)
        ) is not None:
            return False

        outcome = evaluate_board_task_approval(
            db, workspace_id=workspace_id, task_id=task_id, agent_id=agent_id,
            # PRD-192 S3: a real priced figure — auto_below_budget can bind (C.5).
            estimated_cost_usd=_estimate_board_task_cost_usd(db, task_id, agent_id),
        )
        if not outcome.requires_approval:
            return False

        # Block the task until a human grants the pending grant.
        task = db.query(BoardTask).get(task_id)
        if task is not None and task.status == "in_progress":
            task.status = "blocked"
            task.blocked_at = datetime.now(timezone.utc)
            grant_id = getattr(outcome.grant, "id", None)
            task.blocked_reason = (
                f"Awaiting human approval (grant #{grant_id}): {outcome.reason}"
            )
            db.commit()
            logger.info(
                "[BoardTasks] task %s blocked pending approval grant #%s",
                task_id, grant_id,
            )
        return True
    except Exception:
        try:
            from modules.policy.flag import enforcement_active

            _fail_closed = enforcement_active()
        except Exception:
            _fail_closed = False

        if not _fail_closed:
            logger.warning(
                "[BoardTasks] approval gate errored for task %s — proceeding "
                "(per-tool PolicyGate still applies)", task_id, exc_info=True,
            )
            return False

        # Enforce stage: BLOCK pending approval, never launch on a gate error.
        logger.error(
            "[BoardTasks] approval gate errored for task %s — BLOCKED pending "
            "approval (policy plane enforce stage fails closed)", task_id,
            exc_info=True,
        )
        try:
            db.rollback()  # the failed gate may have poisoned the transaction
            task = db.query(BoardTask).get(task_id)
            if task is not None and task.status == "in_progress":
                task.status = "blocked"
                task.blocked_at = datetime.now(timezone.utc)
                task.blocked_reason = (
                    "Approval gate errored — blocked pending approval "
                    "(policy plane enforce stage fails closed)"
                )
                db.commit()
        except Exception:
            logger.warning(
                "[BoardTasks] could not mark task %s blocked after gate error "
                "— task still NOT launched", task_id, exc_info=True,
            )
        return True


def _agent_runtime_kind(db: Session, agent_id: int) -> str:
    """PRD-234 S1a: the assigned agent's runtime (``api`` unless it declares ``cli``)."""
    try:
        row = db.query(Agent.configuration).filter(Agent.id == agent_id).first()
    except Exception:  # noqa: BLE001 — a test double or a broken session
        # The factory guard (AgentFactory._runtime_mismatch) is the second line
        # of defence for cli agents; a lookup this lane cannot make is treated as
        # the default runtime so an API run's fate never changes here.
        logger.debug("[BoardTasks] runtime lookup unavailable for agent %s", agent_id, exc_info=True)
        return RUNTIME_API
    configuration = None
    if row is not None:
        try:
            configuration = row[0]  # a (configuration,) row
        except (TypeError, IndexError, KeyError):
            # A test double or a full Agent object instead of a column row: read
            # the attribute; anything unreadable is an API agent (today's default).
            configuration = getattr(row, "configuration", None)
    return runtime_kind_of(configuration)


def _waiting_for_a_host(task: Any) -> bool:
    """Whether a ticket's line says it waits for a CLI host (_note_no_host_for_cli's words)."""
    from services.cli_ticket_lane import NO_HOST_REASON, is_no_cli_host_reason

    reason = getattr(task, "blocked_reason", None)
    return reason == NO_HOST_REASON or is_no_cli_host_reason(reason)


def _note_no_host_for_cli(db: Session, task: "BoardTask") -> bool:
    """A ``cli`` agent's ticket waits for the paired host; while none is online
    the ticket says so (the lane's own line), cleared once one is back. Returns
    True when the row changed. No-op for API-runtime agents."""
    # A test double or a partial row may carry no assignee: nothing to note.
    agent_id = getattr(task, "assigned_agent_id", None) if task is not None else None
    if not agent_id:
        return False
    if _agent_runtime_kind(db, agent_id) != RUNTIME_CLI:
        return False
    from services.cli_ticket_lane import (
        NO_HOST_REASON, agent_cli_provider, host_online, is_no_cli_host_reason, no_cli_host_reason_for,
    )
    ours = task.blocked_reason == NO_HOST_REASON or is_no_cli_host_reason(task.blocked_reason)
    if not host_online(db, task.workspace_id):
        wanted = NO_HOST_REASON
    else:
        # CLI adapter design §8.2: online, but does any host run THIS agent's CLI?
        wanted = no_cli_host_reason_for(db, task.workspace_id, agent_cli_provider(db, agent_id))
    if wanted:
        if task.blocked_reason != wanted:
            task.blocked_reason = wanted
            return True
    elif ours:
        task.blocked_reason = None
        return True
    return False


def _park_for_cli_host(db: Session, task_id: int, workspace_id: str, agent_id: int) -> None:
    """A ``cli`` agent's ticket is never executed by this process. Leave it
    ``assigned`` (reverting a direct-launch flip to ``in_progress``) so a paired
    CLI host claims it, and wake claimants. PRD-234 §Terms / review §B3."""
    task = db.query(BoardTask).get(task_id)
    changed = False
    if task is not None and task.status == "in_progress":
        task.status = "assigned"
        task.lease_until = None
        changed = True
    # 2026-09-07 (owner: "it just goes back to assigned"): a parked ticket with
    # no host online says so instead of sitting silently in 'assigned'.
    if _note_no_host_for_cli(db, task):
        changed = True
    if changed:
        notify_board_event(
            db, workspace_id=workspace_id, task_id=task_id, status="assigned",
            event="task_updated",
        )
    notify_task_available(db, workspace_id=workspace_id, task_id=task_id)
    db.commit()  # F118: the notices ride this commit
    logger.info(
        "[BoardTasks] task %d belongs to cli agent %d — parked 'assigned' for the CLI host",
        task_id, agent_id,
    )


def ending_summary(task: Any) -> Optional[str]:
    """PRD-238 S5: one honest line about how a session ended, from ``runtime_ref``.

    Reads only what the CLI host recorded (exit reason, permission denials,
    the last tool, files touched, attempt) — never transcript text. Returns
    None when the ticket carries no runtime reference (an API-run ticket).
    """
    ref = getattr(task, "runtime_ref", None)
    if not isinstance(ref, dict) or not ref:
        return None
    bits: List[str] = []
    reason = ref.get("exit_reason")
    if reason:
        bits.append(f"exit: {reason}")
    denials = ref.get("denials")
    if isinstance(denials, int) and denials > 0:
        bits.append(f"{denials} permission denial{'s' if denials != 1 else ''}")
    tools = ref.get("recent_tools")
    if isinstance(tools, (list, tuple)) and tools:
        last = tools[-1]
        # F168: the host's entries carry ``tool``; ``name`` is the older shape.
        last_name = (last.get("tool") or last.get("name")) if isinstance(last, dict) else str(last)
        if last_name:
            bits.append(f"last tool: {last_name}")
    files = ref.get("files_touched")
    if isinstance(files, (list, tuple)) and files:
        bits.append(f"{len(files)} file{'s' if len(files) != 1 else ''} touched")
    attempt = ref.get("attempt")
    if isinstance(attempt, int) and attempt > 1:
        bits.append(f"attempt {attempt}")
    return "; ".join(bits)[:500] or None


# A later turn's result is only an improvement if it says more. Night 1
# (2026-09-18) lost ticket #255's six delivered files when a re-claim wrote a
# 488-character "I produced nothing" over the real write-up (F013).
RESULT_KEEP_RATIO = 0.5


def _kept_result(existing: Optional[str], incoming: Optional[str]) -> Optional[str]:
    """Whichever of the two actually reports the work.

    An incoming result replaces the old one unless it is substantially shorter —
    then the longer account is kept and the newer one appended beneath it, so
    nothing is lost either way and the ticket still shows what the last turn said.
    """
    if not incoming:
        return existing
    if not existing:
        return incoming
    if len(incoming) >= len(existing) * RESULT_KEEP_RATIO:
        return incoming
    return f"{existing}\n\n---\n\n_A later run reported:_ {incoming}"


async def finalize_board_task_run(
    db: Session,
    *,
    task_id: int,
    workspace_id: str,
    agent_id: int,
    exec_result: Optional[Dict[str, Any]],
    review_mode: str = "auto",
    force_review: bool = False,
) -> Optional[str]:
    """PRD-234 S1a: the ONE completion writer for a board-task run.

    Extracted from ``_launch_task_execution`` with its behaviour unchanged (PRD-171
    F023: an error result closes ``failed``, never ``done``; a success closes
    ``done``/``review`` per ``review_mode``, dispatches ``task_complete`` only on
    ``done`` and writes the report row) so an API run and a CLI-host session result
    land identically. ``force_review`` (a session with permission denials) turns an
    auto ``done`` into ``review`` — "couldn't run the tests" never reads as finished.
    A ``cancelled`` result closes ``cancelled``.

    Returns the terminal status written, or ``None`` when the task was no longer
    ``in_progress`` (finished, cancelled or requeued meanwhile) — the write is
    skipped, never forced.
    """
    exec_result = exec_result or {}
    exec_status = exec_result.get("status")

    # Extract response text
    llm_text = (
        exec_result.get("result")
        or exec_result.get("response")
        or exec_result.get("output")
        or exec_result.get("content")
        or ""
    )

    # F175 review (MEDIUM): three writers close a run here (its own result, a CLI
    # host's result, the stall sweep). Lock the row so a second writer waits for the
    # first, sees its ending, and leaves it, instead of both reading in_progress
    # and the later commit overwriting the earlier one.
    task = db.get(BoardTask, task_id, with_for_update=True, populate_existing=True)
    if not task or task.status != "in_progress":
        return None

    if exec_status == "error":
        task.status = "failed"
        task.error_message = str(
            exec_result.get("error") or "Agent execution failed"
        )[:500]
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
        await _dispatch_task_failed(db, workspace_id, task)
        # Surface failures the same way successes are surfaced. The result text is
        # blanked (the error is on the task); the session facts stay so a failed
        # Claude Code session's report still says what ran (PRD-234 S2).
        await _auto_create_task_report(
            db, workspace_id, task,
            {**exec_result, "result": "", "tokens_used": 0},
        )
        db.commit()
        return task.status

    if exec_status == "cancelled":
        task.status = "cancelled"
        task.completed_at = datetime.now(timezone.utc)
        task.lease_until = None
        notify_board_event(  # F118: before the commit it rides
            db, workspace_id=workspace_id, task_id=task_id, status="cancelled",
            event="task_cancelled",
        )
        db.commit()
        # PRD-238 S5: "supervised — I'll report back when it's done" must hold
        # for EVERY ending. A cancelled session only fired a board event; its
        # watch stayed `watching` forever and the chat never heard. Fail-soft,
        # like the completed/failed dispatches above.
        from services.watch_hooks import watch_ingest_terminal

        watch_ingest_terminal(
            db,
            workspace_id=workspace_id,
            target_type="board_task",
            target_id=str(task.id),
            terminal_state="cancelled",
            summary=ending_summary(task),
        )
        return task.status

    task.result = _kept_result(task.result, str(llm_text) if llm_text else None)
    task.error_message = None
    # F183 (night 6, #1097): a result that only asks the owner is not finished
    # work either. The ticket waits behind its question (Questions + Telegram),
    # blocked as platform_ask_human parks one, and the answer re-runs it.
    from services.ticket_owner_ask import park_if_the_result_asks

    if await park_if_the_result_asks(db, task=task, workspace_id=workspace_id, agent_id=agent_id,
                                     output=str(llm_text or ""), exec_result=exec_result):
        return task.status
    # F014 (night 1, #153): a result that names a file the workspace does not
    # have is not finished work, however well it reads.
    from services.result_files import check_named_files

    try:
        file_check = await check_named_files(
            task, str(llm_text or ""), workspace_id, db=db,
            projects_dir=getattr(config, "LOCAL_PROJECTS_DIR", "") or None,
        )
    except Exception:  # noqa: BLE001 — a check that breaks is not a verdict; the ticket still closes
        logger.warning("[board] ticket %s: the named-file check failed", task_id, exc_info=True)
        file_check = None
    if file_check is not None:
        task.result = f"{task.result or ''}\n\n{file_check.note}".strip()
        force_review = force_review or file_check.review
    # F093 (night 3, #484): a result that is only skipped tool calls did nothing.
    from services.result_substance import nothing_done_note

    nothing_done = nothing_done_note(str(llm_text or ""))
    if nothing_done:
        task.result = f"{task.result or ''}\n\n{nothing_done}".strip()
        force_review = True
    task.status = "done" if (review_mode == "auto" and not force_review) else "review"
    task.completed_at = datetime.now(timezone.utc)
    # A ticket that ends well must not still carry the error of an earlier
    # attempt: night 1 left four tickets simultaneously "done" and "failed"
    # depending which field you read, because a retry that succeeded never
    # cleared error_message.
    task.error_message = None
    # PRD-128: dispatch task_complete only on terminal 'done'
    if task.status == "done":
        await _dispatch_task_complete(db, workspace_id, task)
    # Persist a report row for every completed task so it surfaces
    # in Reports / Deliverables / Activity Feed (mirrors heartbeats).
    await _auto_create_task_report(db, workspace_id, task, exec_result)
    db.commit()
    return task.status


def _park_over_budget(db: Session, task_id: int, reason: str) -> None:
    """Hold a ticket that would have started over the day's ceiling.

    ``blocked`` with the reason on it, so the board says why and the ticket
    comes back on its own once the ceiling is raised or the day rolls over —
    it is not failed, and nothing it might have produced is lost.
    """
    try:
        task = db.query(BoardTask).get(task_id)
        if not task or task.status not in ("assigned", "in_progress"):
            return
        task.status = "blocked"
        task.blocked_at = datetime.now(timezone.utc)
        task.blocked_reason = reason
        task.lease_until = None
        notify_board_event(  # F118: before the commit it rides
            db, workspace_id=str(task.workspace_id), task_id=task.id,
            status="blocked", event="task_updated",
        )
        db.commit()
    except Exception:  # noqa: BLE001 — the guard must not become its own failure
        logger.error("[spend-guard] could not park ticket %s", task_id, exc_info=True)
        db.rollback()


def _launch_task_execution(
    task_id: int,
    agent_id: int,
    workspace_id: str,
    prompt: str,
    review_mode: str = "auto",
    attachment_ids: Optional[list] = None,  # PRD-127
):
    """Launch agent execution for a board task as a background coroutine."""

    async def _run():
        from core.database.database import SessionLocal
        db = SessionLocal()
        # PRD-171 F024: heartbeat the dispatch lease for the life of the run.
        heartbeat = asyncio.ensure_future(_lease_heartbeat(task_id))
        try:
            # PRD-181 S2 (F060): board-task approval gate. Before an autonomous
            # board task executes, run it through the SAME approval primitive
            # missions use. If the workspace policy asks (always_ask / over the
            # dollar ceiling), a durable, revocable, expiring approval-grant is
            # created and the task is BLOCKED until a human grants it — not run,
            # not auto-allowed. On grant, the grant API re-queues the task.
            # F034: the day's ceiling stops NEW autonomous work. Checked here,
            # where the next dollar would be spent, and never on a path a human
            # is waiting on. Work already running is untouched.
            from services.daily_spend_guard import refuse_new_work

            _over_budget = refuse_new_work(db, workspace_id, f"board task {task_id}")
            if _over_budget:
                _park_over_budget(db, task_id, _over_budget)
                heartbeat.cancel()
                db.close()
                return

            if _board_task_blocked_pending_approval(db, task_id, agent_id, workspace_id):
                heartbeat.cancel()
                db.close()
                return

            # PRD-234 S1a: a ``runtime: cli`` agent's ticket never runs here — the
            # paired CLI host claims it (the user's own Claude Code session). One
            # check covers all four launch sites; zero factory calls.
            if _agent_runtime_kind(db, agent_id) == RUNTIME_CLI:
                _park_for_cli_host(db, task_id, workspace_id, agent_id)
                heartbeat.cancel()
                db.close()
                return

            from modules.agents.factory.agent_factory import AgentFactory

            factory = AgentFactory(db_session=db)
            exec_result = await factory.execute_with_prompt(
                agent=agent_id,
                prompt=prompt,
                context={
                    "source": "board_task",
                    "task_id": task_id,
                    "workspace_id": workspace_id,
                },
                use_memory=False,
                attachment_ids=attachment_ids,  # PRD-127
            )

            # PRD-234 S1a: ONE completion writer for both runtimes (an API run
            # here, a CLI-host result via api/cli_hosts). Behaviour is the PRD-171
            # F023 block, extracted unchanged — see finalize_board_task_run.
            terminal = await finalize_board_task_run(
                db,
                task_id=task_id,
                workspace_id=workspace_id,
                agent_id=agent_id,
                exec_result=exec_result,
                review_mode=review_mode,
            )

            logger.info(
                "[BoardTasks] Agent %d completed task %d → %s",
                agent_id, task_id, terminal or "?",
            )
        except Exception as e:
            logger.error(
                "[BoardTasks] Task %d execution failed: %s", task_id, e, exc_info=True,
            )
            record_error(
                subsystem="board",
                operation="execute_task",
                error=e,
                workspace_id=workspace_id,
                agent_id=agent_id,
                extra={"task_id": task_id},
            )
            try:
                task = db.query(BoardTask).get(task_id)
                if task and task.status == "in_progress":
                    # PRD-161 S3: fail honestly — a crashed execution becomes
                    # terminal 'failed', not a silent 'done' with an error blob.
                    task.status = "failed"
                    task.error_message = str(e)[:500]
                    task.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    await _dispatch_task_failed(db, workspace_id, task)
                    # Surface failures the same way successes are surfaced.
                    await _auto_create_task_report(
                        db, workspace_id, task,
                        {"result": "", "tokens_used": 0},
                    )
                    db.commit()
            except Exception:
                db.rollback()
        finally:
            # PRD-171 F024: stop heartbeating the moment the run ends — the task
            # has reached its terminal state, so the lease should now be allowed
            # to lapse for any genuinely-abandoned row.
            heartbeat.cancel()
            try:
                await heartbeat
            except (asyncio.CancelledError, Exception):
                pass
            db.close()

    # Guarded launch: a strong ref prevents GC-cancellation mid-run and an
    # uncaught crash is recorded. _run() already records its own caught
    # failures; the boot reaper (W1-S6) recovers any row stranded by a restart.
    launch_guarded(
        _run(),
        subsystem="board",
        operation="execute_task",
        workspace_id=workspace_id,
        agent_id=agent_id,
        extra={"task_id": task_id},
    )


# ── Planning mode ────────────────────────────────────────────────────

# Tight pack budget for board planning: enough to ground questions in workspace
# knowledge + prior failures without dominating a question-generation prompt.
_BOARD_PLANNING_PACK_TOKENS = 4000


async def _board_planning_context(db: Session, workspace_id, goal: str) -> str:
    """The ONE platform planning pack (PRD-164 S1, Q61) for board planning.

    Same assembler as MissionPlanner and AutoBrain —
    ``ContextService.build_planning_context``. Empty string on any failure so
    planning never breaks because context assembly did.
    """
    try:
        from modules.context.service import ContextService

        pack = await ContextService(db).build_planning_context(
            goal=goal,
            workspace_id=str(workspace_id),
            max_tokens=_BOARD_PLANNING_PACK_TOKENS,
        )
        return pack.content if not pack.is_empty else ""
    except Exception:
        logger.warning(
            "[BoardTasks] planning context pack unavailable — continuing without it",
            exc_info=True,
        )
        return ""


@router.post("/plan", dependencies=[Depends(require_workspace_permission("missions:create"))])
async def plan_task(
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Start planning mode: accepts a raw prompt, returns 3-4 clarifying
    multiple-choice questions from the orchestrator LLM.
    """
    body = await request.json()
    raw_prompt = (body.get("raw_prompt") or "").strip()
    if not raw_prompt:
        raise HTTPException(status_code=422, detail="raw_prompt is required")

    from core.llm.manager import LLMManager

    llm = LLMManager(
        service_name="orchestrator",
        workspace_id=str(ctx.workspace_id),
        request_type="planning",
    )

    system = (
        "You are a task planning assistant. The user wants to create a task for an AI agent.\n"
        "Generate exactly 3-4 multiple choice questions to clarify the task scope.\n\n"
        "Return JSON ONLY in this format:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "id": "q1",\n'
        '      "question": "What is the scope?",\n'
        '      "options": ["Option A", "Option B", "Option C"],\n'
        '      "default": 0\n'
        "    }\n"
        "  ],\n"
        '  "suggested_title": "A clear task title",\n'
        '  "suggested_priority": "medium"\n'
        "}"
    )

    # PRD-164 S1 (Q61): ground the questions in what the platform knows —
    # workspace knowledge, prior mission failures, roster.
    messages = [{"role": "system", "content": system}]
    planning_context = await _board_planning_context(db, ctx.workspace_id, raw_prompt)
    if planning_context:
        messages.append({
            "role": "system",
            "content": (
                f"{planning_context}\n\n"
                "Use this platform context: ground your questions in the "
                "workspace's actual documents and agents, and if similar work "
                "previously failed, ask questions that steer the task away "
                "from the failed approach."
            ),
        })
    messages.append({"role": "user", "content": f"Plan this task: {raw_prompt}"})

    response = await llm.generate_response(messages=messages)

    # Extract text from response
    text = _extract_llm_text(response)

    # Parse JSON from the response
    try:
        parsed = _parse_json_from_text(text)
    except ValueError:
        logger.warning("[BoardTasks] Could not parse planning JSON: %s", text[:500])
        parsed = {
            "questions": [],
            "suggested_title": raw_prompt[:100],
            "suggested_priority": "medium",
            "raw_response": text,
        }

    return {"planning": parsed, "raw_prompt": raw_prompt}


@router.post("/plan/refine", dependencies=[Depends(require_workspace_permission("missions:create"))])
async def refine_task(
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Refine a task: accepts raw_prompt + answers to planning questions,
    returns a refined description + suggested title + priority.
    """
    body = await request.json()
    raw_prompt = (body.get("raw_prompt") or "").strip()
    answers = body.get("answers", [])

    if not raw_prompt:
        raise HTTPException(status_code=422, detail="raw_prompt is required")

    from core.llm.manager import LLMManager

    llm = LLMManager(
        service_name="orchestrator",
        workspace_id=str(ctx.workspace_id),
        request_type="planning",
    )

    system = (
        "You are a task planning assistant. Based on the user's task description and their answers "
        "to clarifying questions, generate a refined, clear task description that an AI agent can execute.\n\n"
        "Return JSON ONLY:\n"
        "{\n"
        '  "title": "Clear task title",\n'
        '  "description": "Detailed task description with specific instructions based on the answers",\n'
        '  "priority": "medium",\n'
        '  "suggested_tags": ["tag1", "tag2"]\n'
        "}"
    )

    answers_text = "\n".join(
        [f"Q: {a.get('question', '')} -> A: {a.get('answer', '')}" for a in answers]
    )

    response = await llm.generate_response(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Original request: {raw_prompt}\n\nAnswers:\n{answers_text}"},
        ]
    )

    text = _extract_llm_text(response)

    try:
        parsed = _parse_json_from_text(text)
    except ValueError:
        logger.warning("[BoardTasks] Could not parse refine JSON: %s", text[:500])
        parsed = {
            "title": raw_prompt[:100],
            "description": raw_prompt,
            "priority": "medium",
            "suggested_tags": [],
            "raw_response": text,
        }

    return {"refined": parsed, "raw_prompt": raw_prompt}


# ── Internal helpers ─────────────────────────────────────────────────

def _extract_llm_text(response) -> str:
    """Pull plain text out of an LLM response (handles LLMResponse, dict, or string)."""
    # Handle LLMResponse objects (have .content attribute)
    if hasattr(response, "content"):
        return str(response.content or "")

    if isinstance(response, dict):
        text = (
            response.get("content")
            or response.get("result")
            or response.get("response")
            or response.get("output")
            or ""
        )
        if isinstance(text, dict):
            text = text.get("content") or str(text)
        # Handle choices array (OpenAI format)
        if not text and "choices" in response:
            choices = response["choices"]
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                text = msg.get("content", "")
        return str(text)
    return str(response)


def _parse_json_from_text(text: str) -> dict:
    """Extract JSON from LLM text that may contain markdown fences."""
    cleaned = text.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in response")
