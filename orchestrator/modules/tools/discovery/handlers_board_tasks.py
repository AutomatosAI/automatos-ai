"""Board task CRUD, assignment, and status handlers for PlatformActionExecutor (PRD-72)."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

# list_board_tasks: the page size the model may ask for. "Close all the blocked
# tasks" needs to SEE them all; 50 hid 121 blocked tasks behind a page (2026-09-02).
MAX_LIST_TASKS_LIMIT = 200
# update_board_task_status: how many task_ids one bulk call may carry. The chat
# loop caps a tool at 8 calls per turn, so one-id-per-call could never close more
# than eight tasks — the bulk path exists so "close all …" is one honest call.
MAX_BULK_TASK_IDS = 100

logger = logging.getLogger(__name__)


def _notify_board_safe(db: Session, workspace_id: UUID, task_id: int, status: str, event: str) -> None:
    """Fire the board SSE NOTIFY for an agent-side board write, fail-soft.

    PRD-227 US-001: agent moves must push the same ``board_changed`` frame the
    human PATCH path emits (``api/board_tasks.py`` notify_board_event call sites)
    so a card an agent moves lights up the open Command Center at SSE latency,
    not on the next stale refetch. Best-effort exactly like
    ``services/board_events.py``: a NOTIFY (or import) failure logs and continues
    — it must NEVER raise into the tool call. No new SSE event names — payload
    shape and channel are the human path's.
    """
    try:
        from services.board_events import notify_board_event

        notify_board_event(
            db, workspace_id=workspace_id, task_id=task_id, status=status, event=event
        )
    except Exception:  # noqa: BLE001 — NOTIFY is an optimisation, not a guarantee
        logger.debug(
            "[BoardTasks] board NOTIFY skipped for task %s", task_id, exc_info=True
        )


def _notify_dispatch_safe(db: Session, workspace_id: UUID, task_id: int) -> None:
    """Wake the board dispatch loop for a chat-side assigned write, fail-soft.

    PRD-224 US-001: a chat-created/assigned/re-queued ticket that lands in the
    ``assigned`` state must be claimed on the dispatcher's LISTEN wake, not the
    fallback poll — mirroring the HTTP layer's ``notify_task_available`` call
    sites (``api/board_tasks.py`` create :398 / assign :632 / reject :816 /
    run-now :862). Best-effort exactly like ``notify_task_available`` itself and
    the PRD-227 ``_notify_board_safe`` beside it: a NOTIFY (or import) failure
    logs and continues — it must NEVER raise into the tool call.
    """
    try:
        from services.board_dispatcher import notify_task_available

        notify_task_available(db, workspace_id=workspace_id, task_id=task_id)
    except Exception:  # noqa: BLE001 — NOTIFY is an optimisation, not a guarantee
        logger.debug(
            "[BoardTasks] dispatch NOTIFY skipped for task %s", task_id, exc_info=True
        )


def _is_dispatch_claimable(task) -> bool:
    """True when a task is in the state the board dispatch loop claims: ``assigned``
    with an agent and not a recipe mirror (the recipe executor drives those). This
    is the exact guard the HTTP layer notifies on (``api/board_tasks.py`` :397 and
    :624-629), so a chat-filed ticket wakes the dispatcher on the same condition.
    """
    return (
        task.status == "assigned"
        and task.assigned_agent_id is not None
        and getattr(task, "source_type", None) != "recipe"
    )


def _resolve_active_agent_by_name(db: Session, workspace_id: UUID, agent_name: str):
    """Resolve an agent NAME to a single ACTIVE agent for a board write.

    P224-RVW-4: the write layer must honor the same contract AutoBrain's ASSIGN
    classifier does — resolve only against ACTIVE agents (the same source
    ``AutoBrain._active_agents`` uses) and NEVER silently ``.first()`` on an
    ambiguous name. ``Agent.name`` has no unique constraint (only
    ``(workspace_id, slug)``) and the create/clone guards compare
    case-SENSITIVELY, so 'Atlas' and 'atlas' can coexist and collide under this
    case-insensitive match. Fetch the active roster and match the name in Python
    (same active-only source + case-insensitive comparison the classifier uses),
    dedup by agent id. Returns ``(agent, error)``:

    * exactly one active match   -> ``(agent, None)``
    * no active match            -> ``(None, None)`` — caller decides unassigned vs 'not found'
    * two-or-more active matches  -> ``(None, "<ambiguity message>")`` — caller refuses

    So a same-named pair never yields a row-order-dependent dispatch, and an
    active-vs-inactive pair resolves to the ACTIVE one (the inactive is never in
    the active roster the match runs over).
    """
    from core.models import Agent

    active = (
        db.query(Agent)
        .filter(Agent.workspace_id == workspace_id, Agent.status == "active")
        .all()
    )
    target = agent_name.strip().lower()
    matches = {
        a.id: a for a in active if (getattr(a, "name", "") or "").lower() == target
    }
    if len(matches) == 1:
        return next(iter(matches.values())), None
    if len(matches) >= 2:
        return None, (
            f"Multiple active agents named '{agent_name}' — I can't tell which one "
            "you mean. Rename or deactivate the duplicate, then try again."
        )
    return None, None


async def create_board_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a board task (called by agents via platform_create_task)."""
    from core.models.core import BoardTask

    title = params.get("title")
    description = params.get("description")
    if not title or not description:
        return {"success": False, "error": "title and description are required"}

    # Resolve assigned agent by name — ACTIVE only + ambiguity-aware (P224-RVW-4).
    # A same-named pair refuses rather than silently dispatching to a row-order
    # pick; an active-vs-inactive pair resolves to the active one. No match leaves
    # the ticket unassigned (inbox), exactly as before.
    assigned_agent_id = None
    agent_name = params.get("assigned_agent_name")
    if agent_name:
        agent, ambiguity_error = _resolve_active_agent_by_name(db, workspace_id, agent_name)
        if ambiguity_error:
            return {"success": False, "error": ambiguity_error}
        if agent:
            assigned_agent_id = agent.id

    # Build planning_data if approval_action or other planning fields provided
    planning_data = params.get("planning_data")
    if not planning_data and params.get("approval_action"):
        planning_data = {"approval_action": params["approval_action"]}

    # Determine initial status — tasks with approval_action go to review
    initial_status = params.get("status", "assigned" if assigned_agent_id else "inbox")
    if planning_data and planning_data.get("approval_action"):
        initial_status = "review"

    task = BoardTask(
        workspace_id=workspace_id,
        title=title,
        description=description,
        priority=params.get("priority", "medium"),
        assigned_agent_id=assigned_agent_id,
        status=initial_status,
        created_by_type="agent",
        created_by_id=str(params.get("_agent_id", "")),
        parent_task_id=params.get("parent_task_id"),
        tags=params.get("tags", []),
        planning_data=planning_data,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Auto-approve: execute the approval action immediately, skip human review
    auto_approve = params.get("auto_approve", False)
    if auto_approve and planning_data and planning_data.get("approval_action"):
        approval_action = planning_data["approval_action"]
        action_type = approval_action.get("type")
        try:
            if action_type == "publish_blog":
                from core.services.blog_service import BlogService
                from uuid import UUID as _UUID
                svc = BlogService(db, workspace_id)
                svc.publish_post(_UUID(approval_action["post_id"]))
                logger.info("[BoardTasks] Auto-approved publish_blog for task %s", task.id)
            task.status = "done"
            db.commit()
            # PRD-227 US-001: push the create (+immediate done) to Command Centres.
            _notify_board_safe(db, workspace_id, task.id, task.status, "task_created")
            return {
                "success": True,
                "task_id": task.id,
                "status": "done",
                "title": task.title,
                "auto_approved": True,
                "action_executed": action_type,
            }
        except Exception as approve_err:
            logger.error("[BoardTasks] Auto-approve failed: %s", approve_err, exc_info=True)
            # Fall through to normal review flow

    # Send notification if task lands in review (approval gate)
    if initial_status == "review":
        try:
            from core.services.notification_service import send_workspace_notification
            action_type = (planning_data or {}).get("approval_action", {}).get("type", "")
            action_label = "publish blog post" if action_type == "publish_blog" else "review task"
            msg = f"[Approval Required] {title}\n{description}\nAction: {action_label}\nOpen the Board to approve or reject."
            await send_workspace_notification(str(workspace_id), msg)
        except Exception as notify_err:
            logger.debug("[BoardTasks] Notification skipped: %s", notify_err)

    # PRD-227 US-001: push the new card to Command Centres, same as api/board_tasks.py:389.
    _notify_board_safe(db, workspace_id, task.id, task.status, "task_created")

    # PRD-224 US-001: a chat-created assigned ticket wakes the dispatch loop on the
    # LISTEN channel (mirrors api/board_tasks.py:397-398) so it is claimed at wake
    # latency, not on the next fallback poll.
    if _is_dispatch_claimable(task):
        _notify_dispatch_safe(db, workspace_id, task.id)

    result: Dict[str, Any] = {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "title": task.title,
    }

    # PRD-224 US-005: an ASSIGN-lane assigned ticket is auto-supervised — attach a
    # run_and_report board_task watch here (in the create transaction path) so the
    # LLM cannot forget it. Gated on the server-injected, unspoofable _assign_lane
    # flag + an assigned agent; the AUTO_TICKET_WATCH dial is the actual switch.
    # A non-ASSIGN creation (heartbeat, recipe mirror, plain agent task) carries no
    # _assign_lane, so it attaches nothing.
    if params.get("_assign_lane") and assigned_agent_id:
        from config import config as _config

        if not _config.AUTO_TICKET_WATCH:
            result["supervised"] = False
            result["supervision"] = "not supervised (AUTO_TICKET_WATCH is off)"
        else:
            from modules.tools.discovery.handlers_watches import (
                _origin_chat_id,
                auto_create_ticket_watch,
            )

            watch = auto_create_ticket_watch(
                db,
                workspace_id,
                task_id=task.id,
                title=f"Ticket: {title}",
                success_criteria=description,
                created_by=(str(params["_created_by"]) if params.get("_created_by") else None),
                owner_agent_id=assigned_agent_id,
                origin_chat_id=_origin_chat_id(params),
            )
            if watch is not None:
                result["supervised"] = True
                result["watch_id"] = str(watch.id)
                result["supervision"] = "supervised — I'll report back here when it's done"
            else:
                # P224-RVW-6: the dial is ON but the watch did NOT attach —
                # auto_create_ticket_watch is fail-soft and returns None on any
                # internal error (a broken watcher must never break the ticket).
                # Confirm honestly instead of silently omitting the signal: the
                # ticket runs, but unwatched. The RVW-3 directive echoes this
                # 'supervision' field verbatim, so the user is never told a false
                # "I'll report back here". (A just-created task id can have no
                # pre-existing live watch, so None here is an attach failure — not
                # the idempotent already-watched case.)
                result["supervised"] = False
                result["supervision"] = "supervision unavailable — the ticket will run unwatched"

    return result


async def list_board_tasks(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List board tasks with optional filters."""
    from core.models.core import BoardTask

    query = db.query(BoardTask).filter(
        BoardTask.workspace_id == workspace_id,
    )

    status = params.get("status")
    if status:
        query = query.filter(BoardTask.status == status)

    priority = params.get("priority")
    if priority:
        query = query.filter(BoardTask.priority == priority)

    # Tag filter — return tasks whose tags include ALL requested tags. BoardTask.tags
    # is JSONB, so .contains([...]) compiles to `tags @> [...]` (array containment).
    # HARNESS self-management relies on this to find its own '[HARNESS]' tasks
    # (tagged 'harness' + 'rx:{id}'); without it the lookup returns unrelated tasks.
    tags = params.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    if isinstance(tags, list) and tags:
        query = query.filter(BoardTask.tags.contains(tags))

    agent_name = params.get("assigned_agent_name")
    if agent_name:
        from core.models import Agent
        from sqlalchemy import func as sa_func
        agent = db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            sa_func.lower(Agent.name) == agent_name.lower(),
        ).first()
        if agent:
            query = query.filter(BoardTask.assigned_agent_id == agent.id)
        else:
            return {"success": True, "tasks": [], "total": 0, "note": f"No agent named '{agent_name}' found"}

    limit = min(int(params.get("limit", 20)), MAX_LIST_TASKS_LIMIT)
    # How many match in all, so the model knows whether it saw everything
    # (the page is the sample, total_matching is the truth).
    try:
        total_matching = int(query.count())
    except Exception:  # noqa: BLE001 — a count failure never fails the listing
        total_matching = None
    tasks = query.order_by(BoardTask.created_at.desc()).limit(limit).all()

    # Enrich with agent names
    agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
    agents_map = {}
    if agent_ids:
        from core.models import Agent
        for a in db.query(Agent).filter(Agent.id.in_(agent_ids)).all():
            agents_map[a.id] = a.name

    result = []
    for t in tasks:
        result.append({
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "status": t.status,
            "priority": t.priority,
            "tags": t.tags or [],
            "assigned_agent": agents_map.get(t.assigned_agent_id, "unassigned"),
            "created_at": str(t.created_at) if t.created_at else None,
            "started_at": str(t.started_at) if t.started_at else None,
            "completed_at": str(t.completed_at) if t.completed_at else None,
            "error_message": t.error_message,
        })

    return {
        "success": True,
        "tasks": result,
        "total": len(result),
        "total_matching": total_matching if total_matching is not None else len(result),
        "limit": limit,
    }


async def get_board_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get full details of a single board task."""
    from core.models.core import BoardTask

    task_id = params.get("task_id")
    if not task_id:
        return {"success": False, "error": "task_id is required"}

    task = db.query(BoardTask).filter(
        BoardTask.id == int(task_id),
        BoardTask.workspace_id == workspace_id,
    ).first()

    if not task:
        return {"success": False, "error": f"Task {task_id} not found"}

    # Resolve agent name
    agent_name = None
    if task.assigned_agent_id:
        from core.models import Agent
        agent = db.query(Agent).get(task.assigned_agent_id)
        agent_name = agent.name if agent else None

    return {
        "success": True,
        "task": {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "raw_prompt": task.raw_prompt,
            "status": task.status,
            "priority": task.priority,
            "review_mode": task.review_mode,
            "assigned_agent": agent_name or "unassigned",
            "tags": task.tags or [],
            "result": str(task.result)[:2000] if task.result else None,
            "error_message": task.error_message,
            "created_at": str(task.created_at) if task.created_at else None,
            "started_at": str(task.started_at) if task.started_at else None,
            "completed_at": str(task.completed_at) if task.completed_at else None,
        },
    }


async def assign_board_task(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Assign a board task to an agent by name."""
    from core.models.core import BoardTask

    task_id = params.get("task_id")
    agent_name = params.get("agent_name")
    if not task_id or not agent_name:
        return {"success": False, "error": "task_id and agent_name are required"}

    task = db.query(BoardTask).filter(
        BoardTask.id == int(task_id),
        BoardTask.workspace_id == workspace_id,
    ).first()
    if not task:
        return {"success": False, "error": f"Task {task_id} not found"}

    # ACTIVE only + ambiguity-aware (P224-RVW-4): a same-named pair refuses rather
    # than silently .first()-ing a row-order pick; no active match is 'not found'.
    agent, ambiguity_error = _resolve_active_agent_by_name(db, workspace_id, agent_name)
    if ambiguity_error:
        return {"success": False, "error": ambiguity_error}
    if not agent:
        return {"success": False, "error": f"Agent '{agent_name}' not found"}

    task.assigned_agent_id = agent.id
    if task.status == "inbox":
        task.status = "assigned"
    db.commit()

    # PRD-227 US-001: push the assignment to Command Centres, mirroring the human
    # assign path (api/board_tasks.py update_task notify_board_event, event="task_updated").
    _notify_board_safe(db, workspace_id, task.id, task.status, "task_updated")

    # PRD-224 US-001: assignment wakes the dispatch loop (mirrors
    # api/board_tasks.py:624-632); the loop claims 'assigned' tasks only, so
    # re-assigning a running ticket is a no-op there.
    if _is_dispatch_claimable(task):
        _notify_dispatch_safe(db, workspace_id, task.id)

    return {
        "success": True,
        "task_id": task.id,
        "assigned_agent": agent.name,
        "status": task.status,
    }


async def _update_many_board_task_statuses(
    db: Session, workspace_id: UUID, params: Dict[str, Any], task_ids: Any
) -> Dict[str, Any]:
    """Bulk form of update_board_task_status: every id goes through the SAME
    single-task path (validation, blocked_reason rule, atomic in_progress claim,
    board NOTIFY), so nothing is bypassed — this only removes the one-call-per-task
    round trip that the per-turn tool cap turned into "closed the ones I could see".
    Reports success only when EVERY id updated; the failed list names the rest."""
    if not isinstance(task_ids, (list, tuple)):
        return {"success": False, "error": "task_ids must be a list of task IDs"}
    if len(task_ids) > MAX_BULK_TASK_IDS:
        return {
            "success": False,
            "error": (
                f"task_ids carries {len(task_ids)} ids; the maximum is {MAX_BULK_TASK_IDS} "
                "per call — split the request and report progress to the user"
            ),
        }
    new_status = params.get("status")
    if not new_status:
        return {"success": False, "error": "task_id (or task_ids) and status are required"}

    single = {k: v for k, v in params.items() if k != "task_ids"}
    updated = []
    failed = []
    for raw_id in task_ids:
        try:
            tid = int(raw_id)
        except (TypeError, ValueError):
            failed.append({"task_id": raw_id, "error": "not an integer task id"})
            continue
        result = await update_board_task_status(db, workspace_id, {**single, "task_id": tid})
        if result.get("success"):
            updated.append(tid)
        else:
            failed.append({"task_id": tid, "error": result.get("error", "unknown error")})
    return {
        "success": not failed,
        "partial": bool(updated) and bool(failed),
        "status": new_status,
        "requested": len(task_ids),
        "updated_count": len(updated),
        "updated": updated,
        "failed": failed,
    }


async def update_board_task_status(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Update a board task's status. Moving to in_progress triggers execution.
    With ``task_ids`` (a list) every id is updated to the same status — see
    ``_update_many_board_task_statuses``."""
    from core.models.core import BoardTask

    task_ids = params.get("task_ids")
    if task_ids is not None and task_ids != []:
        return await _update_many_board_task_statuses(db, workspace_id, params, task_ids)

    task_id = params.get("task_id")
    new_status = params.get("status")
    if not task_id or not new_status:
        return {"success": False, "error": "task_id (or task_ids) and status are required"}

    # PRD-227 US-001: agent-side vocabulary reaches parity with the HTTP path by
    # reusing its VALID_STATUSES set — so 'blocked'/'failed' are accepted and any
    # future status the HTTP path adds is accepted identically, never drifting.
    from api.board_tasks import VALID_STATUSES
    if new_status not in VALID_STATUSES:
        return {"success": False, "error": f"Invalid status: {new_status}. Must be one of {sorted(VALID_STATUSES)}"}

    # 'blocked' requires a reason for the agent path (the board card renders it);
    # validated before any DB read so a bad call is a pure, cheap rejection.
    blocked_reason = params.get("blocked_reason")
    if new_status == "blocked" and not blocked_reason:
        return {"success": False, "error": "blocked_reason is required when setting status to 'blocked'"}

    task = db.query(BoardTask).filter(
        BoardTask.id == int(task_id),
        BoardTask.workspace_id == workspace_id,
    ).first()
    if not task:
        return {"success": False, "error": f"Task {task_id} not found"}

    old_status = task.status

    # P224-RVW-5: the assigned->in_progress transition that LAUNCHES the agent must
    # be ATOMIC against the board dispatcher's concurrent claim. RVW-2's
    # old_status != 'in_progress' guard read from THIS unlocked pre-commit SELECT,
    # so if board_dispatcher.claim_tasks committed its FOR UPDATE SKIP LOCKED claim
    # (flipping the row to in_progress and launching once) AFTER this SELECT but
    # BEFORE our write, old_status was still 'assigned', the guard passed, and we
    # launched a SECOND time — the agent's real side effects (emails, external
    # calls) ran twice. Mirror claim_tasks' exactly-once idiom: a conditional UPDATE
    # that wins only when the row is NOT already in_progress, and launch inline ONLY
    # if THIS statement won the transition (it RETURNs a row). So exactly one of
    # {inline, dispatcher} ever reaches execute_with_prompt for a given claim.
    # Workspace ownership was already verified by the SELECT above; the PK uniquely
    # identifies the row, so the atomic claim filters on id + the status precondition.
    if new_status == "in_progress" and task.assigned_agent_id:
        from sqlalchemy import text

        # Capture launch inputs before the write — commit expires ORM attributes.
        agent_id = task.assigned_agent_id
        prompt = task.raw_prompt or task.description or task.title
        review_mode = task.review_mode or "auto"
        now = datetime.now(timezone.utc)

        won = db.execute(
            text(
                "UPDATE board_tasks "
                "SET status = 'in_progress', "
                "    started_at = COALESCE(started_at, :now), "
                "    blocked_at = NULL, blocked_reason = NULL, "
                "    updated_at = :now "
                "WHERE id = :id AND status <> 'in_progress' "
                "RETURNING id"
            ),
            {"id": int(task_id), "now": now},
        ).fetchone()
        db.commit()

        launched = False
        if won is not None:
            # We won the assigned->in_progress transition — the dispatcher did not
            # beat us to this claim. Push the board frame and launch exactly once.
            _notify_board_safe(db, workspace_id, int(task_id), "in_progress", "status_changed")
            from api.board_tasks import _launch_task_execution

            _launch_task_execution(
                task_id=int(task_id),
                agent_id=agent_id,
                workspace_id=str(workspace_id),
                prompt=prompt,
                review_mode=review_mode,
            )
            launched = True
        # If we lost, the dispatcher already claimed + launched this row — no second
        # launch and no redundant board frame (its task_claimed NOTIFY already fired).
        return {
            "success": True,
            "task_id": int(task_id),
            "status": "in_progress",
            "triggered_execution": launched,
        }

    # Every other transition is a plain ORM write with no launch and no dispatcher
    # race (in_progress WITHOUT an assigned agent cannot run, so it falls here too).
    task.status = new_status
    # PRD-227 P227-RVW-4: mirror the HTTP update_task_status in_progress reset
    # (api/board_tasks.py:890-895) — clear the terminal fields so a redone task
    # (done → in_progress → done) does not carry a stale completed_at/error_message/
    # result. Without this, a task that previously failed then succeeds still renders
    # as the red 'failed' strip (board-card.tsx isFailed = error_message != null &&
    # status == 'done') — a board-state lie this PRD exists to kill. started_at is
    # set unconditionally, matching the HTTP path (restart the clock on a redo).
    if new_status == "in_progress":
        task.started_at = datetime.now(timezone.utc)
        task.completed_at = None
        task.error_message = None
        task.result = None
    if new_status in ("done", "review") and not task.completed_at:
        task.completed_at = datetime.now(timezone.utc)
    # Mirror the HTTP path's blocked transitions (api/board_tasks.py:548-553, 898-902).
    if new_status == "blocked" and task.blocked_at is None:
        task.blocked_at = datetime.now(timezone.utc)
        task.blocked_reason = blocked_reason
    if new_status != "blocked" and old_status == "blocked":
        task.blocked_at = None
        task.blocked_reason = None

    db.commit()

    # PRD-227 US-001: push the status change to Command Centres, same payload
    # shape + event name as the human PATCH (api/board_tasks.py:912, "status_changed").
    _notify_board_safe(db, workspace_id, task.id, task.status, "status_changed")

    # PRD-224 US-001: a move back to 'assigned' (re-queue) wakes the dispatch loop
    # so the ticket is claimed on the LISTEN wake, not the fallback poll — the same
    # claimable guard the HTTP layer notifies on.
    if _is_dispatch_claimable(task):
        _notify_dispatch_safe(db, workspace_id, task.id)

    return {
        "success": True,
        "task_id": task.id,
        "status": task.status,
        "triggered_execution": False,
    }
