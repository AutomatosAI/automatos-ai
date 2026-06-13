"""Mission handlers for PlatformActionExecutor (PRD-82A)."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Mirrors the UI suggestion-card's recentMessages.slice(-5) (mission-suggestion-card.tsx).
_CONTEXT_MESSAGE_LIMIT = 5


def _actor(params: Dict[str, Any]) -> str:
    """PRD-163 Q56: the human behind this action — the chatting user's clerk id
    (``_created_by``, injected by the executor) when available, else the agent."""
    return str(params.get("_created_by") or params.get("_agent_id") or "agent")


def _recent_chat_context(db: Session, workspace_id: UUID, chat_id: Any = None,
                         limit: int = _CONTEXT_MESSAGE_LIMIT) -> list:
    """Recent workspace chat turns, oldest-first, as ``[{role, content}]``.

    The executor (Auto-tool) path has no chat_id, so we best-effort read the
    workspace's most recent messages; an explicit chat_id narrows to one thread.
    Parts→text extraction mirrors handlers_search. Best-effort: any failure
    yields no context rather than blocking mission creation.
    """
    from core.models.core import Message

    try:
        q = db.query(Message).filter(Message.workspace_id == workspace_id)
        if chat_id:
            q = q.filter(Message.chat_id == chat_id)
        rows = q.order_by(Message.created_at.desc()).limit(limit).all()
    except Exception as exc:
        logger.warning("[Missions] could not load chat context: %s", exc)
        return []

    context = []
    for m in reversed(rows):
        parts = m.parts if isinstance(m.parts, list) else []
        text_content = " ".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
        ).strip()
        if text_content:
            context.append({"role": m.role, "content": text_content})
    return context


def _plan_task_summary(plan_tasks: list) -> list:
    """The tasks array of the create-mission tool result — the PRD-163 S4
    approval card is built from it verbatim. PRD-164 S2: carries the agent
    match preview (``match_agent``/``match_reason``, mirrored into the plan by
    the coordinator) so the card can show WHO would run each task and WHY.
    """
    summary = []
    for t in plan_tasks[:10]:
        entry = {
            "title": t.get("title", ""),
            "agent_role": t.get("agent_role", ""),
            "sequence": t.get("sequence_number", 0),
        }
        if t.get("match_agent"):
            entry["match_agent"] = t["match_agent"]
        if t.get("match_reason"):
            entry["match_reason"] = t["match_reason"]
        if t.get("match_is_override"):
            entry["match_is_override"] = True
        summary.append(entry)
    return summary


def _create_reply_message(run: Any, task_count: int) -> str:
    """Honest status line — a mission defaults to awaiting_approval, not running."""
    from core.models.orchestration_enums import RunState

    if run.state == RunState.AWAITING_APPROVAL.value:
        return (
            f"Mission {run.id} created with {task_count} task(s) and is awaiting your "
            f"approval. Review the plan and approve it to begin execution."
        )
    if run.state == RunState.RUNNING.value:
        return (
            f"Mission {run.id} created with {task_count} task(s) and is now running "
            f"(auto-approved); the coordinator will execute them automatically."
        )
    return f"Mission {run.id} created with {task_count} task(s) (state: {run.state})."


async def create_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a mission via CoordinatorService.

    Missions default to ``awaiting_approval``; the reply states that honestly and
    tells the caller how to approve. Recent chat is attached as context so the
    planner sees the conversation that motivated the mission.
    """
    goal = params.get("goal")
    if not goal:
        return {"success": False, "error": "goal is required"}

    config = dict(params.get("config") or {})
    created_by = _actor(params)

    # The UI suggestion-card sets context_messages on its API call; the executor
    # path did not. Attach recent conversation here too — never clobber a
    # caller-supplied context.
    if "context_messages" not in config:
        recent = _recent_chat_context(db, workspace_id, chat_id=config.get("chat_id"))
        if recent:
            config["context_messages"] = recent
            config.setdefault("source", "chat")

    try:
        from core.models.orchestration_enums import RunState
        from services.coordinator_service import CoordinatorService

        coordinator = CoordinatorService()
        run = await coordinator.create_mission(
            db=db,
            workspace_id=workspace_id,
            goal=goal,
            created_by=created_by,
            config=config,
        )

        # Summarize the plan for the caller (PRD-164 S2: includes the agent
        # match preview so the approval card can show reasons).
        plan = run.plan or {}
        tasks = plan.get("tasks", [])
        task_summary = _plan_task_summary(tasks)

        return {
            "success": True,
            "mission_id": run.id,
            "state": run.state,
            "awaiting_approval": run.state == RunState.AWAITING_APPROVAL.value,
            "goal": run.goal[:200] if run.goal else "",
            "task_count": len(tasks),
            "tasks": task_summary,
            "message": _create_reply_message(run, len(tasks)),
        }

    except Exception as e:
        logger.error("[Missions] create_mission failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to create mission: {str(e)[:300]}"}


async def list_missions(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List missions in the workspace."""
    from core.models.orchestration import OrchestrationRun

    query = db.query(OrchestrationRun).filter(
        OrchestrationRun.workspace_id == workspace_id,
    )

    state = params.get("state")
    if state:
        query = query.filter(OrchestrationRun.state == state)

    limit = min(int(params.get("limit", 10)), 50)
    runs = query.order_by(OrchestrationRun.created_at.desc()).limit(limit).all()

    result = []
    for r in runs:
        plan = r.plan or {}
        result.append({
            "id": r.id,
            "goal": (r.goal or "")[:150],
            "state": r.state,
            "task_count": len(plan.get("tasks", [])),
            "created_at": str(r.created_at) if r.created_at else None,
            "completed_at": str(r.completed_at) if r.completed_at else None,
            "created_by": r.created_by,
        })

    return {"success": True, "missions": result, "total": len(result)}


async def get_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get full details of a specific mission."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    mission_id = params.get("mission_id")
    if not mission_id:
        return {"success": False, "error": "mission_id is required"}

    # Run ids are UUIDs, not ints — coerce and fail cleanly on a malformed id.
    try:
        run_id = mission_id if isinstance(mission_id, UUID) else UUID(str(mission_id))
    except (ValueError, TypeError):
        return {"success": False, "error": f"Invalid mission_id: {mission_id!r}"}

    run = db.query(OrchestrationRun).filter(
        OrchestrationRun.id == run_id,
        OrchestrationRun.workspace_id == workspace_id,
    ).first()

    if not run:
        return {"success": False, "error": f"Mission {mission_id} not found"}

    # Get tasks
    tasks = db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == run.id,
    ).order_by(OrchestrationTask.sequence_number).all()

    task_details = []
    for t in tasks:
        # PRD-164 S2: surface the persisted match reason (plan preview, then
        # superseded by the dispatch decision) alongside each task. getattr —
        # input_context may be absent (legacy rows / lighter task shapes) or NULL.
        _ic = getattr(t, "input_context", None)
        agent_match = (_ic.get("agent_match") if isinstance(_ic, dict) else None) or {}
        task_details.append({
            "id": t.id,
            "title": t.title,
            "state": t.state,
            "agent_role": t.agent_role,
            "sequence": t.sequence_number,
            "match_agent": agent_match.get("agent_name"),
            "match_reason": agent_match.get("reason"),
            "result_summary": str(t.output)[:500] if t.output else None,
            "error": t.failure_detail,
        })

    return {
        "success": True,
        "mission": {
            "id": run.id,
            "goal": run.goal,
            "state": run.state,
            "config": run.config,
            "plan": run.plan,
            "created_by": run.created_by,
            "created_at": str(run.created_at) if run.created_at else None,
            "completed_at": str(run.completed_at) if run.completed_at else None,
            "tasks": task_details,
        },
    }


# ---------------------------------------------------------------------------
# PRD-163 S1: lifecycle tools — approve / reject / pause / resume / cancel / replan
# Thin wrappers over the existing CoordinatorService lifecycle methods so Auto
# can drive a mission from chat. The transaction is committed by the executor
# (same as create_mission). Notifications resolve to the creating user inside
# the coordinator (run.created_by), satisfying Q56.
# ---------------------------------------------------------------------------

def _resolve_run(db: Session, workspace_id: UUID, params: Dict[str, Any]):
    """Resolve + workspace-scope a run from ``params['mission_id']``.

    Returns ``(run, None)`` or ``(None, error_dict)``.
    """
    from core.models.orchestration import OrchestrationRun

    mission_id = params.get("mission_id")
    if not mission_id:
        return None, {"success": False, "error": "mission_id is required"}
    try:
        run_id = mission_id if isinstance(mission_id, UUID) else UUID(str(mission_id))
    except (ValueError, TypeError):
        return None, {"success": False, "error": f"Invalid mission_id: {mission_id!r}"}

    run = (
        db.query(OrchestrationRun)
        .filter(
            OrchestrationRun.id == run_id,
            OrchestrationRun.workspace_id == workspace_id,
        )
        .first()
    )
    if not run:
        return None, {"success": False, "error": f"Mission {mission_id} not found"}
    return run, None


def _ok(run: Any, verb: str) -> Dict[str, Any]:
    return {
        "success": True,
        "mission_id": str(run.id),
        "state": run.state,
        "message": f"Mission {run.id} {verb} (state: {run.state}).",
    }


async def approve_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Approve a mission plan and start execution (awaiting_approval → running)."""
    run, err = _resolve_run(db, workspace_id, params)
    if err:
        return err
    from services.coordinator_service import CoordinatorService

    actor_id = _actor(params)
    try:
        updated = CoordinatorService().approve_plan(db, run.id, actor_id)
        return _ok(updated, "approved → running")
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[Missions] approve failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to approve mission: {str(e)[:300]}"}


async def reject_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Reject a mission plan (awaiting_approval → failed)."""
    run, err = _resolve_run(db, workspace_id, params)
    if err:
        return err
    reason = params.get("reason") or "Rejected by user"
    from services.coordinator_service import CoordinatorService

    actor_id = _actor(params)
    try:
        updated = CoordinatorService().reject_plan(db, run.id, actor_id, reason=reason)
        return _ok(updated, "rejected")
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[Missions] reject failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to reject mission: {str(e)[:300]}"}


async def pause_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Pause a running mission (running → paused)."""
    run, err = _resolve_run(db, workspace_id, params)
    if err:
        return err
    from services.coordinator_service import CoordinatorService

    actor_id = _actor(params)
    try:
        updated = CoordinatorService().pause_mission(db, run.id, actor_id)
        return _ok(updated, "paused")
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[Missions] pause failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to pause mission: {str(e)[:300]}"}


async def resume_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Resume a paused mission (paused → running)."""
    run, err = _resolve_run(db, workspace_id, params)
    if err:
        return err
    from services.coordinator_service import CoordinatorService

    actor_id = _actor(params)
    try:
        updated = CoordinatorService().resume_mission(db, run.id, actor_id)
        return _ok(updated, "resumed → running")
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[Missions] resume failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to resume mission: {str(e)[:300]}"}


async def cancel_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Cancel a mission (any non-terminal → cancelled)."""
    run, err = _resolve_run(db, workspace_id, params)
    if err:
        return err
    from services.coordinator_service import CoordinatorService

    actor_id = _actor(params)
    try:
        updated = CoordinatorService().cancel_mission(db, run.id, actor_id)
        return _ok(updated, "cancelled")
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[Missions] cancel failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to cancel mission: {str(e)[:300]}"}


async def replan_mission(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Replan a failed mission (failed → replanning → running)."""
    run, err = _resolve_run(db, workspace_id, params)
    if err:
        return err
    from services.coordinator_service import CoordinatorService

    actor_id = _actor(params)
    try:
        updated = await CoordinatorService().replan_mission(
            db, run.id, actor_id, notes=params.get("notes"),
        )
        return _ok(updated, "replanned → running")
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[Missions] replan failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to replan mission: {str(e)[:300]}"}


async def update_mission_plan(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """PRD-163 S4/Q57: apply approval-time task/agent edits to an awaiting-approval
    mission (e.g. reassign a task's agent) so they persist into execution."""
    run, err = _resolve_run(db, workspace_id, params)
    if err:
        return err
    task_edits = params.get("task_edits")
    if not isinstance(task_edits, list) or not task_edits:
        return {"success": False, "error": "task_edits must be a non-empty list"}
    from services.coordinator_service import CoordinatorService

    actor_id = _actor(params)
    try:
        updated = CoordinatorService().update_mission_plan(db, run.id, actor_id, task_edits)
        return _ok(updated, "plan updated")
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("[Missions] update_mission_plan failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to update mission plan: {str(e)[:300]}"}
