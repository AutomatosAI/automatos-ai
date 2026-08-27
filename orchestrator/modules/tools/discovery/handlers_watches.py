"""Watch handlers for PlatformActionExecutor -- PRD-204 S9.

Workspace-scoped CRUD over the watch registry + the fail-soft auto-create
helper the mission/playbook launch handlers call (Section 8 Q1:
``watch_auto_create`` default ON, ``run_and_report`` policy,
success_criteria seeded from the user's request text).

CONTRACT: every field the ActionDefinition marks optional is defaulted
here; only ``target_type``/``target_id`` (create) and ``watch_id``
(get/cancel) are required -- see actions_watches.py.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_VALID_TARGET_TYPES = ("mission", "playbook_execution", "scheduled_playbook", "board_task")
_MAX_LIST_LIMIT = 50
_RECENT_EVENT_LIMIT = 10


def _actor(params: Dict[str, Any]) -> Optional[str]:
    """The human behind this action (executor-injected), else None."""
    created_by = params.get("_created_by")
    return str(created_by) if created_by else None


def _origin_chat_id(params: Dict[str, Any]) -> Optional[UUID]:
    """PRD-205 S4: the server-injected originating conversation (never an
    LLM-supplied arg -- the executor overwrites it from caller_context)."""
    raw = params.get("_origin_chat_id")
    if not raw:
        return None
    try:
        return UUID(str(raw))
    except (ValueError, AttributeError, TypeError):
        return None


def _owner_agent_id(params: Dict[str, Any]) -> Optional[int]:
    raw = params.get("_agent_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def watch_to_dict(watch, *, include_lineage: bool = False) -> Dict[str, Any]:
    """Canonical watch serialization -- shared by the S9 tool handlers and
    the S11 watchlist API (api/watches.py) so both surfaces speak one shape."""
    from services.watch_notifications import format_score_display

    data = {
        "id": str(watch.id),
        "title": watch.title,
        "watch_type": watch.watch_type,
        "target_type": watch.target_type,
        "target_id": watch.target_id,
        "status": watch.status,
        "policy": watch.policy,
        "success_criteria": watch.success_criteria,
        "quality_threshold": watch.quality_threshold,
        "final_score": watch.final_score,
        "final_score_display": format_score_display(watch.final_score),
        "final_verdict": watch.final_verdict,
        "actions_taken": watch.actions_taken,
        "action_budget": watch.action_budget,
        "last_checked_at": str(watch.last_checked_at) if watch.last_checked_at else None,
        "next_check_at": str(watch.next_check_at) if watch.next_check_at else None,
        "deadline_at": str(watch.deadline_at) if watch.deadline_at else None,
        "created_at": str(watch.created_at) if watch.created_at else None,
        "closed_at": str(watch.closed_at) if watch.closed_at else None,
    }
    if include_lineage:
        data["lineage"] = watch.lineage or []
    return data


def _resolve_target(
    db: Session, workspace_id: UUID, target_type: str, target_id: str
) -> Optional[Dict[str, str]]:
    """Workspace-scoped target lookup -> {title, criteria} or None.

    Boundary validation: a watch on a target that does not exist (or lives
    in another workspace) is refused at creation, not parked later.
    """
    try:
        if target_type == "mission":
            from core.models.orchestration import OrchestrationRun

            run = (
                db.query(OrchestrationRun)
                .filter(
                    OrchestrationRun.id == str(target_id),
                    OrchestrationRun.workspace_id == workspace_id,
                )
                .first()
            )
            if run is None:
                return None
            goal = (run.goal or "").strip()
            return {
                "title": f"Watch: {goal[:80]}" if goal else f"Watch: mission {target_id}",
                "criteria": goal or "The mission completes successfully.",
            }

        if target_type == "playbook_execution":
            from core.models.core import RecipeExecution, WorkflowTemplate

            execution = (
                db.query(RecipeExecution)
                .filter(
                    RecipeExecution.execution_id == str(target_id),
                    RecipeExecution.workspace_id == workspace_id,
                )
                .first()
            )
            if execution is None:
                return None
            recipe = (
                db.query(WorkflowTemplate)
                .filter(WorkflowTemplate.id == execution.recipe_id)
                .first()
            )
            name = getattr(recipe, "name", None) or f"playbook {execution.recipe_id}"
            return {
                "title": f"Watch: {name[:80]}",
                "criteria": f"Playbook '{name}' completes and delivers its expected output.",
            }

        if target_type == "scheduled_playbook":
            from core.models.core import WorkflowTemplate

            try:
                recipe_id = int(target_id)
            except (TypeError, ValueError):
                return None
            recipe = (
                db.query(WorkflowTemplate)
                .filter(
                    WorkflowTemplate.id == recipe_id,
                    WorkflowTemplate.workspace_id == workspace_id,
                )
                .first()
            )
            if recipe is None:
                return None
            return {
                "title": f"Watch: {recipe.name[:70]} (schedule)",
                "criteria": f"Scheduled playbook '{recipe.name}' keeps running on time.",
            }

        if target_type == "board_task":
            # PRD-224 US-002: supervise an assigned ticket. target_id is the
            # integer BoardTask id; a non-integer or cross-workspace/unknown id
            # is refused here so the watch is never created on a phantom target.
            from core.models.core import BoardTask

            try:
                task_id = int(target_id)
            except (TypeError, ValueError):
                return None
            task = (
                db.query(BoardTask)
                .filter(
                    BoardTask.id == task_id,
                    BoardTask.workspace_id == workspace_id,
                )
                .first()
            )
            if task is None:
                return None
            label = (task.title or "").strip()
            criteria = (task.description or "").strip() or (
                f"Board task '{label}' is completed to standard."
                if label
                else f"Board task {target_id} is completed to standard."
            )
            return {
                "title": f"Watch: {label[:80]}" if label else f"Watch: task {target_id}",
                "criteria": criteria,
            }
    except Exception:
        logger.warning(
            "[Watches] target resolve failed for %s:%s", target_type, target_id,
            exc_info=True,
        )
    return None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def create_watch(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """platform_create_watch."""
    from core.models.watch_enums import WatchPolicy
    from core.models.watches import (
        DEFAULT_ACTION_BUDGET,
        DEFAULT_QUALITY_THRESHOLD,
    )
    from services.watch_service import WatchAlreadyExistsError, WatchService

    target_type = params.get("target_type")
    target_id = params.get("target_id")
    if target_type not in _VALID_TARGET_TYPES:
        return {
            "success": False,
            "error": f"target_type must be one of {list(_VALID_TARGET_TYPES)}",
        }
    if not target_id:
        return {"success": False, "error": "target_id is required"}
    target_id = str(target_id)

    resolved = _resolve_target(db, workspace_id, target_type, target_id)
    if resolved is None:
        return {
            "success": False,
            "error": f"No {target_type} '{target_id}' found in this workspace",
        }

    policy = params.get("policy") or WatchPolicy.RUN_AND_REPORT.value
    if policy not in {p.value for p in WatchPolicy}:
        return {
            "success": False,
            "error": f"policy must be one of {sorted(p.value for p in WatchPolicy)}",
        }

    threshold = params.get("quality_threshold")
    if threshold is None:
        threshold = DEFAULT_QUALITY_THRESHOLD
    try:
        threshold = max(0.0, min(1.0, float(threshold)))
    except (TypeError, ValueError):
        return {"success": False, "error": "quality_threshold must be a number in [0, 1]"}

    deadline_at = None
    deadline_hours = params.get("deadline_hours")
    if deadline_hours is not None:
        try:
            hours = float(deadline_hours)
        except (TypeError, ValueError):
            return {"success": False, "error": "deadline_hours must be a number"}
        if hours <= 0:
            return {"success": False, "error": "deadline_hours must be positive"}
        from datetime import datetime, timezone

        deadline_at = datetime.now(timezone.utc) + timedelta(hours=hours)

    action_budget = params.get("action_budget")
    if action_budget is None:
        action_budget = DEFAULT_ACTION_BUDGET
    try:
        action_budget = max(0, int(action_budget))
    except (TypeError, ValueError):
        return {"success": False, "error": "action_budget must be an integer"}

    try:
        watch = WatchService.create_watch(
            db,
            workspace_id=workspace_id,
            watch_type=target_type,
            target_type=target_type,
            target_id=target_id,
            title=(params.get("title") or resolved["title"])[:500],
            created_by=_actor(params),
            owner_agent_id=_owner_agent_id(params),
            origin_chat_id=_origin_chat_id(params),
            success_criteria=params.get("success_criteria") or resolved["criteria"],
            quality_threshold=threshold,
            deadline_at=deadline_at,
            policy=policy,
            action_budget=action_budget,
        )
    except WatchAlreadyExistsError:
        existing = WatchService.find_live_watch(
            db, workspace_id=workspace_id, target_type=target_type, target_id=target_id
        )
        return {
            "success": True,
            "existing": True,
            "watch": watch_to_dict(existing) if existing else None,
            "message": f"That {target_type} is already being watched.",
        }
    except Exception as e:
        logger.error("[Watches] create_watch failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to create watch: {str(e)[:300]}"}

    return {
        "success": True,
        "existing": False,
        "watch": watch_to_dict(watch),
        "message": (
            f"Watching {target_type} {target_id} to a verdict "
            f"(policy {policy}, bar {threshold:.2f})."
        ),
    }


async def list_watches(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """platform_list_watches."""
    from services.watch_service import WatchService

    try:
        limit = min(int(params.get("limit", 20)), _MAX_LIST_LIMIT)
    except (TypeError, ValueError):
        limit = 20

    try:
        watches = WatchService.list_watches(
            db,
            workspace_id,
            status=params.get("status"),
            watch_type=params.get("watch_type"),
            include_closed=bool(params.get("include_closed", False)),
            limit=limit,
        )
    except Exception as e:
        logger.error("[Watches] list_watches failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to list watches: {str(e)[:300]}"}

    return {
        "success": True,
        "watches": [watch_to_dict(w) for w in watches],
        "total": len(watches),
    }


async def get_watch(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """platform_get_watch."""
    from core.models.watches import WatchEvent
    from services.watch_service import WatchService

    watch_id = params.get("watch_id")
    if not watch_id:
        return {"success": False, "error": "watch_id is required"}

    try:
        watch = WatchService.get_watch(db, workspace_id, str(watch_id))
    except Exception:
        watch = None
    if watch is None:
        return {"success": False, "error": f"Watch {watch_id} not found"}

    events = (
        db.query(WatchEvent)
        .filter(WatchEvent.watch_id == watch.id)
        .order_by(WatchEvent.created_at.desc())
        .limit(_RECENT_EVENT_LIMIT)
        .all()
    )
    return {
        "success": True,
        "watch": watch_to_dict(watch, include_lineage=True),
        "recent_events": [
            {
                "event_type": e.event_type,
                "summary": e.summary,
                "score": e.score,
                "action_taken": e.action_taken,
                "requires_attention": e.requires_attention,
                "created_at": str(e.created_at) if e.created_at else None,
            }
            for e in events
        ],
    }


async def cancel_watch(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """platform_cancel_watch."""
    from services.orchestration_state import InvalidTransitionError
    from services.watch_service import WatchService

    watch_id = params.get("watch_id")
    if not watch_id:
        return {"success": False, "error": "watch_id is required"}

    try:
        watch = WatchService.cancel_watch(
            db, workspace_id, str(watch_id), reason=params.get("reason")
        )
    except ValueError as e:
        return {"success": False, "error": str(e)[:300]}
    except InvalidTransitionError:
        return {
            "success": False,
            "error": "That watch is already closed and cannot be cancelled.",
        }
    except Exception as e:
        logger.error("[Watches] cancel_watch failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to cancel watch: {str(e)[:300]}"}

    return {
        "success": True,
        "watch": watch_to_dict(watch),
        "message": f"Watch {watch.id} cancelled.",
    }


# ---------------------------------------------------------------------------
# Auto-create (Section 8 Q1) -- called by the launch handlers, fail-soft
# ---------------------------------------------------------------------------


def auto_create_watch(
    db: Session,
    workspace_id: UUID,
    *,
    target_type: str,
    target_id: str,
    title: str,
    success_criteria: str,
    created_by: Optional[str] = None,
    owner_agent_id: Optional[int] = None,
    origin_chat_id: Optional[UUID] = None,
):
    """Create the default run_and_report watch on Auto-launched work.

    Gated on the ``watch_auto_create`` workspace setting (default ON).
    Idempotent (one live watch per target) and fail-soft: NEVER raises into
    the launching handler -- a broken watcher must not break a launch.
    Returns the Watch or None.
    """
    try:
        from services.watch_service import (
            WatchService,
            watch_auto_create_enabled,
        )

        if not watch_auto_create_enabled(db, workspace_id):
            return None
        if WatchService.find_live_watch(
            db,
            workspace_id=workspace_id,
            target_type=target_type,
            target_id=str(target_id),
        ) is not None:
            return None

        watch = WatchService.create_watch(
            db,
            workspace_id=workspace_id,
            watch_type=target_type,
            target_type=target_type,
            target_id=str(target_id),
            title=title[:500],
            created_by=created_by,
            owner_agent_id=owner_agent_id,
            success_criteria=success_criteria,
            origin_chat_id=origin_chat_id,
        )
        logger.info(
            "[Watches] auto-created watch %s on %s:%s",
            watch.id,
            target_type,
            target_id,
        )
        return watch
    except Exception:
        logger.warning(
            "[Watches] auto-create failed for %s:%s -- launch unaffected",
            target_type,
            target_id,
            exc_info=True,
        )
        return None
