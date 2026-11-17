"""The pieces of ``handlers_board_tasks.create_board_task`` (platform_create_task).

Moved out of it unchanged in what they do, so the handler fits the 50-line rule
(``handlers_board_tasks`` is over 800 lines and does not grow): the review gate it keeps,
its planning data, the new row, a card's auto-approval, the note that a card waits in
Review, and the ASSIGN lane's supervision. The handler itself stages an assigned ticket's
consent before the commit that files it, so the dispatcher never claims the ticket first.

The board notices, the dispatcher wake and the consent stay in the handler's module,
where the tests that quiet them reach them; the auto-approval is handed the board notice.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

__all__ = [
    "assign_lane_supervision", "auto_approve_on_create", "new_chat_ticket", "planning_data_of",
    "review_mode_or_refusal", "tell_the_owner_if_it_waits",
]


def review_mode_or_refusal(params: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """The review gate the new card keeps, or the refusal of a gate the board does not have."""
    # F180 (#1110): the review gate is kept as asked, never quietly made 'auto'.
    # 'human' was dropped because this tool took only 'auto' and 'manual'.
    from api.board_tasks import VALID_REVIEW_MODES, board_review_mode

    review_mode = "auto"
    if params.get("review_mode") is not None:
        review_mode = board_review_mode(params["review_mode"])
        if review_mode is None:
            return None, {"success": False, "error": (
                f"Invalid review_mode: {params['review_mode']!r}. Must be one of {sorted(VALID_REVIEW_MODES)}: "
                "'human' waits in Review for a person, 'llm' for a model, 'auto' closes it Done.")}
    return review_mode, None


def planning_data_of(params: Dict[str, Any]) -> Any:
    """The new card's planning data: the one given, or its approval action alone."""
    # Build planning_data if approval_action or other planning fields provided
    planning_data = params.get("planning_data")
    if not planning_data and params.get("approval_action"):
        planning_data = {"approval_action": params["approval_action"]}
    return planning_data


def new_chat_ticket(
    workspace_id: Any, params: Dict[str, Any], *, title: str, description: str, assigned_agent_id: Optional[int],
    initial_status: str, planning_data: Any, review_mode: str, sla_deadline: Any,
) -> Any:
    """The card Auto files, built and not added."""
    from core.models.core import BoardTask

    return BoardTask(
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
        # PRD-234 S3: Auto can set the review gate and a due date when filing.
        review_mode=review_mode,
        sla_deadline=sla_deadline,
    )


def auto_approve_on_create(
    db: Session, workspace_id: Any, task: Any, planning_data: Any, params: Dict[str, Any],
    notify_board: Callable[..., None],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """The answer when the card's approval action ran at once, and whether a widget call's auto_approve was held."""
    # Auto-approve: execute the approval action immediately, skip human review.
    # F155: a public widget visitor's call is never an approval; the task waits.
    from core.security.surface import widget_turn

    auto_approve_held = bool(params.get("auto_approve")) and widget_turn()
    auto_approve = params.get("auto_approve", False) and not auto_approve_held
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
            notify_board(db, workspace_id, task.id, task.status, "task_created")
            db.commit()  # F119: the notice rides this commit
            return {
                "success": True,
                "task_id": task.id,
                "status": "done",
                "title": task.title,
                "auto_approved": True,
                "action_executed": action_type,
            }, auto_approve_held
        except Exception as approve_err:
            logger.error("[BoardTasks] Auto-approve failed: %s", approve_err, exc_info=True)
            # Fall through to normal review flow
    return None, auto_approve_held


async def tell_the_owner_if_it_waits(
    workspace_id: Any, initial_status: str, title: str, description: str, planning_data: Any,
) -> None:
    """The workspace notice that a new card waits in Review for its approval."""
    # Send notification if task lands in review (approval gate)
    if initial_status == "review":
        try:
            from core.services.notification_service import send_workspace_notification
            action_type = (planning_data or {}).get("approval_action", {}).get("type", "")
            action_label = "publish blog post" if action_type == "publish_blog" else "review task"
            msg = f"[Approval Required] {title}\n{description}\nAction: {action_label}\nOpen the Board to approve or reject."
            await send_workspace_notification(str(workspace_id), msg)
        except Exception as notify_err:  # noqa: BLE001 — the notice is a courtesy; the card is filed
            logger.debug("[BoardTasks] Notification skipped: %s", notify_err)


def assign_lane_supervision(
    db: Session, workspace_id: Any, params: Dict[str, Any], task: Any, title: str, description: str,
    assigned_agent_id: Optional[int],
) -> Dict[str, Any]:
    """The answer's supervision fields for an ASSIGN-lane card; none for any other."""
    supervision: Dict[str, Any] = {}
    # PRD-224 US-005: an ASSIGN-lane assigned ticket is auto-supervised — attach a
    # run_and_report board_task watch here (in the create transaction path) so the
    # LLM cannot forget it. Gated on the server-injected, unspoofable _assign_lane
    # flag + an assigned agent; the AUTO_TICKET_WATCH dial is the actual switch.
    # A non-ASSIGN creation (heartbeat, recipe mirror, plain agent task) carries no
    # _assign_lane, so it attaches nothing.
    if params.get("_assign_lane") and assigned_agent_id:
        from config import config as _config

        if not _config.AUTO_TICKET_WATCH:
            supervision["supervised"] = False
            supervision["supervision"] = "not supervised (AUTO_TICKET_WATCH is off)"
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
                supervision["supervised"] = True
                supervision["watch_id"] = str(watch.id)
                supervision["supervision"] = "supervised — I'll report back here when it's done"
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
                supervision["supervised"] = False
                supervision["supervision"] = "supervision unavailable — the ticket will run unwatched"
    return supervision
