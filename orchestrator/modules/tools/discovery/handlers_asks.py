"""PRD-225 — handler for ``platform_ask_human``: park, notify, return.

Contract (US-002): validate the subject in the caller's workspace, stage a
``kind='question'`` grant row, park the subject, fire ``question_pending`` through
the dispatcher (urgent when the blocked cascade is large), and RETURN IMMEDIATELY
with ``{ask_id, parked: True}``. No sleep, no poll — the tool never waits for the
answer. The asker is the server-minted ``_agent_id`` (exec_platform strips any
caller-supplied one), never a tool parameter.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_SUBJECTS = ("board_task", "playbook_run", "tool_call")
_PARKED_STATUSES = ("done", "failed")  # a terminal task is not re-parked


async def ask_human(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Raise a human question that parks its subject and returns at once."""
    from core.models.approval_grants import KIND_QUESTION
    from core.services.approval_grants import DEFAULT_TTL_SECONDS, create_grant
    from services.ask_cascade import count_downstream_blocked, is_urgent_cascade

    subject_type = (params.get("subject_type") or "").strip()
    subject_id = str(params.get("subject_id") or "").strip()
    question = (params.get("question") or "").strip()
    options = params.get("options")
    expires_hours = params.get("expires_hours")

    # Asker identity is the trusted, server-minted runtime agent id — NEVER a
    # tool parameter (exec_platform re-injects _agent_id and strips any spoof).
    asked_by_agent_id = params.get("_agent_id")
    agent_name = params.get("_agent_name")

    # --- validation -------------------------------------------------------
    if subject_type not in _SUBJECTS:
        return {"success": False, "error": f"subject_type must be one of {_SUBJECTS}"}
    if not subject_id:
        return {"success": False, "error": "subject_id is required"}
    if not question:
        return {"success": False, "error": "question is required"}
    if options is not None and not isinstance(options, list):
        return {"success": False, "error": "options must be a list of strings"}

    # The board-task subject is validated against the workspace and parked by a
    # status flip. playbook_run / tool_call subjects follow the grant precedent
    # (issue_tool_grant): the pending row IS the park — the subject id references
    # the parked work, no separate table state to flip.
    board_task = None
    if subject_type == "board_task":
        from core.models.core import BoardTask

        try:
            task_pk = int(subject_id)
        except (TypeError, ValueError):
            return {"success": False, "error": "board_task subject_id must be an integer id"}
        board_task = (
            db.query(BoardTask)
            .filter(BoardTask.id == task_pk, BoardTask.workspace_id == workspace_id)
            .first()
        )
        if board_task is None:
            return {"success": False, "error": "board task not found in this workspace"}

    # --- stage the ask row ------------------------------------------------
    try:
        ttl_seconds = (
            int(expires_hours) * 3600 if expires_hours else DEFAULT_TTL_SECONDS
        )
    except (TypeError, ValueError):
        ttl_seconds = DEFAULT_TTL_SECONDS

    grant = create_grant(
        db, workspace_id,
        subject_type=subject_type,
        subject_id=subject_id,
        kind=KIND_QUESTION,
        question_md=question,
        options=options,
        asked_by_agent_id=(int(asked_by_agent_id) if asked_by_agent_id else None),
        agent_id=(int(asked_by_agent_id) if asked_by_agent_id else None),
        reason="Awaiting human answer",
        ttl_seconds=ttl_seconds,
    )

    # --- park the subject -------------------------------------------------
    if board_task is not None and board_task.status not in _PARKED_STATUSES:
        board_task.status = "blocked"
        board_task.blocked_at = datetime.now(timezone.utc)
        board_task.blocked_reason = f"Awaiting human answer (ask #{grant.id})"

    # Durably park first, then notify — a notification fault leaves a working
    # in-app ask (the tab reads grants, not notifications).
    db.commit()

    downstream = count_downstream_blocked(db, workspace_id, subject_type, subject_id)
    await _dispatch_question_pending(
        db, workspace_id, grant_id=grant.id, question=question,
        agent_id=grant.asked_by_agent_id, agent_name=agent_name,
        urgent=is_urgent_cascade(downstream),
    )
    db.commit()

    return {
        "success": True,
        "ask_id": grant.id,
        "parked": True,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "downstream_blocked": downstream,
        "message": (
            f"Parked {subject_type} {subject_id} and asked the human (ask #{grant.id}). "
            "Move on to other work — the answer will resume it."
        ),
    }


async def _dispatch_question_pending(
    db: Session,
    workspace_id: UUID,
    *,
    grant_id: int,
    question: str,
    agent_id: Optional[int],
    agent_name: Optional[str],
    urgent: bool,
) -> None:
    """Fire ``question_pending`` through the canonical dispatcher. Best-effort:
    a dispatch fault never breaks the (already-committed) park."""
    try:
        from core.services.notification_dispatcher import NotificationDispatcher

        await NotificationDispatcher(db, str(workspace_id)).dispatch(
            event_type="question_pending",
            title=f"Question from {agent_name}" if agent_name else "A question needs your answer",
            message=question,
            link_type="question",
            link_id=str(grant_id),
            agent_id=agent_id,
            agent_name=agent_name,
            status="action_required",
            severity="urgent" if urgent else None,
        )
    except Exception:  # noqa: BLE001 — notification must never break the ask
        logger.warning(
            "[handlers_asks] question_pending dispatch failed for ask %s",
            grant_id, exc_info=True,
        )
