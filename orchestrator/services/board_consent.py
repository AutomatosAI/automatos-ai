"""PRD-234 — a human's explicit board action IS the approval the policy asks for.

Under ``always_ask`` every dispatched board ticket waits for a grant. That is the
right posture for autonomous dispatch (schedules, heartbeats, tickets Auto files).
It is the wrong posture when the operator has just dragged the ticket to
In Progress or pressed Run Now: the owner's words (2026-09-03) — "if I move it to
In Progress that means I approve". So those two actions record the grant
themselves, with the same primitive the Governance button uses, and the gate
then finds an active grant and proceeds. Nothing else changes: the grant is
durable, audited on the row (``granted_by``), and expires like any other.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

WHY_MOVED_TO_IN_PROGRESS = "the operator moved the ticket to In Progress"
WHY_RUN_NOW = "the operator pressed Run Now"


def actor_ref(ctx: Any) -> str:
    """``user:<id>`` — the same shape the approvals API records."""
    uid = getattr(ctx, "user_id", None) or getattr(ctx, "internal_user_id", None)
    return f"user:{uid}" if uid is not None else "user:unknown"


def record_operator_consent(
    db: Any, *, workspace_id: Any, task_id: int, agent_id: Optional[int], actor: str, why: str,
) -> str:
    """Make sure an ACTIVE board-task grant exists for ``task_id``.

    Returns ``"active"`` (one already authorised the task), ``"granted"`` (a
    pending grant was approved), ``"created"`` (a granted grant was recorded), or
    ``"error"`` (nothing recorded — the gate will ask as before; never raises).
    """
    try:
        from core.models.approval_grants import SUBJECT_BOARD_TASK
        from core.services.approval_grants import (
            create_grant, find_active_grant, find_pending_grant, grant_grant,
        )
        subject = str(task_id)
        if find_active_grant(db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=subject) is not None:
            return "active"
        pending = find_pending_grant(db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=subject)
        if pending is not None:
            grant_grant(pending, granted_by=actor)
            db.commit()
            return "granted"
        grant = create_grant(
            db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=subject,
            agent_id=agent_id, reason=why,
        )
        grant_grant(grant, granted_by=actor)
        db.commit()
        return "created"
    except Exception:  # noqa: BLE001 — consent is a convenience; the gate stays the safety net
        logger.warning("[BoardConsent] could not record consent for task %s", task_id, exc_info=True)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return "error"
