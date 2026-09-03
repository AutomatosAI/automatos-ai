"""PRD-234 — a human's explicit board action IS the approval the policy asks for.

Under ``always_ask`` every dispatched board ticket waits for a grant. That is the
right posture for autonomous dispatch (schedules, heartbeats, tickets Auto files).
It is the wrong posture when the operator has just dragged the ticket to
In Progress or pressed Run Now: the owner's words (2026-09-03) — "if I move it to
In Progress that means I approve". So those two actions record the grant
themselves, with the same primitive the Governance button uses, and the gate
then finds an active grant and proceeds. Nothing else changes: the grant is
durable, audited on the row (``granted_by``), and expires like any other.

Extended the same day ("all tasks just jump straight to blocked"): on the LOCAL
edition there is one operator, so a ticket they file and assign — on the board,
or by asking Auto in chat — is their approval too, and it is recorded at
creation so the dispatcher's claim finds an active grant. SaaS keeps the
maker/checker posture: a filed ticket still waits for a Governance approval.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

WHY_MOVED_TO_IN_PROGRESS = "the operator moved the ticket to In Progress"
WHY_RUN_NOW = "the operator pressed Run Now"
WHY_CREATED_AND_ASSIGNED = "the operator created the ticket and assigned it"
WHY_ASKED_IN_CHAT = "the operator asked Auto for it in chat"

LOCAL_EDITION = "local"
SKIPPED = "skipped"


def actor_ref(ctx: Any) -> str:
    """``user:<id>`` — the same shape the approvals API records.

    ``RequestContext`` carries the principal as ``ctx.user.id`` (the Clerk id, or
    the local user's id); older callers exposed ``user_id`` directly.
    """
    uid = getattr(ctx, "user_id", None) or getattr(ctx, "internal_user_id", None)
    if uid is None:
        user = getattr(ctx, "user", None)
        uid = getattr(user, "id", None) or getattr(user, "email", None)
    return f"user:{uid}" if uid else "user:unknown"


def actor_from_user_id(user_id: Any) -> str:
    """The same ``user:<id>`` shape for a user id threaded through a tool call."""
    return f"user:{user_id}" if user_id else "user:unknown"


def edition() -> str:
    """The running edition (``local`` | ``saas``) from the canonical config object.

    ``config.py`` exports the settings instance as ``config.config`` (the name
    every service imports); the first cut looked for ``settings`` and fell back
    to the bare module, which has no ``AUTH_EDITION`` — so the local edition read
    as saas and ticket 85 was parked behind grant #21 like before.
    """
    try:
        from config import config as app_config

        return str(getattr(app_config, "AUTH_EDITION", "saas") or "saas").lower()
    except Exception:  # noqa: BLE001 — an unreadable edition keeps the asking posture
        return "saas"


def creation_is_consent() -> bool:
    """True on the local edition only — one operator, so filing a ticket IS approving it."""
    return edition() == LOCAL_EDITION


def consent_for_created_ticket(
    db: Any, *, workspace_id: Any, task: Any, actor: str, why: str,
) -> str:
    """Pre-approve a ticket the operator just filed AND assigned (local edition).

    Returns ``"skipped"`` when the edition asks, or the ticket is not about to be
    dispatched (unassigned, or not in ``assigned``); otherwise the outcome of
    :func:`record_operator_consent`.
    """
    if not creation_is_consent():
        return SKIPPED
    agent_id = getattr(task, "assigned_agent_id", None)
    if getattr(task, "status", None) != "assigned" or not agent_id:
        return SKIPPED
    return record_operator_consent(
        db, workspace_id=workspace_id, task_id=task.id, agent_id=agent_id, actor=actor, why=why,
    )


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
