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

from core import editions

logger = logging.getLogger(__name__)

WHY_MOVED_TO_IN_PROGRESS = "the operator moved the ticket to In Progress"
WHY_RUN_NOW = "the operator pressed Run Now"
WHY_CREATED_AND_ASSIGNED = "the operator created the ticket and assigned it"
WHY_ASSIGNED_BY_HAND = "the operator assigned the ticket on the board"  # F275
WHY_ASKED_IN_CHAT = "the operator asked Auto for it in chat"
WHY_SCHEDULED_LANE = "the operator's standing schedule filed it (heartbeat / lane)"
WHY_SCHEDULED_AND_ASSIGNED = "the operator scheduled the ticket and assigned it"

SKIPPED = "skipped"
# What record_operator_consent found or did (its return value).
ACTIVE = "active"
GRANTED = "granted"
CREATED = "created"
ERROR = "error"

# PRD-234 D16: tool calls that file, assign or re-queue a board ticket carry the
# driving human so the local edition can record consent before dispatch.
# platform_schedule_task joins them (calendar): a scheduled board ticket is
# filed at fire time, and the operator who scheduled it is that consent.
# (Moved from platform_executor.py, which imports it: PRD-268 W3 K3.)
OPERATOR_CONSENT_ACTIONS = (
    "platform_create_task",
    "platform_assign_task",
    "platform_update_task",
    "platform_update_task_status",
    "platform_schedule_task",
)


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


def driver_from_caller_context(caller_context: Any) -> Optional[str]:
    """The human behind a chat tool call, or None for an autonomous run.

    ``user_id`` is the driving Clerk principal (SaaS); on the local edition the
    operator has no Clerk id (``users.clerk_user_id`` is NULL), so the chat
    service also threads the internal ``driving_user_id``. Heartbeat, schedule
    and channel runs thread neither — they keep asking.
    """
    cc = caller_context if isinstance(caller_context, dict) else {}
    driver = cc.get("user_id") or cc.get("driving_user_id")
    return str(driver) if driver else None


def creation_is_consent() -> bool:
    """True where one person runs the install (``multi_person`` off: the local edition), so
    filing a ticket IS approving it. An edition that cannot be read keeps the asking posture."""
    try:
        return not editions.current().multi_person
    except Exception:  # an unreadable edition keeps the asking posture
        logger.exception("[board_consent] the edition could not be read; filing a ticket is not its approval")
        return False


def consent_for_lane_ticket(
    db: Any, *, workspace_id: Any, task: Any, source_type: str, commit: bool = True,
) -> str:
    """Pre-approve a ticket a scheduled lane just filed (local edition).

    Owner's words (2026-09-07): "if I ask Auto to create a task then just do it —
    less approval, more automation." On the local edition the operator enabled
    the heartbeat / schedule that files the ticket, so the standing schedule IS
    the approval; ``always_ask`` keeps holding only work nobody asked for.
    SaaS is untouched (``creation_is_consent`` is False there). ``commit`` is
    :func:`record_operator_consent`'s.
    """
    return consent_for_created_ticket(
        db, workspace_id=workspace_id, task=task,
        actor=f"lane:{source_type or 'schedule'}", why=WHY_SCHEDULED_LANE, commit=commit,
    )


def consent_for_created_ticket(
    db: Any, *, workspace_id: Any, task: Any, actor: str, why: str, commit: bool = True,
) -> str:
    """Pre-approve a ticket the operator just filed AND assigned (local edition).

    Returns ``"skipped"`` when the edition asks, or the ticket is not about to be
    dispatched (unassigned, or not in ``assigned``); otherwise the outcome of
    :func:`record_operator_consent`, which ``commit`` is passed to.
    """
    if not creation_is_consent():
        return SKIPPED
    agent_id = getattr(task, "assigned_agent_id", None)
    if getattr(task, "status", None) != "assigned" or not agent_id:
        return SKIPPED
    return record_operator_consent(
        db, workspace_id=workspace_id, task_id=task.id, agent_id=agent_id, actor=actor, why=why, commit=commit,
    )


def record_operator_consent(
    db: Any, *, workspace_id: Any, task_id: int, agent_id: Optional[int], actor: str, why: str,
    commit: bool = True,
) -> str:
    """Make sure an ACTIVE board-task grant exists for ``task_id``.

    Returns ``"active"`` (one already authorised the task), ``"granted"`` (a
    pending grant was approved), ``"created"`` (a granted grant was recorded), or
    ``"error"`` (nothing recorded — the gate will ask as before; never raises).

    ``commit=False`` is for the code that files the ticket: the grant is staged in
    the caller's transaction, inside a savepoint, and nothing is committed, so the
    ticket and its grant become visible in one commit. Committed apart, the
    dispatcher claimed scheduled ticket 343 between the two commits, found no grant
    and parked it behind a new request. A failure then rolls back the savepoint
    alone: the caller's ticket still commits, and the gate asks as before.
    """
    try:
        if not commit:
            with db.begin_nested():
                return _stage_grant(db, workspace_id=workspace_id, task_id=task_id, agent_id=agent_id,
                                    actor=actor, why=why)
        outcome = _stage_grant(db, workspace_id=workspace_id, task_id=task_id, agent_id=agent_id,
                               actor=actor, why=why)
        if outcome != ACTIVE:
            db.commit()
        return outcome
    except Exception:  # noqa: BLE001 — consent is a convenience; the gate stays the safety net
        logger.warning("[BoardConsent] could not record consent for task %s", task_id, exc_info=True)
        if not commit:
            return ERROR  # the savepoint is rolled back; the caller's own writes are left to commit
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return ERROR


def _stage_grant(
    db: Any, *, workspace_id: Any, task_id: int, agent_id: Optional[int], actor: str, why: str,
) -> str:
    """Stage the granted board-task grant ``record_operator_consent`` promises; commits nothing."""
    from core.models.approval_grants import KIND_APPROVAL, SUBJECT_BOARD_TASK
    from core.services.approval_grants import (
        create_grant, find_active_grant, find_pending_grant, grant_grant,
    )
    subject = str(task_id)
    if find_active_grant(db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=subject) is not None:
        return ACTIVE
    # F259: only an approval is consent. An open question waits for its answer:
    # granting it here dropped it from the Questions tab unanswered.
    pending = find_pending_grant(db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=subject,
                                 kind=KIND_APPROVAL)
    if pending is not None:
        grant_grant(pending, granted_by=actor)
        return GRANTED
    grant = create_grant(
        db, workspace_id, subject_type=SUBJECT_BOARD_TASK, subject_id=subject,
        agent_id=agent_id, reason=why,
    )
    grant_grant(grant, granted_by=actor)
    return CREATED
