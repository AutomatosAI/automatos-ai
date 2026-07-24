"""HARNESS self-management command handler (PRD-141 US-025).

Processes inbound ``/approve`` and ``/reject`` commands for queued high-risk
HARNESS prescriptions (the ones US-024 escalates to a channel). Every command
is gated behind two checks, in this order, BEFORE any state is touched:

  1. ``HARNESS_SELF_MANAGEMENT_ENABLED`` — Phase 5 stays inert by default, so a
     disabled platform answers every command with a plain "disabled" message
     and changes nothing (mirrors ``_apply_approved_board_tasks``).
  2. A workspace-ADMIN authorization check on the *calling user*. A caller who
     is not an active ``owner``/``admin`` member of the workspace can never
     mutate state — the command is refused before any task or agent is read or
     written. This is the security boundary US-026 reviews.

``/approve`` short-circuits the weekly tick: it applies the prescription now via
the existing ``HarnessService._auto_apply_prescription`` and records the board
task id in the shared US-021 ledger, so the next tick never re-applies it.
``/reject`` marks the board task ``rejected`` so ``_get_rejected_signatures``
suppresses the same prescription on future ticks.

Wiring into the live channel inbound path is done in US-026 (the gate that
enables the flag on a canary workspace and exercises this path end-to-end).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from services.harness_service import get_harness_service

logger = logging.getLogger(__name__)


def _make_executor(db, workspace_id: UUID):
    """Construct a PlatformActionExecutor (lazy import keeps the heavy action /
    RAG dependency chain out of module load — and is the test injection seam)."""
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    return PlatformActionExecutor(db=db, workspace_id=workspace_id)


def _caller_user_id(caller_identity: Optional[Dict[str, Any]]) -> Optional[int]:
    """Extract a positive integer platform user_id from caller_identity, or None.

    Fail-closed and deterministic: rejects a non-dict identity, a missing/None
    user_id, a bool (``True`` would otherwise coerce to user 1), a non-numeric
    value, and any non-positive id. Centralising the coercion here means the
    authz query and the audit fields all see the exact same validated id rather
    than relying on DB-driver coercion of caller-controlled input.
    """
    if not isinstance(caller_identity, dict):
        return None
    raw = caller_identity.get("user_id")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        uid = int(raw)
    except (TypeError, ValueError):
        return None
    return uid if uid > 0 else None


def _caller_is_workspace_admin(
    db, workspace_id: UUID, caller_identity: Optional[Dict[str, Any]]
) -> bool:
    """Return True only if caller_identity is an active owner/admin member.

    The single authorization primitive for every HARNESS command mutation, and
    deliberately fail-closed: a non-dict identity, a missing ``user_id``, an
    unknown user, a non-admin role, or any error all yield False. Mirrors the
    membership check the PlatformActionExecutor admin gate uses, but keyed to
    the *specific* calling user rather than "the workspace has some admin".
    """
    user_id = _caller_user_id(caller_identity)
    if user_id is None:
        return False
    try:
        from core.workspaces.models import WorkspaceMember

        member = (
            db.query(WorkspaceMember)
            .filter(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
                WorkspaceMember.role.in_(("owner", "admin")),
                WorkspaceMember.is_active.is_(True),
            )
            .first()
        )
        return member is not None
    except Exception:
        logger.exception(
            "[HARNESS] authorization check failed for workspace=%s user=%s",
            workspace_id, user_id,
        )
        return False


def _find_task_by_rx(tasks: Any, rx_id: str) -> Optional[Dict[str, Any]]:
    """Find the queued HARNESS board task carrying the ``rx:{rx_id}`` tag.

    US-024 tags every queued prescription with ``harness`` and ``rx:{rx_id}``;
    the lookup keys on both so an unrelated ``harness`` task can never match.
    """
    rx_tag = f"rx:{rx_id}"
    if not isinstance(tasks, list):
        return None
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tags = task.get("tags") or []
        if "harness" in tags and rx_tag in tags:
            return task
    return None


async def handle_harness_command(
    db,
    workspace_id: UUID,
    command: str,
    rx_id: str,
    caller_identity: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Process a ``/approve`` or ``/reject`` command for a HARNESS prescription.

    Returns ``{"success": bool, "message": str, ...}``. Authorization is enforced
    FIRST: an unauthorized caller is refused before any state is read or written.
    """
    from config import config

    # Phase 5 stays dark unless explicitly enabled. No command does anything.
    if not config.HARNESS_SELF_MANAGEMENT_ENABLED:
        return {"success": False, "message": "HARNESS self-management is disabled"}

    # Authorization is non-negotiable and happens before ANY mutation.
    if not _caller_is_workspace_admin(db, workspace_id, caller_identity):
        logger.warning(
            "[HARNESS] Unauthorized '%s' for rx=%s in workspace=%s by user=%s",
            command, rx_id, workspace_id,
            (caller_identity or {}).get("user_id", "unknown")
            if isinstance(caller_identity, dict) else "unknown",
        )
        return {
            "success": False,
            "unauthorized": True,
            "message": "Only a workspace admin can approve or reject HARNESS changes",
        }

    cmd = (command or "").lstrip("/").strip().lower()
    if cmd not in ("approve", "reject"):
        return {"success": False, "message": f"Unknown command: {command}"}

    if not rx_id:
        return {"success": False, "message": "Missing prescription id"}

    svc = get_harness_service()
    executor = _make_executor(db, workspace_id)

    # Locate the queued board task for this prescription (tag-keyed).
    list_result = await executor.execute("platform_list_tasks", {"tags": ["harness"]})
    tasks: Any = []
    if isinstance(list_result, dict):
        tasks = list_result.get("data", list_result.get("tasks", [])) or []
    task = _find_task_by_rx(tasks, rx_id)
    if not task:
        return {
            "success": False,
            "message": f"No pending HARNESS change found for {rx_id}",
        }

    # A rejected prescription must never be resurrected by a later /approve:
    # the board task keeps its 'harness'/'rx:' tags after rejection, so it still
    # surfaces in the tag-filtered list and _find_task_by_rx still matches it.
    # /reject of an already-rejected task is a harmless idempotent no-op.
    if cmd == "approve" and str(task.get("status") or "").lower() == "rejected":
        return {
            "success": False,
            "message": f"{rx_id} was already rejected and cannot be approved",
        }

    if cmd == "approve":
        return await _approve(db, svc, executor, workspace_id, task, rx_id, caller_identity)
    return await _reject(db, workspace_id, task, rx_id, caller_identity)


async def _approve(
    db,
    svc,
    executor: Any,
    workspace_id: UUID,
    task: Dict[str, Any],
    rx_id: str,
    caller_identity: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Actuate an approved prescription as a GOVERNED ACTIVATION (PRD-179 S3,
    F048) and record it in the US-021 ledger.

    The actuation is routed through the Wave-4 policy plane's ask verdict
    (``core.services.approval_policy.evaluate_approval``) — the same verdict every
    other governed action passes through — rather than a dark direct apply. The
    admin's explicit /approve is the per-request override the plane already
    models, but the plane can still deny (fail-safe). Only when the verdict
    approves does the prescription actuate; the board task is then marked done
    with the actuation result written to it (status=done, result != null).
    """
    from core.services.approval_policy import evaluate_approval

    task_id = str(task.get("id"))
    user_id = _caller_user_id(caller_identity)

    # Idempotency: the shared applied-tasks ledger is the source of truth, so a
    # second /approve (or a later tick) never re-applies the same change.
    ledger = svc._read_applied_tasks(workspace_id)
    applied_ids = {str(i) for i in ledger.get("applied_task_ids", [])}
    if task_id in applied_ids:
        return {
            "success": True,
            "already_applied": True,
            "message": f"{rx_id} was already applied",
        }

    agents_by_name = await svc._resolve_agents_by_name(executor)
    rx = svc._parse_harness_task(task, agents_by_name=agents_by_name)
    if not rx or rx.get("target_id") is None:
        # Never apply a change to an unresolved / guessed target.
        return {
            "success": False,
            "message": f"Could not resolve the target for {rx_id}; nothing applied",
        }

    # GOVERNED ACTIVATION: route through the policy plane's ask verdict before
    # actuating. A self-management change makes no LLM spend, so the estimated
    # cost is $0; the admin approve is the explicit per-request override. The
    # plane can still refuse (e.g. a locked-down workspace policy), in which case
    # the prescription does NOT actuate.
    decision = evaluate_approval(
        db, workspace_id, 0.0, override_auto_approve=True
    )
    if not decision.auto_approve:
        logger.info(
            "[HARNESS] Policy plane declined actuation of rx=%s in workspace=%s: %s",
            rx_id, workspace_id, decision.reason,
        )
        return {
            "success": False,
            "message": f"Policy plane declined {rx_id}: {decision.reason}",
        }

    current_before = svc._snapshot_current_value(rx)
    apply_result = await svc._auto_apply_prescription(executor, rx)
    if not apply_result.get("success"):
        return {
            "success": False,
            "message": f"Failed to apply {rx_id}: {apply_result.get('error', 'unknown')}",
        }

    # Apply succeeded — mark the board task done WITH the actuation result, so a
    # completed governed action never leaves a null result (and a failed apply
    # above never marks the task done). status='done' also sets completed_at.
    await executor.execute(
        "platform_update_task_status", {"task_id": task.get("id"), "status": "done"}
    )
    _record_board_task_result(db, workspace_id, task.get("id"), apply_result, decision)

    entry = {
        "task_id": task_id,
        "prescription_id": rx.get("prescription_id"),
        "target_id": rx.get("target_id"),
        "target_name": rx.get("target_name"),
        "change_type": rx.get("change_type"),
        "current_value_before": current_before,
        "proposed_value": rx.get("proposed_value"),
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "approved_via": "command",
        "approved_by": user_id,
        "policy_verdict": decision.reason,
    }
    applied_ids.add(task_id)
    svc._write_applied_tasks(workspace_id, ledger, applied_ids, [entry])

    logger.info(
        "[HARNESS] APPROVED rx=%s (%s for %s) in workspace=%s by user=%s via policy plane",
        rx_id, rx.get("change_type"), rx.get("target_name"), workspace_id, user_id,
    )
    return {
        "success": True,
        "message": f"Applied {rx.get('change_type')} for {rx.get('target_name')}",
        "change_type": rx.get("change_type"),
        "target_name": rx.get("target_name"),
    }


def _record_board_task_result(
    db, workspace_id: UUID, raw_task_id: Any, apply_result: Dict[str, Any], decision
) -> None:
    """Write the actuation outcome onto the board task's ``result`` column.

    The status action carries no result field, so — as with ``_reject`` — the
    ORM row is set directly (this is internal server code, not an LLM tool call).
    Best-effort: a failure here must not undo an already-applied change, so it is
    logged and swallowed. Keeps the completed governed task from carrying a null
    result (the F048 acceptance: status=done, result != null).
    """
    import json as _json

    from core.models.core import BoardTask

    try:
        row = (
            db.query(BoardTask)
            .filter(
                BoardTask.id == int(raw_task_id),
                BoardTask.workspace_id == workspace_id,
            )
            .first()
        )
    except (TypeError, ValueError):
        row = None
    if not row:
        return
    try:
        row.result = _json.dumps({
            "actuated": True,
            "policy_verdict": decision.reason,
            "apply_result": apply_result.get("data", apply_result),
        })
        row.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        logger.warning(
            "[HARNESS] Could not persist board-task result for task %s", raw_task_id,
            exc_info=True,
        )


async def _reject(
    db,
    workspace_id: UUID,
    task: Dict[str, Any],
    rx_id: str,
    caller_identity: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mark the board task rejected so it is never proposed or applied again."""
    from core.models.core import BoardTask

    raw_id = task.get("id")
    try:
        row = (
            db.query(BoardTask)
            .filter(
                BoardTask.id == int(raw_id),
                BoardTask.workspace_id == workspace_id,
            )
            .first()
        )
    except (TypeError, ValueError):
        row = None
    if not row:
        return {
            "success": False,
            "message": f"No pending HARNESS change found for {rx_id}",
        }

    # The board action layer only permits the five kanban statuses, so set
    # 'rejected' on the ORM row directly (this is internal server code, not an
    # LLM tool call). The title is left untouched: it carries the signature
    # _get_rejected_signatures matches on to suppress re-proposal next tick.
    user_id = _caller_user_id(caller_identity)
    row.status = "rejected"
    row.blocked_reason = f"Rejected by workspace admin (user {user_id}) via command"
    row.blocked_at = datetime.now(timezone.utc)
    db.commit()

    logger.info(
        "[HARNESS] REJECTED rx=%s in workspace=%s by user=%s", rx_id, workspace_id, user_id,
    )
    return {
        "success": True,
        "rejected": True,
        "message": f"Rejected {rx_id}; HARNESS will not propose it again",
    }
