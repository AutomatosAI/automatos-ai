"""
Playbook rerun + tweak -- PRD-204 S7
====================================

The platform's first rerun primitive: copy a prior execution's inputs into a
NEW ``RecipeExecution`` (``retry_of`` lineage, ``attempt_count+1`` -- the
``task_reconciler`` stall-retry idiom) with optional per-execution
``step_overrides`` that the executor merges at run start.
``workflow_recipes.steps`` is NEVER mutated -- a tweak affects exactly one
execution.

Approval plane (PRD-204 Section 8 Q3): every rerun rides
``evaluate_approval`` with the ORIGINAL run's ``llm_usage`` dollar sum as
the estimate. Auto path launches immediately; ask path parks a durable
``ApprovalGrant(subject_type=SUBJECT_PLAYBOOK_RUN)`` -- the first real use
of that subject kind -- carrying the full rerun spec in ``details`` so the
grant endpoint can launch it later with zero re-derivation.

Grant details schema (stable contract for the approvals inbox + S8):

    {
      "watch_action": "rerun",            # discriminator (S8 adds more)
      "watch_id": "<uuid>" | null,
      "rerun_of": "exec-...",             # original execution_id
      "recipe_id": 123,
      "spec": {
        "input_data": {...},              # copied from the original
        "step_overrides": {"<step_id>": {"prompt_template": "..."}} | null,
        "triggered_by": "rerun" | "watch_rerun",
        "attempt_count": 2
      },
      "executed_result": {...}            # written by resume/deny
    }

``resume_playbook_run_grant`` / ``fail_playbook_run_grant`` are the
``_requeue_subject`` / ``_fail_subject`` branches for ALL watch corrective
actions -- ``details["watch_action"]`` discriminates; S8 registers the
mission-action executors via ``WATCH_GRANT_EXECUTORS``.

Transactions: ``request_rerun`` COMMITS on the auto path (the playbook
engine's task opens its own session, so the row must be visible before
launch -- the execute-route precedent); the ask path only flushes and the
caller commits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from core.models.core import RecipeExecution, WorkflowTemplate
from core.services.approval_policy import ApprovalDecision, evaluate_approval

logger = logging.getLogger(__name__)

TRIGGERED_BY_HUMAN = "rerun"
TRIGGERED_BY_WATCH = "watch_rerun"

# S8 registers mission-action executors here: {watch_action: async fn(db, grant) -> dict}
WATCH_GRANT_EXECUTORS: Dict[str, Callable[[Session, Any], Awaitable[Dict[str, Any]]]] = {}


@dataclass(frozen=True)
class RerunOutcome:
    """Result of a gated rerun request."""

    launched: bool
    execution_id: Optional[str]
    grant_id: Optional[int]
    decision: ApprovalDecision
    message: str


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


def validate_step_overrides(
    recipe: WorkflowTemplate, step_overrides: Any
) -> Tuple[Optional[Dict[str, Dict[str, str]]], Optional[str]]:
    """Validate + normalise ``{step_id: {prompt_template: str}}``.

    Returns (normalised, None) or (None, error). Unknown step ids and
    non-string prompts are rejected at the boundary (never trust input).
    """
    if step_overrides in (None, {}):
        return None, None
    if not isinstance(step_overrides, dict):
        return None, "step_overrides must be an object of {step_id: {prompt_template}}"

    known_ids = {
        str(s.get("step_id"))
        for s in (recipe.steps or [])
        if isinstance(s, dict) and s.get("step_id")
    }
    normalised: Dict[str, Dict[str, str]] = {}
    for step_id, override in step_overrides.items():
        sid = str(step_id)
        if sid not in known_ids:
            return None, f"Unknown step_id in step_overrides: {sid}"
        if not isinstance(override, dict):
            return None, f"step_overrides[{sid}] must be an object"
        prompt = override.get("prompt_template")
        if not isinstance(prompt, str) or not prompt.strip():
            return None, (
                f"step_overrides[{sid}].prompt_template must be a non-empty string"
            )
        normalised[sid] = {"prompt_template": prompt}
    return (normalised or None), None


# ---------------------------------------------------------------------------
# Cost estimate (the original run's llm_usage dollar sum)
# ---------------------------------------------------------------------------


def estimate_rerun_cost_usd(
    db: Session, workspace_id: UUID | str, execution_id: str
) -> float:
    """What the original execution cost -- the honest rerun estimate."""
    try:
        from services.report_service import compute_execution_metrics

        metrics = compute_execution_metrics(
            db, workspace_id, execution_id=str(execution_id)
        )
        return float(metrics.get("cost_usd") or 0.0)
    except Exception:
        logger.warning(
            "[WatchRerun] cost estimate failed for %s -- using 0.0",
            execution_id,
            exc_info=True,
        )
        return 0.0


# ---------------------------------------------------------------------------
# The copy idiom (task_reconciler precedent, ORM shape)
# ---------------------------------------------------------------------------


def create_rerun_execution(
    db: Session,
    recipe: WorkflowTemplate,
    original: RecipeExecution,
    *,
    step_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    triggered_by: str = TRIGGERED_BY_HUMAN,
) -> RecipeExecution:
    """Stage the copied execution row (flush only -- caller owns commit).

    Copies ``input_data``, chains ``retry_of``, bumps ``attempt_count``.
    ``step_overrides`` live in ``execution_metadata`` for the executor's
    run-start merge -- the recipe row is untouched.
    """
    metadata: Dict[str, Any] = {
        "execution_type": "recipe_direct",
        "total_steps": len(recipe.steps or []),
        "rerun_of": original.execution_id,
    }
    if step_overrides:
        metadata["step_overrides"] = step_overrides

    execution = RecipeExecution(
        execution_id=f"rerun-{uuid4().hex[:12]}",
        recipe_id=recipe.id,
        workspace_id=original.workspace_id,
        status="pending",
        input_data=dict(original.input_data or {}),
        current_step=0,
        triggered_by=triggered_by,
        execution_metadata=metadata,
        attempt_count=(original.attempt_count or 1) + 1,
        retry_of=original.execution_id,
    )
    db.add(execution)
    db.flush()
    return execution


def launch_execution(execution: RecipeExecution) -> None:
    """Fire the execution through the consolidated PlaybookEngine seam
    (PRD-142 W3-S12: every backend launch site goes through the engine).
    The row MUST be committed before this call (the task opens its own
    session)."""
    from services.playbook_engine import get_playbook_engine

    get_playbook_engine().launch(
        recipe_execution_id=execution.execution_id,
        recipe_id=execution.recipe_id,
        workspace_id=UUID(str(execution.workspace_id)),
        input_data=execution.input_data or {},
    )


# ---------------------------------------------------------------------------
# Notifications (best-effort, never raise)
# ---------------------------------------------------------------------------


async def _notify_approval_pending(
    db: Session, workspace_id, grant, recipe: WorkflowTemplate
) -> None:
    try:
        from core.services.notification_dispatcher import NotificationDispatcher

        dispatcher = NotificationDispatcher(db, str(workspace_id))
        await dispatcher.dispatch(
            event_type="approval_pending",
            title=f"Approval needed: rerun '{recipe.name}'",
            message=(
                f"A rerun of playbook '{recipe.name}' is waiting for approval "
                f"(estimated ${float(grant.estimated_cost_usd or 0):.2f}). "
                "Review it in the approvals inbox."
            ),
            link_type="approval_grant",
            link_id=str(grant.id),
            status="warning",
        )
    except Exception:
        logger.error(
            "[WatchRerun] approval_pending dispatch failed for grant %s",
            getattr(grant, "id", "?"),
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# The gated request
# ---------------------------------------------------------------------------


async def request_rerun(
    db: Session,
    *,
    workspace_id: UUID | str,
    recipe: WorkflowTemplate,
    original: RecipeExecution,
    step_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    triggered_by: str = TRIGGERED_BY_HUMAN,
    watch=None,
    override_auto_approve: bool = False,
) -> RerunOutcome:
    """Rerun an execution through the approval policy.

    Auto path: creates + COMMITS + launches the copy; a supervising watch
    follows the new execution (lineage repoint -- same watch, one verdict).
    Ask path: stages a PENDING ``SUBJECT_PLAYBOOK_RUN`` grant with the rerun
    spec in details, parks the watch in ``awaiting_approval``, fires the
    ``approval_pending`` notification. Caller commits.
    """
    estimated = estimate_rerun_cost_usd(db, workspace_id, original.execution_id)
    decision = evaluate_approval(
        db, workspace_id, estimated, override_auto_approve=override_auto_approve
    )

    if decision.auto_approve:
        execution = create_rerun_execution(
            db,
            recipe,
            original,
            step_overrides=step_overrides,
            triggered_by=triggered_by,
        )
        if watch is not None:
            _follow_watch(db, watch, execution, step_overrides, triggered_by)
        db.commit()  # row must be visible to the engine's own session
        launch_execution(execution)
        logger.info(
            "[WatchRerun] launched rerun %s of %s (auto: %s)",
            execution.execution_id,
            original.execution_id,
            decision.reason,
        )
        return RerunOutcome(
            launched=True,
            execution_id=execution.execution_id,
            grant_id=None,
            decision=decision,
            message=f"Rerun {execution.execution_id} launched.",
        )

    # --- ask path: durable grant carrying the spec ---
    from core.models.approval_grants import SUBJECT_PLAYBOOK_RUN
    from core.services.approval_grants import create_grant, find_pending_grant

    existing = find_pending_grant(
        db,
        workspace_id,
        subject_type=SUBJECT_PLAYBOOK_RUN,
        subject_id=original.execution_id,
    )
    if existing is not None:
        logger.info(
            "[WatchRerun] pending rerun grant %s already exists for %s",
            existing.id,
            original.execution_id,
        )
        return RerunOutcome(
            launched=False,
            execution_id=None,
            grant_id=existing.id,
            decision=decision,
            message="A rerun approval is already pending for this execution.",
        )

    grant = create_grant(
        db,
        workspace_id,
        subject_type=SUBJECT_PLAYBOOK_RUN,
        subject_id=original.execution_id,
        tool_name="playbook_rerun",
        reason=decision.reason,
        estimated_cost_usd=decision.estimated_cost,
    )
    watch_id = str(watch.id) if watch is not None and getattr(watch, "id", None) else None
    grant.details = {
        "watch_action": "rerun",
        "watch_id": watch_id,
        "rerun_of": original.execution_id,
        "recipe_id": recipe.id,
        "spec": {
            "input_data": dict(original.input_data or {}),
            "step_overrides": step_overrides,
            "triggered_by": triggered_by,
            "attempt_count": (original.attempt_count or 1) + 1,
        },
    }
    db.flush()

    if watch is not None:
        _park_watch_awaiting_approval(db, watch, grant)

    await _notify_approval_pending(db, workspace_id, grant, recipe)

    return RerunOutcome(
        launched=False,
        execution_id=None,
        grant_id=grant.id,
        decision=decision,
        message=(
            f"Rerun needs approval ({decision.reason}). "
            f"Grant {grant.id} is pending in the approvals inbox."
        ),
    )


def _follow_watch(db, watch, execution, step_overrides, triggered_by) -> None:
    """The watch follows the rerun (Section 8 Q9); overrides land on the
    FOLLOW event snapshot for before/after comparison. Fail-soft."""
    try:
        from services.watch_service import WatchService

        WatchService.follow(
            db,
            watch,
            new_target_type="playbook_execution",
            new_target_id=execution.execution_id,
            reason=f"rerun of {execution.retry_of} ({triggered_by})",
            snapshot={
                "rerun_of": execution.retry_of,
                "step_overrides": step_overrides,
                "triggered_by": triggered_by,
            },
        )
    except Exception:
        logger.warning(
            "[WatchRerun] watch follow failed for %s (non-fatal)",
            getattr(watch, "id", "?"),
            exc_info=True,
        )


def _park_watch_awaiting_approval(db, watch, grant) -> None:
    """awaiting_approval park + event. Fail-soft (the grant is the truth)."""
    try:
        from core.models.watch_enums import WatchEventType, WatchStatus
        from services.watch_service import WatchService

        WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.STATUS_CHANGE.value,
            event_key=f"grant-pending:{grant.id}",
            summary=f"Corrective action parked on approval grant {grant.id}",
            snapshot={"grant_id": grant.id, "watch_action": "rerun"},
        )
        if WatchStatus(watch.status) != WatchStatus.AWAITING_APPROVAL:
            WatchService.transition(
                db,
                watch,
                WatchStatus.AWAITING_APPROVAL,
                reason=f"awaiting grant {grant.id}",
            )
    except Exception:
        logger.warning(
            "[WatchRerun] could not park watch %s on grant %s (non-fatal)",
            getattr(watch, "id", "?"),
            grant.id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Grant resume/deny -- the _requeue_subject / _fail_subject branches
# ---------------------------------------------------------------------------


def _load_watch(db: Session, grant) -> Optional[Any]:
    details = grant.details if isinstance(grant.details, dict) else {}
    watch_id = details.get("watch_id")
    if not watch_id:
        return None
    try:
        from services.watch_service import WatchService

        return WatchService.get_watch(db, grant.workspace_id, watch_id)
    except Exception:
        logger.warning("[WatchRerun] watch load failed for grant %s", grant.id, exc_info=True)
        return None


async def resume_playbook_run_grant(db: Session, grant) -> None:
    """Granted: execute the stored watch action (PRD-204 S7/S8).

    ``details["watch_action"]`` discriminates. ``rerun`` is handled here;
    mission actions (replan/reassign/spawn_agent) execute via the S8
    registry. The outcome lands on ``details.executed_result`` -- fail LOUD
    but contained (an honest failure on the grant, never an exception out of
    the grant endpoint).
    """
    details = dict(grant.details) if isinstance(grant.details, dict) else {}
    action = details.get("watch_action")

    try:
        if action == "rerun":
            result = await _resume_rerun(db, grant, details)
        elif action in WATCH_GRANT_EXECUTORS:
            result = await WATCH_GRANT_EXECUTORS[action](db, grant)
        else:
            result = {
                "success": False,
                "error": f"grant carries no executable watch_action ({action!r})",
            }
    except Exception as exc:  # noqa: BLE001 -- contained per grant contract
        logger.error(
            "[WatchRerun] playbook_run grant %s resume failed", grant.id, exc_info=True
        )
        result = {"success": False, "error": str(exc)[:500]}

    grant.details = {
        **details,
        "executed_result": {**result, "executed_at": _utcnow_iso()},
    }


async def _resume_rerun(db: Session, grant, details: Dict[str, Any]) -> Dict[str, Any]:
    spec = details.get("spec") or {}
    recipe = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.id == details.get("recipe_id"),
            WorkflowTemplate.workspace_id == grant.workspace_id,
        )
        .first()
    )
    original = (
        db.query(RecipeExecution)
        .filter(
            RecipeExecution.execution_id == details.get("rerun_of"),
            RecipeExecution.workspace_id == grant.workspace_id,
        )
        .first()
    )
    watch = _load_watch(db, grant)

    if recipe is None or original is None:
        _park_watch_needs_attention(
            db, watch, reason=f"grant {grant.id}: rerun target no longer exists"
        )
        return {"success": False, "error": "recipe or original execution not found"}

    execution = create_rerun_execution(
        db,
        recipe,
        original,
        step_overrides=spec.get("step_overrides"),
        triggered_by=spec.get("triggered_by") or TRIGGERED_BY_WATCH,
    )

    if watch is not None:
        _resume_watch_to_watching(db, watch)
        _follow_watch(
            db, watch, execution, spec.get("step_overrides"), spec.get("triggered_by")
        )

    # The engine's task opens its own session: persist the row (and the
    # watch/grant state staged so far) before launching.
    db.commit()
    launch_execution(execution)
    logger.info(
        "[WatchRerun] grant %s resumed -> launched %s (rerun of %s)",
        grant.id,
        execution.execution_id,
        original.execution_id,
    )
    return {"success": True, "execution_id": execution.execution_id}


def fail_playbook_run_grant(db: Session, grant) -> None:
    """Denied: no launch; a supervised watch parks in needs_attention."""
    details = dict(grant.details) if isinstance(grant.details, dict) else {}
    watch = _load_watch(db, grant)
    _park_watch_needs_attention(
        db,
        watch,
        reason=f"approval grant {grant.id} denied by a human reviewer",
    )
    grant.details = {
        **details,
        "executed_result": {
            "success": False,
            "error": "denied by a human reviewer",
            "executed_at": _utcnow_iso(),
        },
    }


def _resume_watch_to_watching(db, watch) -> None:
    try:
        from core.models.watch_enums import WatchStatus
        from services.watch_service import WatchService

        if WatchStatus(watch.status) == WatchStatus.AWAITING_APPROVAL:
            WatchService.transition(
                db, watch, WatchStatus.WATCHING, reason="grant approved"
            )
    except Exception:
        logger.warning(
            "[WatchRerun] watch %s resume-to-watching failed (non-fatal)",
            getattr(watch, "id", "?"),
            exc_info=True,
        )


def _park_watch_needs_attention(db, watch, *, reason: str) -> None:
    if watch is None:
        return
    try:
        from core.models.watch_enums import WatchEventType, WatchStatus
        from services.watch_service import WatchService

        WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.STATUS_CHANGE.value,
            event_key=f"grant-denied:{reason[:80]}",
            summary=reason,
            requires_attention=True,
        )
        if WatchStatus(watch.status) not in (
            WatchStatus.NEEDS_ATTENTION,
            WatchStatus.PASSED,
            WatchStatus.FAILED,
            WatchStatus.EXPIRED,
            WatchStatus.CANCELLED,
        ):
            WatchService.transition(
                db, watch, WatchStatus.NEEDS_ATTENTION, reason=reason
            )
    except Exception:
        logger.warning(
            "[WatchRerun] could not park watch %s (non-fatal)",
            getattr(watch, "id", "?"),
            exc_info=True,
        )
