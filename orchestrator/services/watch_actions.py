"""
Watch direction-change actions -- PRD-204 S8
============================================

The mission-watch corrective actions: ``replan`` / ``reassign`` /
``spawn_agent`` / ``escalate``. Every action rides the SAME rails as S7's
rerun: ``WatchService.record_action`` is the budget hard-stop (exhausted ->
escalate), ``evaluate_approval`` is the gate (auto executes now; ask parks
a ``SUBJECT_PLAYBOOK_RUN`` grant whose ``details.watch_action``
discriminates -- the S7 resume/deny branches already handle it via the
``WATCH_GRANT_EXECUTORS`` registry this module populates on import).

Action semantics (all reuse existing production paths -- nothing new grants
itself powers):

- ``replan``    -- ``CoordinatorService.replan_mission`` (the
  platform_replan_mission path) with the watch diagnosis as replanner
  notes; ``failed -> replanning -> running`` transitions stay authoritative.
- ``reassign``  -- requeue THE failed task to a different capable agent:
  ``AgentMatcher.rank`` excluding the agent that failed; if none,
  ``no_capable_agent`` -> escalate. Transition-legal shape (the replan
  idiom, narrowed): run ``failed -> replanning``, failed task ``->
  skipped``, a clone task pinned to the chosen agent (``agent_role`` set to
  the agent's NAME -- the PRD-163 S4 explicit-override the dispatcher
  always honours), dependency edges re-wired, run ``-> running``, clone
  ``-> queued``.
- ``spawn_agent`` -- blueprints only: the workspace must have a blueprint
  (specified or default); the agent is created through the SAME
  ``handlers_agents.create_agent`` path Auto uses, then
  ``blueprint_validator.validate_agent`` stays authoritative (strict-mode
  failures roll the agent back). Empty/defaulted blueprint ``rules`` pass
  -- the onboarding-wall regression stays closed. ALWAYS grant-gated in v1
  (Section 8 Q5): only a ``full_auto`` decision auto-approves;
  ``auto_below_budget``'s auto lane deliberately does NOT.
- ``escalate``  -- ``escalation_service.escalate_watch`` board card (a
  watch-flavoured sibling of the stall escalation, NOT an overload) +
  ``watch_escalation`` notification + watch ``-> escalated``
  (terminal-unless-renewed). Never gated, never budgeted: it is the escape
  hatch.

Budget ordering (documented decision): the action is recorded against the
budget at INITIATION -- before the gate -- so a grant that a human later
denies still consumed an attempt. That is deliberate: the budget bounds how
many times the watcher may ASK the platform to do something, which also
prevents infinite ask-loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from core.services.approval_policy import FULL_AUTO, evaluate_approval

logger = logging.getLogger(__name__)

ACTION_RERUN = "rerun"          # playbook watches (S7 owns execution)
ACTION_REPLAN = "replan"
ACTION_REASSIGN = "reassign"
ACTION_SPAWN_AGENT = "spawn_agent"
ACTION_ESCALATE = "escalate"

MISSION_ACTIONS = frozenset({ACTION_REPLAN, ACTION_REASSIGN, ACTION_SPAWN_AGENT})
# PRD-224 US-003: a board ticket's only corrective action is a re-run through
# the board's own run-now machinery (escalate stays the ungated escape hatch).
BOARD_ACTIONS = frozenset({ACTION_RERUN})


@dataclass(frozen=True)
class WatchActionOutcome:
    """What happened to a requested corrective action."""

    action: str
    executed: bool = False        # ran to completion now
    parked: bool = False          # awaiting an approval grant
    escalated: bool = False       # handed to a human instead
    grant_id: Optional[int] = None
    detail: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Cost estimate for the gate
# ---------------------------------------------------------------------------


def estimate_mission_action_cost_usd(run) -> float:
    """The mission's spend so far -- the honest proxy for redoing the failed
    subtree (replan/reassign re-run similar work; spawn adds an agent doing
    similar work)."""
    spent = (getattr(run, "budget_spent", None) or {}).get("cost")
    if isinstance(spent, (int, float)) and spent > 0:
        return float(spent)
    return 0.0


# ---------------------------------------------------------------------------
# Entry point (used by the S10 decider and, indirectly, the grant resume)
# ---------------------------------------------------------------------------


async def run_mission_action(
    db: Session,
    watch,
    action: str,
    *,
    diagnosis: Optional[str] = None,
    notes: Optional[str] = None,
    spawn_spec: Optional[Dict[str, Any]] = None,
) -> WatchActionOutcome:
    """Attempt a corrective action on a mission watch.

    Flow: resolve run -> (escalate short-circuits) -> budget rail ->
    approval gate -> execute now / park on grant. Never raises; failures
    come back as outcomes the decider can escalate on.
    """
    from services.watch_service import WatchService

    if action == ACTION_ESCALATE:
        await escalate_watch_now(
            db, watch, reason=diagnosis or notes or "Watcher requested escalation"
        )
        return WatchActionOutcome(action=action, escalated=True, detail="escalated")

    if action not in MISSION_ACTIONS:
        return WatchActionOutcome(
            action=action, error=f"unknown mission action {action!r}"
        )

    run = _resolve_run(db, watch)
    if run is None:
        await escalate_watch_now(
            db, watch, reason=f"target mission {watch.target_id} no longer exists"
        )
        return WatchActionOutcome(
            action=action, escalated=True, error="target mission missing"
        )

    # --- budget hard rail (record at initiation; see module docstring) ---
    _, allowed = WatchService.record_action(
        db,
        watch,
        action=action,
        summary=diagnosis or f"Watcher corrective action: {action}",
        snapshot={"diagnosis": diagnosis, "notes": notes},
    )
    if not allowed:
        await escalate_watch_now(
            db,
            watch,
            reason=(
                f"Action budget exhausted "
                f"({watch.actions_taken}/{watch.action_budget}) -- "
                f"refused '{action}'"
            ),
        )
        return WatchActionOutcome(
            action=action, escalated=True, detail="action budget exhausted"
        )

    # --- approval gate ---
    estimated = estimate_mission_action_cost_usd(run)
    decision = evaluate_approval(db, watch.workspace_id, estimated)
    auto = decision.auto_approve
    if action == ACTION_SPAWN_AGENT and decision.policy != FULL_AUTO:
        # Section 8 Q5: spawn is ALWAYS grant-gated in v1 -- only full_auto
        # (with the autonomy gate, enforced inside evaluate_approval)
        # auto-approves; auto_below_budget's auto lane does not apply.
        auto = False

    if auto:
        outcome = await _execute_action(
            db, watch, run, action,
            diagnosis=diagnosis, notes=notes, spawn_spec=spawn_spec,
        )
        if outcome.error and not outcome.escalated:
            await escalate_watch_now(
                db, watch, reason=f"'{action}' failed: {outcome.error}"
            )
            return WatchActionOutcome(
                action=action, escalated=True, error=outcome.error
            )
        return outcome

    grant = _park_action_grant(
        db, watch, run, action,
        decision=decision, diagnosis=diagnosis, notes=notes, spawn_spec=spawn_spec,
    )
    await _notify_action_approval_pending(db, watch, grant, action)
    return WatchActionOutcome(
        action=action,
        parked=True,
        grant_id=grant.id,
        detail=f"awaiting approval grant {grant.id} ({decision.reason})",
    )


async def run_board_task_action(
    db: Session,
    watch,
    action: str,
    *,
    diagnosis: Optional[str] = None,
) -> WatchActionOutcome:
    """PRD-224 US-003: corrective action on a board-task watch.

    ``rerun`` re-dispatches the ticket through the SAME run-now machinery the
    board uses (``api/board_tasks._redispatch_task`` -- the shared function, not
    the HTTP route), budget-railed exactly like ``run_mission_action``: the
    action is recorded against the budget at initiation, so exhaustion hard-stops
    to escalation. No approval gate and no diagnosis LLM -- a re-dispatch is a
    plain replay of already-authorised work. ``escalate`` is the ungated escape
    hatch. Never raises; failures come back as escalated outcomes.
    """
    from services.watch_service import WatchService

    if action == ACTION_ESCALATE:
        await escalate_watch_now(
            db, watch, reason=diagnosis or "Watcher requested escalation"
        )
        return WatchActionOutcome(action=action, escalated=True, detail="escalated")

    if action not in BOARD_ACTIONS:
        return WatchActionOutcome(action=action, error=f"unknown board action {action!r}")

    task = _resolve_board_task(db, watch)
    if task is None:
        await escalate_watch_now(
            db, watch, reason=f"target task {watch.target_id} no longer exists"
        )
        return WatchActionOutcome(
            action=action, escalated=True, error="target task missing"
        )
    if task.assigned_agent_id is None:
        await escalate_watch_now(
            db, watch, reason=f"task {watch.target_id} has no assigned agent to re-run"
        )
        return WatchActionOutcome(
            action=action, escalated=True, error="no assigned agent"
        )

    # --- budget hard rail (record at initiation; see module docstring) ---
    _, allowed = WatchService.record_action(
        db,
        watch,
        action=action,
        summary=diagnosis or f"Watcher corrective re-run of task {task.id}",
        snapshot={"diagnosis": diagnosis, "board_task_id": task.id},
    )
    if not allowed:
        await escalate_watch_now(
            db,
            watch,
            reason=(
                f"Action budget exhausted "
                f"({watch.actions_taken}/{watch.action_budget}) -- refused '{action}'"
            ),
        )
        return WatchActionOutcome(
            action=action, escalated=True, detail="action budget exhausted"
        )

    try:
        from api.board_tasks import _redispatch_task

        _redispatch_task(db, task)
    except Exception as exc:  # noqa: BLE001 -- outcome-shaped, we escalate
        logger.error(
            "[WatchActions] board re-run failed for task %s",
            getattr(task, "id", "?"),
            exc_info=True,
        )
        await escalate_watch_now(db, watch, reason=f"re-run failed: {str(exc)[:200]}")
        return WatchActionOutcome(action=action, escalated=True, error=str(exc)[:300])

    # The watch follows the re-run (same target; the lineage append records the
    # corrective attempt and pulls the next check forward). Fail-soft: a lineage
    # hiccup never undoes the re-dispatch.
    try:
        WatchService.follow(
            db,
            watch,
            new_target_type="board_task",
            new_target_id=str(task.id),
            reason=(
                f"Watch re-run of task {task.id}"
                + (f": {diagnosis}" if diagnosis else "")
            )[:300],
        )
    except Exception:  # noqa: BLE001 -- lineage is bookkeeping, not the action
        logger.warning(
            "[WatchActions] board rerun lineage follow failed (non-fatal)",
            exc_info=True,
        )

    logger.info(
        "[WatchActions] re-dispatched board task %s for watch %s", task.id, watch.id
    )
    return WatchActionOutcome(
        action=action, executed=True, detail=f"task {task.id} re-dispatched"
    )


def _resolve_board_task(db: Session, watch):
    from core.models.core import BoardTask

    try:
        task_id = int(watch.target_id)
    except (TypeError, ValueError):
        return None
    try:
        return (
            db.query(BoardTask)
            .filter(
                BoardTask.id == task_id,
                BoardTask.workspace_id == watch.workspace_id,
            )
            .first()
        )
    except Exception:
        logger.warning(
            "[WatchActions] board task resolve failed for %s", watch.target_id,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


async def _execute_action(
    db: Session,
    watch,
    run,
    action: str,
    *,
    diagnosis: Optional[str],
    notes: Optional[str],
    spawn_spec: Optional[Dict[str, Any]],
) -> WatchActionOutcome:
    if action == ACTION_REPLAN:
        return await _execute_replan(db, watch, run, diagnosis=diagnosis, notes=notes)
    if action == ACTION_REASSIGN:
        return await _execute_reassign(db, watch, run, diagnosis=diagnosis)
    if action == ACTION_SPAWN_AGENT:
        return await _execute_spawn_agent(
            db, watch, run, spawn_spec=spawn_spec, diagnosis=diagnosis
        )
    return WatchActionOutcome(action=action, error=f"no executor for {action!r}")


async def _execute_replan(
    db: Session, watch, run, *, diagnosis: Optional[str], notes: Optional[str]
) -> WatchActionOutcome:
    """Drive the EXISTING coordinator replan path with the watch diagnosis
    as replanner context (the platform_replan_mission service method)."""
    from core.models.orchestration_enums import ActorType
    from services.coordinator_service import CoordinatorService

    replan_notes = "\n\n".join(
        p for p in (notes, f"Watcher diagnosis: {diagnosis}" if diagnosis else None) if p
    ) or None

    try:
        await CoordinatorService().replan_mission(
            db,
            run.id,
            actor_id="watcher",
            notes=replan_notes,
            actor_type=ActorType.COORDINATOR,
            trigger="watch",
        )
    except Exception as exc:  # noqa: BLE001 -- outcome-shaped, caller escalates
        logger.error(
            "[WatchActions] replan failed for run %s", run.id, exc_info=True
        )
        return WatchActionOutcome(action=ACTION_REPLAN, error=str(exc)[:300])

    _pull_check_forward(db, watch)
    logger.info("[WatchActions] replanned mission %s for watch %s", run.id, watch.id)
    return WatchActionOutcome(
        action=ACTION_REPLAN, executed=True, detail=f"mission {run.id} replanning"
    )


async def _execute_reassign(
    db: Session,
    watch,
    run,
    *,
    diagnosis: Optional[str],
    preferred_agent_name: Optional[str] = None,
) -> WatchActionOutcome:
    """Requeue the failed task to a DIFFERENT capable agent."""
    from core.models.core import Agent
    from core.models.orchestration import (
        OrchestrationTask,
        OrchestrationTaskDependency,
    )
    from core.models.orchestration_enums import ActorType, RunState, TaskState
    from modules.coordination.agent_matcher import AgentMatcher
    from services.orchestration_board_bridge import (
        create_task_board_task,
        sync_board_status,
    )
    from services.orchestration_state import transition_run, transition_task

    failed_task = (
        db.query(OrchestrationTask)
        .filter(
            OrchestrationTask.run_id == run.id,
            OrchestrationTask.state == TaskState.FAILED.value,
        )
        .order_by(OrchestrationTask.sequence_number)
        .first()
    )
    if failed_task is None:
        return WatchActionOutcome(
            action=ACTION_REASSIGN, error="no failed task to reassign"
        )

    # --- choose the replacement agent (capability matching, prior excluded)
    input_context = failed_task.input_context if isinstance(failed_task.input_context, dict) else {}
    prior_agent_id = (input_context.get("agent_match") or {}).get("agent_id")

    if preferred_agent_name:
        chosen_name = preferred_agent_name  # spawn path pins its new agent
    else:
        agents = (
            db.query(Agent)
            .filter(Agent.workspace_id == run.workspace_id, Agent.status == "active")
            .all()
        )
        ranked = AgentMatcher.rank(
            db,
            failed_task,
            agents,
            task_spec={
                "agent_role": failed_task.agent_role,
                "required_tools": input_context.get("required_tools", []),
            },
        )
        candidates = [r for r in ranked if r.agent_id != prior_agent_id]
        if not candidates:
            return WatchActionOutcome(
                action=ACTION_REASSIGN, error="no_capable_agent"
            )
        chosen_name = candidates[0].agent_name

    # --- transition-legal requeue (narrow replan shape) ---
    try:
        transition_run(
            db, run=run, new_state=RunState.REPLANNING,
            actor_type=ActorType.COORDINATOR, actor_id="watcher",
            reason=f"Watch reassign of task {failed_task.id} -> {chosen_name}",
        )

        failed_task.failure_reason_code = "reassigned_by_watch"
        failed_task.failure_detail = (
            f"Reassigned by the watcher to '{chosen_name}'"
            + (f": {diagnosis}" if diagnosis else "")
        )
        transition_task(
            db, task=failed_task, new_state=TaskState.SKIPPED,
            actor_type=ActorType.COORDINATOR, actor_id="watcher",
            reason="Replaced by watch reassign",
        )
        sync_board_status(db, failed_task)

        clone = OrchestrationTask(
            run_id=run.id,
            title=failed_task.title,
            description=failed_task.description,
            task_type=failed_task.task_type,
            sequence_number=failed_task.sequence_number,
            agent_role=chosen_name,  # explicit-override pin (PRD-163 S4)
            state=TaskState.PENDING.value,
            state_type="initial",
            verification_criteria=failed_task.verification_criteria,
            input_context={
                **{k: v for k, v in input_context.items() if k != "agent_match"},
                "watch_reassign": {
                    "replaces_task_id": str(failed_task.id),
                    "excluded_agent_id": prior_agent_id,
                    "diagnosis": diagnosis,
                },
            },
            max_retries=failed_task.max_retries,
            complexity=failed_task.complexity,
            estimated_tokens=failed_task.estimated_tokens,
        )
        db.add(clone)
        db.flush()

        # Clone inherits the failed task's prerequisites; dependents repoint.
        for dep in (
            db.query(OrchestrationTaskDependency)
            .filter(OrchestrationTaskDependency.task_id == failed_task.id)
            .all()
        ):
            db.add(
                OrchestrationTaskDependency(
                    task_id=clone.id, depends_on_task_id=dep.depends_on_task_id
                )
            )
        db.query(OrchestrationTaskDependency).filter(
            OrchestrationTaskDependency.depends_on_task_id == failed_task.id
        ).update(
            {OrchestrationTaskDependency.depends_on_task_id: clone.id},
            synchronize_session=False,
        )

        transition_run(
            db, run=run, new_state=RunState.RUNNING,
            actor_type=ActorType.COORDINATOR, actor_id="watcher",
            reason="Watch reassign complete -- resuming",
        )
        transition_task(
            db, task=clone, new_state=TaskState.QUEUED,
            actor_type=ActorType.COORDINATOR, actor_id="watcher",
            reason=f"Queued for '{chosen_name}' after watch reassign",
        )
        try:
            create_task_board_task(db, run, clone)
        except Exception:
            logger.warning(
                "[WatchActions] board card for reassigned task failed (non-fatal)",
                exc_info=True,
            )
        sync_board_status(db, clone)
    except Exception as exc:  # noqa: BLE001 -- outcome-shaped, caller escalates
        logger.error(
            "[WatchActions] reassign failed for run %s", run.id, exc_info=True
        )
        return WatchActionOutcome(action=ACTION_REASSIGN, error=str(exc)[:300])

    _pull_check_forward(db, watch)
    logger.info(
        "[WatchActions] reassigned task %s -> %s (clone %s) for watch %s",
        failed_task.id,
        chosen_name,
        clone.id,
        watch.id,
    )
    return WatchActionOutcome(
        action=ACTION_REASSIGN,
        executed=True,
        detail=f"task requeued to '{chosen_name}'",
    )


async def _execute_spawn_agent(
    db: Session,
    watch,
    run,
    *,
    spawn_spec: Optional[Dict[str, Any]],
    diagnosis: Optional[str],
) -> WatchActionOutcome:
    """Instantiate an agent from an existing blueprint ONLY, then put it to
    work (reassign onto it; replan when there is no failed task)."""
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import TaskState
    from modules.tools.discovery.handlers_agents import create_agent
    from services.blueprint_validator import (
        get_blueprint_by_id,
        get_default_blueprint,
        validate_agent,
    )

    spec = dict(spawn_spec or {})

    # Blueprints only -- no free-form spawning (Section 8 Q5).
    blueprint = None
    blueprint_id = spec.get("blueprint_id")
    if blueprint_id:
        try:
            from uuid import UUID as _UUID

            blueprint = get_blueprint_by_id(db, run.workspace_id, _UUID(str(blueprint_id)))
        except (ValueError, TypeError):
            blueprint = None
    else:
        blueprint = get_default_blueprint(db, run.workspace_id)
    if blueprint is None:
        return WatchActionOutcome(
            action=ACTION_SPAWN_AGENT,
            error="no blueprint available (spawn is blueprints-only)",
        )

    failed_task = (
        db.query(OrchestrationTask)
        .filter(
            OrchestrationTask.run_id == run.id,
            OrchestrationTask.state == TaskState.FAILED.value,
        )
        .order_by(OrchestrationTask.sequence_number)
        .first()
    )
    role = (failed_task.agent_role if failed_task is not None else None) or "specialist"
    name = spec.get("name") or f"{role}-watch-{uuid4().hex[:6]}"

    params: Dict[str, Any] = {
        "name": name,
        "agent_type": spec.get("agent_type", "chatbot"),
        "description": spec.get("description")
        or f"Spawned by the watcher from blueprint '{blueprint.name}' to recover mission {run.id}",
        "tags": ["watch-spawned", f"blueprint:{blueprint.id}"],
    }
    if spec.get("system_prompt"):
        params["system_prompt"] = spec["system_prompt"]
    if spec.get("model_id"):
        params["model_id"] = spec["model_id"]

    result = await create_agent(db, run.workspace_id, params)
    if not result.get("success"):
        return WatchActionOutcome(
            action=ACTION_SPAWN_AGENT,
            error=f"agent creation failed: {result.get('error')}"[:300],
        )
    agent_id = result["agent"]["id"]

    # Blueprint rules validation stays authoritative. Defaulted/empty rules
    # pass cleanly (onboarding-wall guard); strict-mode failures roll the
    # spawn back.
    validation = validate_agent(db, run.workspace_id, agent_id, blueprint_id=blueprint.id)
    if validation.get("enforce_mode") == "strict" and not validation.get("pass"):
        from core.models.core import Agent

        db.query(Agent).filter(Agent.id == agent_id).delete()
        db.flush()
        return WatchActionOutcome(
            action=ACTION_SPAWN_AGENT,
            error=(
                "blueprint validation failed: "
                + "; ".join(validation.get("failures", []))
            )[:300],
        )

    # Put the new agent to work.
    if failed_task is not None:
        follow_up = await _execute_reassign(
            db, watch, run, diagnosis=diagnosis, preferred_agent_name=name
        )
    else:
        follow_up = await _execute_replan(
            db, watch, run,
            diagnosis=diagnosis,
            notes=f"A new agent '{name}' (id {agent_id}) was spawned to help; assign it the failed work.",
        )
    if follow_up.error:
        return WatchActionOutcome(
            action=ACTION_SPAWN_AGENT,
            error=f"agent {agent_id} spawned but follow-up {follow_up.action} failed: {follow_up.error}"[:300],
        )

    detail = f"spawned agent '{name}' (id {agent_id}) and {follow_up.detail}"
    warnings = validation.get("warnings") or []
    if warnings:
        detail += f" (blueprint warnings: {'; '.join(warnings)[:200]})"
    logger.info("[WatchActions] %s for watch %s", detail, watch.id)
    return WatchActionOutcome(action=ACTION_SPAWN_AGENT, executed=True, detail=detail)


# ---------------------------------------------------------------------------
# Escalate (the ungated escape hatch)
# ---------------------------------------------------------------------------


async def escalate_watch_now(db: Session, watch, *, reason: str) -> None:
    """Board card + watch_escalation notification + watch -> escalated."""
    from core.models.watch_enums import WatchEventType, WatchStatus
    from services.escalation_service import escalate_watch
    from services.watch_notifications import dispatch_watch_notification
    from services.watch_service import WatchService

    try:
        card = escalate_watch(db, watch.workspace_id, watch, reason)
    except Exception:
        logger.error(
            "[WatchActions] escalation card failed for watch %s",
            getattr(watch, "id", "?"),
            exc_info=True,
        )
        card = None

    WatchService.ingest(
        db,
        watch,
        event_type=WatchEventType.STATUS_CHANGE.value,
        event_key=f"escalated:{watch.target_type}:{watch.target_id}:{watch.actions_taken or 0}",
        summary=f"Escalated to a human: {reason}",
        snapshot={"board_task_id": getattr(card, "id", None)},
        requires_attention=True,
    )
    if WatchStatus(watch.status) != WatchStatus.ESCALATED:
        WatchService.transition(db, watch, WatchStatus.ESCALATED, reason=reason)

    await dispatch_watch_notification(
        db,
        watch,
        event_type="watch_escalation",
        title=f"Watch escalated: {(watch.title or '')[:100]}",
        message=reason,
        status="warning",
    )


# ---------------------------------------------------------------------------
# Grant parking + resume executors (the S7 playbook_run flow, more actions)
# ---------------------------------------------------------------------------


def _park_action_grant(
    db: Session,
    watch,
    run,
    action: str,
    *,
    decision,
    diagnosis: Optional[str],
    notes: Optional[str],
    spawn_spec: Optional[Dict[str, Any]],
):
    """PENDING SUBJECT_PLAYBOOK_RUN grant carrying the mission-action spec.

    One pending corrective action per run at a time (idempotency via
    find_pending_grant -- a second ask reuses the pending grant).
    """
    from core.models.approval_grants import SUBJECT_PLAYBOOK_RUN
    from core.models.watch_enums import WatchEventType, WatchStatus
    from core.services.approval_grants import create_grant, find_pending_grant
    from services.watch_service import WatchService

    existing = find_pending_grant(
        db,
        watch.workspace_id,
        subject_type=SUBJECT_PLAYBOOK_RUN,
        subject_id=str(run.id),
    )
    grant = existing
    if grant is None:
        grant = create_grant(
            db,
            watch.workspace_id,
            subject_type=SUBJECT_PLAYBOOK_RUN,
            subject_id=str(run.id),
            tool_name=f"watch_{action}",
            reason=decision.reason,
            estimated_cost_usd=decision.estimated_cost,
        )
        grant.details = {
            "watch_action": action,
            "watch_id": str(watch.id),
            "mission_id": str(run.id),
            "spec": {
                "diagnosis": diagnosis,
                "notes": notes,
                "spawn_spec": spawn_spec,
            },
        }
        db.flush()

    WatchService.ingest(
        db,
        watch,
        event_type=WatchEventType.STATUS_CHANGE.value,
        event_key=f"grant-pending:{grant.id}",
        summary=f"Corrective action '{action}' parked on approval grant {grant.id}",
        snapshot={"grant_id": grant.id, "watch_action": action},
    )
    if WatchStatus(watch.status) != WatchStatus.AWAITING_APPROVAL:
        WatchService.transition(
            db, watch, WatchStatus.AWAITING_APPROVAL,
            reason=f"awaiting grant {grant.id}",
        )
    return grant


async def _notify_action_approval_pending(db: Session, watch, grant, action: str) -> None:
    from services.watch_notifications import dispatch_watch_notification

    await dispatch_watch_notification(
        db,
        watch,
        event_type="approval_pending",
        title=f"Approval needed: {action} for '{(watch.title or '')[:80]}'",
        message=(
            f"The watcher wants to run '{action}' on the watched mission "
            f"(grant {grant.id}, estimated ${float(grant.estimated_cost_usd or 0):.2f}). "
            "Review it in the approvals inbox."
        ),
        status="warning",
    )


def _resolve_run(db: Session, watch):
    from core.models.orchestration import OrchestrationRun

    try:
        return (
            db.query(OrchestrationRun)
            .filter(
                OrchestrationRun.id == watch.target_id,
                OrchestrationRun.workspace_id == watch.workspace_id,
            )
            .first()
        )
    except Exception:
        logger.warning(
            "[WatchActions] run resolve failed for %s", watch.target_id, exc_info=True
        )
        return None


def _pull_check_forward(db: Session, watch) -> None:
    """The corrective attempt is in flight -- recheck promptly."""
    from datetime import datetime, timezone

    try:
        watch.next_check_at = datetime.now(timezone.utc)
        db.flush()
    except Exception:
        logger.debug("[WatchActions] next_check pull failed", exc_info=True)


# ---------------------------------------------------------------------------
# Grant-resume executors (registered into the S7 registry on import)
# ---------------------------------------------------------------------------


async def _grant_execute_mission_action(db: Session, grant) -> Dict[str, Any]:
    """Execute a granted mission action from its stored spec. The grant is
    already the approval -- no second gate, no second budget charge."""
    from services.watch_rerun import _load_watch, _resume_watch_to_watching

    details = dict(grant.details) if isinstance(grant.details, dict) else {}
    action = details.get("watch_action")
    spec = details.get("spec") or {}
    watch = _load_watch(db, grant)
    if watch is None:
        return {"success": False, "error": "watch no longer exists"}

    run = _resolve_run(db, watch)
    if run is None:
        return {"success": False, "error": "target mission no longer exists"}

    _resume_watch_to_watching(db, watch)
    outcome = await _execute_action(
        db,
        watch,
        run,
        action,
        diagnosis=spec.get("diagnosis"),
        notes=spec.get("notes"),
        spawn_spec=spec.get("spawn_spec"),
    )
    if outcome.error:
        await escalate_watch_now(
            db, watch, reason=f"granted '{action}' failed: {outcome.error}"
        )
        return {"success": False, "error": outcome.error}
    return {"success": True, "detail": outcome.detail}


def _register_grant_executors() -> None:
    from services.watch_rerun import WATCH_GRANT_EXECUTORS

    for action in (ACTION_REPLAN, ACTION_REASSIGN, ACTION_SPAWN_AGENT):
        WATCH_GRANT_EXECUTORS.setdefault(action, _grant_execute_mission_action)


_register_grant_executors()
