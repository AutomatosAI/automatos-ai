"""
Coordinator Service — PRD-82A Sequential Mission Coordinator
=============================================================

Main orchestration service: 5s tick loop, mission lifecycle methods,
and the glue between planner, dispatcher, reconciler, and verifier.

Key patterns:
- DB-authoritative, stateless coordinator — every tick reads from DB
- SessionLocal per tick (no stored DB session on singleton)
- Parallel dispatch via MissionDispatcher.dispatch_ready() + asyncio.gather()
- Output summary built when all tasks verified (Section 6)
- Soft budget tracking with warning events (Section 9)

Source: PRD-82A Sections 6, 8, 9, 12 (US-014)
        PRD-102 Section 3.3 (coordinator design)
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from config import COMPLEXITY_TOKEN_BUDGET, Config, config
from core.database.database import end_open_transaction
from core.models.core import Agent
from core.models.system_settings import SystemSetting
from core.models.orchestration import (
    OrchestrationArchive,
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
    OrchestrationTaskDependency,
)
from core.models.orchestration_enums import (
    ActorType,
    EventType,
    FailureReasonCode,
    RunState,
    StopReason,
    TaskState,
    TaskType,
    TERMINAL_RUN_STATES,
    DONE_TASK_STATES,
)
from modules.coordination import progress_ledger
from modules.coordination.agent_matcher import AgentMatcher, build_match_annotation
from modules.coordination.dispatcher import MissionDispatcher
from modules.coordination.planner import (
    DecompositionResult,
    MissionPlanner,
    PlanValidationError,
)
from modules.coordination.primitive_heartbeat import _emit_missions_primitive
from modules.coordination.reconciler import MissionReconciler
from modules.coordination.verification import ConsistencyResult, VerificationService
from services.orchestration_board_bridge import (
    create_mission_board_task,
    create_task_board_task,
    sync_board_status,
)
from services.orchestration_deps import DependencyResolver
from services.orchestration_state import (
    ConflictError,
    InvalidTransitionError,
    emit_event,
    transition_run,
    transition_task,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mission Power Modes — per-mode caps for the LLM tier and tool iterations.
# Token budget is deliberately NOT a power-mode concern: it is resolved from
# the agent's own Max Output Tokens setting (see AgentFactory, falling back to
# the model's registry ceiling, then DEFAULT_MAX_OUTPUT_TOKENS).
# "standard" is the default when power_mode is absent from mission config.
# These are the hardcoded FALLBACK only. Operators retune live values via
# system_settings (category 'power_modes', key '<mode>'); see
# _get_power_mode_caps(). Stored settings win; absent keys fall back here.
# ---------------------------------------------------------------------------
# PRD-163 S5: timeout scales with power mode — light work shouldn't hang for the
# full window, and max-power deep work needs more than the 4-min default.
_POWER_MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "light":    {"max_tool_iterations": 5,  "force_llm_tier": "system_llm", "timeout_seconds": 120},
    "standard": {"max_tool_iterations": 10, "force_llm_tier": None, "timeout_seconds": 240},
    "max":      {"max_tool_iterations": 50, "force_llm_tier": "orchestrator_llm", "timeout_seconds": 600},
}


# PRD-163 S4/Q57: approval-time plan editing — only these per-task fields are
# editable from the approval card. Structural changes (add/remove tasks, rewire
# dependencies) go through plan-import (Q54), not this field-PATCH path.
_EDITABLE_TASK_FIELDS = ("agent_role", "title", "description")


# ---------------------------------------------------------------------------
# PRD-164 S4 (Q22): dispatch context goes through the field digest.
# One value cap shared by field injection and the upstream digest rows, so a
# row read straight from the DB is byte-identical to what the field would
# echo back for the same task output (~1000 tokens at 4 chars/token).
# ---------------------------------------------------------------------------
FIELD_VALUE_CAP_CHARS = 4000

_BASE64_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")


def _sanitize_for_field(raw: str) -> str:
    """Strip inline base64 image blobs before a task output enters dispatch
    context (field injection or upstream digest rows) — a single data-URI
    would otherwise burn the whole row cap on undecodable noise."""
    return _BASE64_IMAGE_RE.sub("[image — see generated-images API]", raw or "")


def annotate_plan_with_matches(plan: Optional[Dict[str, Any]],
                               match_by_seq: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """PRD-164 S2: mirror per-task agent-match previews into the ``run.plan``
    snapshot (keyed by ``sequence_number``) so the approval card — which is
    built from the plan tasks — can show WHO would run each task and WHY.

    Pure and immutable: returns a NEW plan dict (the JSON column needs the
    reassignment to track the change); the input plan is never mutated.
    """
    plan = plan or {}
    new_tasks: List[Dict[str, Any]] = []
    for pt in (plan.get("tasks") or []):
        seq = pt.get("sequence_number")
        match = match_by_seq.get(int(seq)) if seq is not None else None
        if match:
            pt = {
                **pt,
                "match_agent": match.get("agent_name"),
                "match_agent_id": match.get("agent_id"),
                "match_reason": match.get("reason"),
                "match_is_override": bool(match.get("is_override", False)),
            }
        new_tasks.append(pt)
    return {**plan, "tasks": new_tasks}


def apply_plan_task_edits(tasks: List[Any], plan: Optional[Dict[str, Any]],
                          edits: List[Dict[str, Any]]) -> tuple:
    """Apply per-task field edits to OrchestrationTask rows and mirror them into
    the ``run.plan`` snapshot. Pure w.r.t. its arguments (no DB) so it is unit
    testable: the caller supplies the loaded task rows + plan dict.

    Each edit matches a task by (in priority order) ``task_id`` (str of the row
    id), ``temp_id`` (resolved to a sequence via the plan snapshot), or
    ``sequence_number``. Only ``_EDITABLE_TASK_FIELDS`` are honoured. Task rows
    are mutated in place (the ORM idiom); a NEW plan dict is returned so the JSON
    column's change is tracked. Returns ``(new_plan, fields_changed)``.
    """
    by_id = {str(getattr(t, "id", "")): t for t in tasks}
    by_seq = {int(getattr(t, "sequence_number", -1)): t for t in tasks}

    plan = plan or {}
    plan_tasks = list(plan.get("tasks") or [])
    # temp_id -> sequence_number, so a temp_id edit can find the row by sequence.
    temp_to_seq = {
        str(pt.get("temp_id")): pt.get("sequence_number")
        for pt in plan_tasks if pt.get("temp_id") is not None
    }

    fields_changed = 0
    edited_rows = set()
    for edit in (edits or []):
        if not isinstance(edit, dict):
            continue
        task = None
        if edit.get("task_id") is not None:
            task = by_id.get(str(edit["task_id"]))
        if task is None and edit.get("temp_id") is not None:
            seq = temp_to_seq.get(str(edit["temp_id"]))
            if seq is not None:
                task = by_seq.get(int(seq))
        if task is None and edit.get("sequence_number") is not None:
            try:
                task = by_seq.get(int(edit["sequence_number"]))
            except (ValueError, TypeError):
                task = None
        if task is None:
            continue
        for field in _EDITABLE_TASK_FIELDS:
            if field in edit and edit[field] is not None:
                new_val = edit[field]
                if getattr(task, field, None) != new_val:
                    setattr(task, field, new_val)
                    fields_changed += 1
        edited_rows.add(int(getattr(task, "sequence_number", -1)))

    # Mirror the row state back into the plan snapshot (by sequence_number).
    if edited_rows:
        seq_to_row = {int(getattr(t, "sequence_number", -1)): t for t in tasks}
        new_plan_tasks = []
        for pt in plan_tasks:
            seq = pt.get("sequence_number")
            row = seq_to_row.get(int(seq)) if seq is not None else None
            if row is not None and int(seq) in edited_rows:
                pt = {**pt, **{f: getattr(row, f, pt.get(f)) for f in _EDITABLE_TASK_FIELDS}}
            new_plan_tasks.append(pt)
        plan = {**plan, "tasks": new_plan_tasks}

    return plan, fields_changed


def _get_power_mode_caps(power_mode: str, db: Session) -> Dict[str, Any]:
    """Resolve power-mode caps: ``system_settings('power_modes', <mode>)`` merged
    over ``_POWER_MODE_DEFAULTS``.

    Operators can retune caps at runtime (no deploy) by storing a JSON object
    under category ``power_modes``, key ``<mode>`` — e.g.
    ``{"max_tool_iterations": 20}``. Stored keys override the defaults; absent
    keys fall back. An unknown mode falls back to ``standard``.

    Must run on the serial DB path (e.g. ``_prepare_task``). Do NOT call from
    ``_run_agent_io`` — that runs concurrently via ``asyncio.gather`` with no DB
    access; pass the already-resolved caps down instead.
    """
    defaults = _POWER_MODE_DEFAULTS.get(power_mode, _POWER_MODE_DEFAULTS["standard"])
    caps: Dict[str, Any] = dict(defaults)  # copy — never mutate the module default

    try:
        setting = (
            db.query(SystemSetting)
            .filter(
                SystemSetting.category == "power_modes",
                SystemSetting.key == power_mode,
            )
            .first()
        )
        if setting and setting.value:
            override = json.loads(setting.value)
            if isinstance(override, dict):
                caps.update(override)
    except Exception:
        logger.warning(
            "Could not load power_mode caps for '%s' from system_settings; "
            "using hardcoded defaults.",
            power_mode,
            exc_info=True,
        )

    return caps


def _workspace_power_mode_default(workspace_id, db: Session) -> Optional[str]:
    """The workspace's default power mode (``workspace.settings['power_mode']``),
    set via ``platform_set_power_mode`` or a HARNESS ``power_mode_*`` prescription
    (PRD-142 Wave 4, W4-S5). Returns the stored mode when it's a known tier, else
    None so the caller falls back to ``'standard'``.

    Best-effort by design: this runs on the per-task dispatch path, so a lookup
    failure (or an unknown stored value) degrades silently to the default rather
    than failing the task. One indexed workspace read; fine at pilot scale.
    """
    try:
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        mode = (ws.settings or {}).get("power_mode") if ws else None
        return mode if mode in _POWER_MODE_DEFAULTS else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Synthesis model override (Fix 1)
# ---------------------------------------------------------------------------
# PRD-128: Unified notification dispatch helpers
# ---------------------------------------------------------------------------

async def _dispatch_mission_event(
    db: Session,
    run: OrchestrationRun,
    event_type: str,
    title: str,
    message: Optional[str],
    agent_id: Optional[int] = None,
    agent_name: Optional[str] = None,
    status: str = "ok",
) -> None:
    """Fire a mission-related event through NotificationDispatcher.

    Uses the current coordinator DB session so the notification row joins
    the outer tick transaction (the tick commits once per run). Failures
    are logged but never block the coordinator.

    Resolves ``run.created_by`` (Clerk user ID) to an integer ``user_id``
    so notifications target the mission creator — not the entire workspace.
    """
    try:
        from core.models.core import User
        from core.services.notification_dispatcher import NotificationDispatcher

        # Resolve Clerk ID → integer user_id so notifications are
        # scoped to the mission creator, not broadcast workspace-wide.
        user_id: Optional[int] = None
        if run.created_by:
            user_row = (
                db.query(User.id)
                .filter(User.clerk_user_id == run.created_by)
                .first()
            )
            if user_row:
                user_id = user_row[0]

        dispatcher = NotificationDispatcher(db, str(run.workspace_id))
        await dispatcher.dispatch(
            event_type=event_type,
            title=title,
            message=message,
            link_type="mission",
            link_id=str(run.id),
            agent_id=agent_id,
            agent_name=agent_name,
            status=status,
            user_id=user_id,
        )
    except Exception:
        logger.error(
            "[Coordinator] %s dispatch failed for run %s",
            event_type,
            getattr(run, "id", "?"),
            exc_info=True,
        )


async def notify_mission_failed(db: Session, run: OrchestrationRun) -> None:
    """PRD-204 S4: dispatch ``mission_failed`` at the run-failure boundary.

    ONE owner of the failure message shape, called from every async
    failure site (reconciler task-failure cascades, joiner halt, plan
    validation). Reuses the ``_dispatch_mission_event`` seam (resolves
    ``run.created_by`` to a user_id; never raises). Previously a failed
    run emitted only an internal RUN_FAILED audit event -- the user was
    never told.
    """
    try:
        detail = (
            getattr(run, "stop_detail", None)
            or getattr(run, "stop_reason", None)
            or "Mission failed"
        )
        goal = getattr(run, "goal", None) or "Mission"
        await _dispatch_mission_event(
            db=db,
            run=run,
            event_type="mission_failed",
            title=f"Mission failed: {goal[:110]}",
            message=str(detail)[:500],
            status="error",
        )
    except Exception:
        # A notification must never break the failure path it reports on.
        logger.error(
            "[Coordinator] mission_failed notify failed for run %s",
            getattr(run, "id", "?"),
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# PRD-227 US-002 — mission lifecycle narration into the launching chat thread
# ---------------------------------------------------------------------------
#
# A mission launched from a chat narrates its lifecycle back INTO that thread
# (approved/started → each task done/failed → completed/failed/cancelled),
# reusing the PRD-205 background→chat seam (``deliver_background_message``) —
# NEVER a parallel send path. Target = the run's originating chat (captured on
# ``run.config['origin_chat_id']`` at create time, the same server-injected
# origin watches use), else the creator's per-(workspace,user) Auto thread
# (the messenger's ``find_or_create_auto_chat`` fallback, reached by passing
# ``clerk_user_id=run.created_by`` with no chat_id). Task-level lines are
# throttled: suppressed for runs with more than ``MISSION_NARRATION_TASK_CAP``
# tasks; run-level lines always send. Source label ``"Auto · mission"`` +
# ``link_type="mission"`` so the bell/badge deep-link the run. Distinct from
# PRD-224 watch verdicts (judged outcomes): these are lifecycle events, worded
# so the two never read alike. All sends are fail-soft.

_MISSION_NARRATION_LABEL = "Auto · mission"


def _mission_task_count(run: OrchestrationRun) -> int:
    """Planned task count for a run, read from ``run.plan['tasks']``."""
    plan = getattr(run, "plan", None) or {}
    if isinstance(plan, dict):
        return len(plan.get("tasks") or [])
    return 0


def _narrate_mission(
    db: Session,
    run: OrchestrationRun,
    text: str,
    *,
    level: str,
    event: str,
) -> None:
    """Post one mission-lifecycle line into the launching chat (or Auto thread).

    ``level='task'`` lines are suppressed when the run has more than
    ``Config.MISSION_NARRATION_TASK_CAP`` tasks; ``level='run'`` lines always
    send. Routed through ``deliver_background_message`` (the only PRD-205
    producer seam) and wrapped fail-soft, so a chat failure never breaks the
    coordinator tick or lifecycle transition that produced the event.

    The delivery runs on an INDEPENDENT short-lived session, never the
    coordinator's shared ``db`` (PRD-227 P227-RVW-1) — see below.
    """
    try:
        if level == "task" and _mission_task_count(run) > Config.MISSION_NARRATION_TASK_CAP:
            return
        cfg = getattr(run, "config", None) or {}
        origin_chat = cfg.get("origin_chat_id") if isinstance(cfg, dict) else None

        from core.database.database import SessionLocal
        from services.chat_messenger import deliver_background_message

        # PRD-227 P227-RVW-1: narration must NEVER commit or roll back the
        # coordinator's SHARED session. ChatService.save_message hard-commits,
        # and deliver_background_message rolls back on failure; on the
        # coordinator's own session — this fires MID-transaction, before the
        # caller commits (approve_plan / _record_task_result / cancel_mission /
        # the tick terminal observer / approval-expiry) — that would either
        # commit half-built mission state early or, on a transient chat-write
        # failure, roll back the caller's uncommitted transition (RUNNING +
        # queued tasks) and silently strand the mission. An independent
        # short-lived session (the isolation the tick already uses for side
        # effects) confines the commit AND any rollback to the message insert
        # alone — the coordinator transaction is untouched on both paths. This
        # matches US-001's notify_board_event, which never commits the caller's
        # session either.
        narration_db = SessionLocal()
        try:
            deliver_background_message(
                narration_db,
                workspace_id=run.workspace_id,
                text=text,
                source={"origin": "mission", "label": _MISSION_NARRATION_LABEL, "event": event},
                chat_id=str(origin_chat) if origin_chat else None,
                clerk_user_id=getattr(run, "created_by", None),
                link_type="mission",
                link_id=str(run.id),
            )
        finally:
            narration_db.close()
    except Exception:  # noqa: BLE001 — narration is best-effort, never fatal
        logger.error(
            "[Coordinator] mission narration (%s) failed for run %s",
            event, getattr(run, "id", "?"), exc_info=True,
        )


def _narrate_run_terminal(db: Session, run: OrchestrationRun) -> None:
    """Narrate a run reaching a terminal state. Run-level → never throttled.

    Reads ``run.state``; a non-terminal state produces nothing. Wording is
    lifecycle-framed so it stays distinguishable from a PRD-224 watch verdict.
    """
    state = getattr(run, "state", None)
    goal = getattr(run, "goal", None) or "Mission"
    if state == RunState.COMPLETED.value:
        text = f"Mission complete: {goal[:150]}"
    elif state == RunState.FAILED.value:
        detail = getattr(run, "stop_detail", None) or getattr(run, "stop_reason", None) or "failed"
        text = f"Mission failed: {str(detail)[:200]}"
    elif state == RunState.CANCELLED.value:
        text = f"Mission cancelled: {goal[:150]}"
    else:
        return
    _narrate_mission(db, run, text, level="run", event=f"run_{state}")


async def notify_mission_budget_paused(db: Session, run: OrchestrationRun) -> None:
    """PRD-204 S4: dispatch ``mission_budget_paused`` at the budget-pause
    transition (the dispatcher blocked and moved the run to PAUSED --
    previously silent). Single owner of this event; the dead
    ``escalation_service.notify_budget_exceeded`` board-card path was
    removed in the same story.
    """
    try:
        spent = getattr(run, "budget_spent", None) or {}
        budget_cfg = getattr(run, "budget_config", None) or {}
        token_budget = getattr(run, "token_budget_estimate", None)
        parts = []
        if spent.get("cost") is not None:
            parts.append(f"spent ${float(spent.get('cost') or 0):.2f}")
        if budget_cfg.get("max_cost") is not None:
            parts.append(f"ceiling ${float(budget_cfg['max_cost']):.2f}")
        if token_budget:
            parts.append(
                f"tokens {getattr(run, 'tokens_used', 0) or 0}/{token_budget}"
            )
        detail = "; ".join(parts) or "budget exceeded"
        goal = getattr(run, "goal", None) or "Mission"
        await _dispatch_mission_event(
            db=db,
            run=run,
            event_type="mission_budget_paused",
            title=f"Mission paused on budget: {goal[:100]}",
            message=(
                f"Budget exceeded ({detail}). Increase the budget and resume "
                f"to continue."
            ),
            status="warning",
        )
    except Exception:
        # A notification must never break the pause path it reports on.
        logger.error(
            "[Coordinator] mission_budget_paused notify failed for run %s",
            getattr(run, "id", "?"),
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Mission cost audit
# ---------------------------------------------------------------------------

_cost_logger = logging.getLogger("llm.cost_audit")


def _log_mission_cost_summary(db: Session, run) -> float:
    """Query llm_usage for this mission run and log a cost summary.

    Returns estimated total cost in USD (0.0 if no data).
    """
    try:
        from sqlalchemy import func as sa_func
        from core.models.core import LLMUsage

        rows = (
            db.query(
                LLMUsage.model_id,
                sa_func.count().label("calls"),
                sa_func.sum(LLMUsage.input_tokens).label("input_tokens"),
                sa_func.sum(LLMUsage.output_tokens).label("output_tokens"),
                sa_func.sum(LLMUsage.total_cost).label("total_cost"),
            )
            .filter(LLMUsage.execution_id == str(run.id))
            .group_by(LLMUsage.model_id)
            .all()
        )

        total_cost = 0.0
        total_tokens = 0
        lines = []
        for row in rows:
            cost = float(row.total_cost or 0)
            tokens = int(row.input_tokens or 0) + int(row.output_tokens or 0)
            total_cost += cost
            total_tokens += tokens
            lines.append(
                f"  {row.model_id}: {row.calls} calls, "
                f"{row.input_tokens}in/{row.output_tokens}out tokens, "
                f"${cost:.4f}"
            )

        summary = "\n".join(lines) if lines else "  (no usage data recorded)"
        _cost_logger.info(
            "MISSION_COST_SUMMARY run_id=%s total_cost_usd=%.4f "
            "total_tokens=%d\n%s",
            run.id, total_cost, total_tokens, summary,
        )

        # Budget alert
        try:
            from core.llm.manager import get_system_setting
            budget = float(get_system_setting("llm_cost_audit", "mission_budget_alert_usd", "2.00"))
        except Exception:
            budget = 2.00

        if total_cost > budget:
            _cost_logger.warning(
                "MISSION_BUDGET_EXCEEDED run_id=%s cost=%.4f budget=%.2f — "
                "review model selection in Settings > Coordination",
                run.id, total_cost, budget,
            )

        return total_cost
    except Exception:
        logger.debug("Mission cost summary failed", exc_info=True)
        return 0.0


async def _store_mission_memory_safe(
    db: Session,
    run_id,
    outcome: str,
    failure_reason: Optional[str] = None,
) -> None:
    """PRD-131d Phase 1: persist mission summary to L2+L3 memory.

    Wrapped in a try/except so memory failures never break a mission transition.
    """
    try:
        from core.services.mission_memory_service import MissionMemoryService
        await MissionMemoryService(db=db).store_mission_summary(
            run_id=run_id,
            outcome=outcome,
            failure_reason=failure_reason,
        )
    except Exception:
        logger.warning(
            "Mission memory storage skipped for run %s (outcome=%s)",
            run_id, outcome, exc_info=True,
        )


# ---------------------------------------------------------------------------
# CoordinatorService
# ---------------------------------------------------------------------------


def pick_final_output_task(tasks):
    """Last task in sequence order with a non-empty output — the mission's
    final work product (e.g. a 'Finalize …' writer task). Ties within a
    parallel group resolve to the later task in original order (sorted() is
    stable). Returns None when no task produced output."""
    final = None
    for t in sorted(tasks or [], key=lambda t: (t.sequence_number or 0)):
        if t.output and str(t.output).strip():
            final = t
    return final


class CoordinatorService:
    """
    Stateless coordinator that orchestrates sequential missions.

    - ``tick()`` runs every 5s: dispatches next tasks + reconciles active runs.
    - Lifecycle methods: create, approve, reject, review, pause, resume, cancel.
    - PRD-108: missions get a shared vector field (Qdrant) for inter-agent context.
    - No stored DB session — each method/tick acquires its own via SessionLocal.
    """

    def __init__(self):
        self._tick_running: bool = False
        self._scheduler = None
        self._owns_scheduler: bool = False
        self._last_archive_at: Optional[datetime] = None
        self._last_field_compaction_at: Optional[datetime] = None  # PRD-166 S1
        self._field = None  # Lazy-init via factory

    def _get_field(self):
        """Lazy-init the PRD-108 shared context backend (vector_field or redis)."""
        if self._field is None:
            try:
                from modules.context.factory import get_shared_context
                self._field = get_shared_context()
                if self._field:
                    logger.info("[PRD-108] Shared context backend initialized: %s", self._field._backend_name)
            except Exception as e:
                logger.warning("[PRD-108] Shared context unavailable: %s", e)
        return self._field

    async def _create_mission_field(
        self,
        db: Session,
        run: OrchestrationRun,
    ) -> Optional[str]:
        """Create a shared vector field for a mission. Returns field_id or None."""
        field = self._get_field()
        if not field:
            return None

        try:
            # Get agent IDs from the roster
            agents = (
                db.query(Agent)
                .filter(
                    and_(
                        Agent.workspace_id == run.workspace_id,
                        Agent.status == "active",
                    )
                )
                .all()
            )
            team_ids = [a.id for a in agents]

            # Seed the field with the mission goal. PRD-166 S1: carry provenance
            # so patterns keep workspace/mission lineage into the workspace field.
            field_id = await field.create_context(
                team_agent_ids=team_ids,
                initial_data={"mission_goal": run.goal},
                provenance={"workspace_id": str(run.workspace_id), "mission_id": str(run.id)},
            )

            # Store field_id in run config (no migration needed — JSONB)
            run.config = {**(run.config or {}), "field_id": field_id}
            db.flush()

            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.TASK_CREATED,  # Closest match
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                payload={"field_id": field_id, "event": "field.created"},
            )

            logger.info("[PRD-108] Created field %s for mission %s", field_id, run.id)
            return field_id
        except Exception as e:
            # Use repr(e) — str(e) can be empty for some HTTP errors (e.g. OpenRouter 402)
            status_code = getattr(e, "status_code", None)
            detail = f"{type(e).__name__}: {e!r}"
            if status_code:
                detail = f"HTTP {status_code} — {detail}"
            logger.warning(
                "[PRD-108] Failed to create field for mission %s: %s",
                run.id, detail, exc_info=True,
            )
            return None

    async def _inject_task_output_into_field(
        self,
        run: OrchestrationRun,
        task: OrchestrationTask,
        agent_id: int,
    ) -> None:
        """After a task completes, inject its output into the shared field."""
        field = self._get_field()
        field_id = (run.config or {}).get("field_id")
        if not field or not field_id or not task.output:
            return
        try:
            await field.inject(
                context_id=field_id,
                key=task.title or f"task_{task.sequence_number}",
                # Cap to prevent embedding blow-up; sanitized so a base64 blob
                # can't burn the cap (same treatment as upstream digest rows).
                value=_sanitize_for_field(str(task.output))[:FIELD_VALUE_CAP_CHARS],
                agent_id=agent_id,
                strength=1.0,
                provenance={
                    "workspace_id": str(run.workspace_id),
                    "mission_id": str(run.id),
                    "task_id": str(task.id),
                },
            )
            logger.info(
                "[PRD-108] Injected output from task %s into field %s",
                task.id, field_id,
            )
        except Exception as e:
            logger.warning("[PRD-108] Failed to inject task output: %r", e, exc_info=True)

    async def _attach_field_digest(
        self,
        db: Session,
        run: OrchestrationRun,
        task: OrchestrationTask,
        field_id: Optional[str],
        agent_id: int,
        upstream_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """PRD-166 S3 + PRD-164 S4 (Q22): pin THE budgeted dispatch digest into
        a task's prompt — the single channel for upstream/accumulated context.

        Immediate upstream-dependency outputs (``upstream_rows``, collected by
        ``_collect_upstream_digest_rows``) are merged AHEAD of semantic field
        hits, deduped against the field's echo of the same outputs, and the
        whole block is trimmed to ``Config.FIELD_QUERY_TOKEN_BUDGET`` — the
        per-task budget that replaced the 8K-chars-per-upstream stuffing.
        Anything the budget drops stays reachable via ``platform_field_query``.

        When the mission is tight on budget (CRITICAL/EXCEEDED) the digest is
        DROPPED to save tokens and a ``RUN_FIELD_CONTEXT_DROPPED`` warning is
        emitted — the budget-gate checkpoint for dispatch context.
        """
        from modules.context import field_scoring

        upstream_rows = upstream_rows or []

        # Budget-gate: drop the digest rather than spend tokens we don't have.
        try:
            from modules.coordination.dispatcher import BudgetStatus, MissionDispatcher
            status = MissionDispatcher._get_budget_status(run)
        except Exception:
            status, BudgetStatus = None, None
        if status is not None and status in (BudgetStatus.CRITICAL, BudgetStatus.EXCEEDED):
            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.RUN_FIELD_CONTEXT_DROPPED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                task_id=task.id,
                payload={"reason": "budget", "budget_status": status.value},
            )
            logger.info("[Field] Digest dropped for budget on task %s (%s)", task.id, status.value)
            return

        # Semantic field hits — fail-open to upstream rows only, so a missing
        # or unhealthy Qdrant degrades dispatch context, never breaks dispatch.
        field_rows: List[Dict[str, Any]] = []
        field = self._get_field()
        if field and field_id:
            try:
                query = f"{task.title or ''}\n{task.description or ''}".strip() or (run.goal or "")
                results = await field.query(
                    context_id=field_id,
                    query=query,
                    agent_id=agent_id,
                    top_k=Config.FIELD_QUERY_TOP_K,
                )
                field_rows = [{"key": r["key"], "value": r["value"]} for r in (results or [])]
            except Exception:
                logger.warning("[Field] digest query failed for task %s", task.id, exc_info=True)

        merged = field_scoring.merge_dispatch_rows(upstream_rows, field_rows)
        if not merged:
            return
        kept, truncated = field_scoring.budget_results(merged, Config.FIELD_QUERY_TOKEN_BUDGET)
        if not kept:
            return
        task.input_context = {
            **(task.input_context or {}),
            "field_digest": field_scoring.format_digest(kept, truncated=truncated),
        }
        logger.info(
            "[Field] Pinned dispatch digest (%d patterns, %d upstream) into task %s",
            len(kept), len(upstream_rows), task.id,
        )

    async def _seed_field_with_documents(
        self,
        db: Session,
        field,
        field_id: str,
        attachments: List[Dict[str, Any]],
        workspace_id,
    ) -> None:
        """Inject uploaded document references into the shared field at elevated strength."""
        from core.models.core import Document

        for att in attachments:
            doc_id = att.get("document_id")
            if not doc_id:
                continue
            try:
                doc = (
                    db.query(Document)
                    .filter(Document.id == doc_id, Document.workspace_id == workspace_id)
                    .first()
                )
                if not doc:
                    continue
                await field.inject(
                    context_id=field_id,
                    key=f"reference_doc:{doc.filename}",
                    value=(
                        f"Reference document '{doc.filename}' uploaded by user as mission context. "
                        f"Type: {doc.file_type or 'unknown'}, Size: {doc.file_size or 0} bytes. "
                        f"Tags: {doc.tags or []}. Description: {doc.description or 'N/A'}"
                    ),
                    agent_id=0,
                    strength=1.2,  # Above default so reference material ranks higher
                    provenance={"workspace_id": str(workspace_id)},
                )
                logger.info("[PRD-108] Seeded field %s with doc %s (%s)", field_id, doc_id, doc.filename)
            except Exception as e:
                logger.warning("[PRD-108] Failed to seed doc %s into field: %r", doc_id, e, exc_info=True)

    async def _save_mission_output_as_document(
        self,
        db: Session,
        run: OrchestrationRun,
    ) -> Optional[int]:
        """On mission completion, route the assembled synthesis through the
        knowledge flywheel (PRD-164 S3, Q58) so it is retrievable via RAG and
        the Knowledge Graph next turn. ``source_type='agent_output'``.

        Honors the per-workspace opt-out: an opted-out workspace ingests
        NOTHING (the run is marked so the sweep doesn't retry forever).

        For app_builder missions, also downloads the zip bundle from the
        workspace and saves it as a separate downloadable document. Can also
        emit a rendered document deliverable via the PRD-167 template path
        when ``run.config['emit_document']`` is set.
        """
        from services.knowledge_flywheel import flywheel_enabled, ingest_agent_output

        try:
            if not flywheel_enabled(db, run.workspace_id):
                # Q58 opt-out: ingest nothing; mark so the tick sweep moves on.
                run.config = {**(run.config or {}), "output_ingest": "skipped_opt_out"}
                db.flush()
                logger.info(
                    "[Mission] Workspace %s opted out of the knowledge flywheel — "
                    "skipping output ingest for mission %s",
                    run.workspace_id, run.id,
                )
                return None

            tasks = (
                db.query(OrchestrationTask)
                .filter(
                    OrchestrationTask.run_id == run.id,
                    OrchestrationTask.state == TaskState.VERIFIED.value,
                )
                .order_by(OrchestrationTask.sequence_number)
                .all()
            )
            if not tasks:
                return None

            # Assemble markdown output
            parts = [f"# Mission: {run.goal}\n"]
            for t in tasks:
                parts.append(f"## {t.sequence_number}. {t.title}\n")
                if t.output:
                    parts.append(str(t.output))
                parts.append("")
            content = "\n".join(parts)

            slug = re.sub(r"[^a-z0-9]+", "-", run.goal[:60].lower()).strip("-")

            document_id = await ingest_agent_output(
                db,
                run.workspace_id,
                content=content,
                filename=f"mission-output-{slug}.md",
                source="mission_synthesis",
                source_id=str(run.id),
                title=f"Mission output: {run.goal[:120]}",
                description=f"Output from completed mission: {run.goal[:200]}",
                created_by="coordinator",
                extra_tags=["mission-output", f"mission:{run.id}"],
            )
            if document_id is not None:
                run.config = {**(run.config or {}), "output_document_id": document_id}
                db.flush()
                logger.info("[Mission] Saved output document %s for mission %s", document_id, run.id)

            # --- PRD-167 wiring: emit a rendered document deliverable -------
            emitted_deliverable = await self._emit_mission_document(db, run, content)

            # 2026-07-30: emission was opt-in via the emit_document spec and
            # nothing sets it by default — every mission's Deliverables tab was
            # empty. When no spec produced a render, default-promote the FINAL
            # task output as the mission deliverable (intermediates stay on the
            # Reports surface).
            if not emitted_deliverable:
                await self._register_final_output_deliverable(db, run, tasks)

            # --- App builder: also save the zip bundle ---
            template_used = (run.config or {}).get("template_used")
            if template_used == "app_builder":
                from api.documents import get_document_manager

                doc_manager = get_document_manager(str(run.workspace_id))
                zip_doc_id = await self._save_app_bundle_zip(db, run, slug, doc_manager)
                if zip_doc_id:
                    run.config = {**(run.config or {}), "app_bundle_document_id": zip_doc_id}
                    db.flush()

            return document_id
        except Exception as e:
            logger.warning("[Mission] Failed to save output document for %s: %s", run.id, e, exc_info=True)
            # PRD-179 S2 (F049): stamp a failure marker so the flywheel sweep's
            # SQL-side exclusion drops this run next tick instead of re-selecting
            # it forever. Without it, one poison run silently starves the backlog.
            try:
                run.config = {
                    **(run.config or {}),
                    "output_ingest_failed": datetime.now(timezone.utc).isoformat(),
                }
                db.flush()
            except Exception:
                logger.debug(
                    "[Mission] Could not persist ingest-failure marker for run %s",
                    run.id, exc_info=True,
                )
            return None

    async def _register_final_output_deliverable(
        self,
        db: Session,
        run: OrchestrationRun,
        tasks,
    ) -> Optional[str]:
        """Register the mission's final task output as a mission deliverable.

        The mission Deliverables tab lists deliverables with
        ``source_id == run.id``, but until 2026-07-30 nothing wrote
        mission-sourced rows unless the run carried an explicit
        ``emit_document`` spec — so the tab was empty for every mission ever
        run. Promotes ONLY the final output (see ``pick_final_output_task``);
        intermediate task outputs stay on the Reports surface. Fail-soft;
        idempotent via ``config['emitted_deliverable_id']`` (shared with the
        spec path so the two never double-emit).
        """
        if (run.config or {}).get("emitted_deliverable_id"):
            return None
        final_task = pick_final_output_task(tasks)
        if final_task is None:
            return None
        try:
            from core.workspace_client import WorkspaceClient
            from services.deliverable_service import DeliverableService

            slug = re.sub(
                r"[^a-z0-9]+", "-", (final_task.title or run.goal or "")[:60].lower()
            ).strip("-") or "final-output"
            file_path = f"missions/{str(run.id)[:8]}-{slug}.md"
            content = str(final_task.output)

            ws_client = WorkspaceClient(str(run.workspace_id))
            write_result = await ws_client.write_file(file_path, content)
            if not write_result.get("success"):
                logger.warning(
                    "[Mission] Could not write final-output file for %s: %s",
                    run.id, write_result.get("error", "unknown"),
                )
                return None

            svc = DeliverableService(db, run.workspace_id)
            registration = svc.register(
                file_path=file_path,
                title=final_task.title or f"Mission output: {run.goal[:80]}",
                source_type="mission",
                source_id=str(run.id),
                # agent_name stays None — list_deliverables COALESCEs the
                # agents join, so the real agent name renders from agent_id.
                agent_id=final_task.assigned_agent_id,
                # .md infers 'report', which register() refuses (native-service
                # guard). This is the polished final document, not a filed
                # report — classify it as such.
                artifact_type="document",
                summary=content[:280],
                file_size_bytes=len(content.encode("utf-8")),
            )
            if not registration.get("success"):
                logger.warning(
                    "[Mission] Final-output deliverable registration failed for %s: %s",
                    run.id, registration.get("error", "unknown"),
                )
                return None

            deliverable_id = registration.get("deliverable_id")
            run.config = {**(run.config or {}), "emitted_deliverable_id": str(deliverable_id)}
            db.flush()
            logger.info(
                "[Mission] Registered final-output deliverable %s for mission %s",
                deliverable_id, run.id,
            )
            return str(deliverable_id)
        except Exception as e:
            logger.warning(
                "[Mission] Final-output deliverable promotion failed for %s: %s",
                run.id, e, exc_info=True,
            )
            return None

    async def _emit_mission_document(
        self,
        db: Session,
        run: OrchestrationRun,
        content: str,
    ) -> Optional[str]:
        """PRD-164 S3: mission completion can emit ``generate_document`` via the
        PRD-167 template path (merged — wired directly, no feature flag).

        Driven by ``run.config['emit_document']``::

            {"format": "pdf"|"docx"|"xlsx", "title": "...",
             "template_id": "<uuid>" | "template_name": "..."}

        The rendered file is registered as a mission deliverable (it shows in
        the mission Deliverables tab). The synthesis markdown is already
        flywheel-ingested by the caller, so the render is NOT re-ingested —
        no duplicate knowledge. Fail-soft: a render failure never fails the
        mission output flow. Idempotent via config['emitted_deliverable_id'].
        """
        spec = (run.config or {}).get("emit_document")
        if not spec or not isinstance(spec, dict):
            return None
        if (run.config or {}).get("emitted_deliverable_id"):
            return None

        try:
            from uuid import UUID as _UUID

            from modules.documents.generation_service import DocumentGenerationService

            fmt = str(spec.get("format") or "pdf").lower()
            title = spec.get("title") or f"Mission report: {run.goal[:120]}"
            template_id = None
            if spec.get("template_id"):
                template_id = _UUID(str(spec["template_id"]))

            gen_service = DocumentGenerationService(db, run.workspace_id)
            result = await gen_service.generate(
                title=title,
                format=fmt,
                data={"content": content},
                workspace_id=run.workspace_id,
                template_name=spec.get("template_name"),
                template_id=template_id,
            )
            registration = gen_service.register_as_deliverable(
                result,
                title=title,
                source_type="mission",
                source_id=str(run.id),
                agent_name="coordinator",
                template_id=template_id,
            )
            deliverable_id = (registration or {}).get("deliverable_id")
            run.config = {
                **(run.config or {}),
                "emitted_document": result.filename,
                "emitted_deliverable_id": deliverable_id or result.filename,
            }
            db.flush()
            logger.info(
                "[Mission] Emitted %s document '%s' for mission %s (deliverable %s)",
                fmt, result.filename, run.id, deliverable_id,
            )
            return deliverable_id
        except Exception:
            logger.warning(
                "[Mission] emit_document failed for mission %s (non-fatal)",
                run.id, exc_info=True,
            )
            return None

    async def _save_app_bundle_zip(
        self,
        db: Session,
        run: OrchestrationRun,
        slug: str,
        doc_manager,
    ) -> Optional[int]:
        """Download the app zip bundle from workspace and save as a document."""
        from core.workspace_client import WorkspaceClient
        from pathlib import Path

        try:
            ws_client = WorkspaceClient(str(run.workspace_id))
            result = await ws_client.download_file("artifacts/app-bundle.zip")

            if not result.get("success"):
                logger.warning(
                    "[Mission] App bundle zip not found for mission %s: %s",
                    run.id,
                    result.get("error", "unknown"),
                )
                return None

            # Write zip bytes to temp file
            output_dir = Path("/tmp/automatos_mission_outputs")
            output_dir.mkdir(exist_ok=True)
            zip_path = output_dir / f"app-{slug}.zip"
            zip_path.write_bytes(result["content"])

            try:
                zip_doc_id = await doc_manager.upload_document(
                    file_path=str(zip_path),
                    filename=f"app-{slug}.zip",
                    tags=["mission-output", "app-bundle", f"mission:{run.id}"],
                    description=f"App bundle from mission: {run.goal[:200]}",
                    created_by="coordinator",
                )
                logger.info(
                    "[Mission] Saved app bundle zip %s for mission %s",
                    zip_doc_id,
                    run.id,
                )
                return zip_doc_id
            finally:
                if zip_path.exists():
                    zip_path.unlink()
        except Exception as e:
            logger.warning(
                "[Mission] Failed to save app bundle zip for %s: %s",
                run.id,
                e,
                exc_info=True,
            )
            return None

    async def _cleanup_terminal_fields(self, db: Session) -> None:
        """Archive fields for missions that have ended (BINDING D7 stepping stone).

        Fields now share ONE Qdrant collection (``field_memory``) keyed by a
        ``field_id`` payload filter — there is no longer a per-mission
        collection to tear down. Destroying the data here made a completed
        mission's Field tab read ``not_created`` with zero patterns forever.
        Instead, archive in place: stamp ``field_archived`` + ``field_expired_at``
        and KEEP both ``field_id`` and the data, so the field stays queryable
        after the mission ends. The orphan reaper (_cleanup_orphan_field_data)
        still removes data whose run row has been purged.
        """
        terminal_with_fields = (
            db.query(OrchestrationRun)
            .filter(
                OrchestrationRun.state.in_([s.value for s in TERMINAL_RUN_STATES]),
                OrchestrationRun.config["field_id"].astext.isnot(None),
                OrchestrationRun.config["field_archived"].astext.is_(None),
            )
            .limit(50)
            .all()
        )
        field = self._get_field()
        field_inner = getattr(field, "_inner", field) if field else None
        for run in terminal_with_fields:
            cfg = run.config or {}
            field_id = cfg.get("field_id")
            if not field_id or cfg.get("field_archived"):
                continue
            # PRD-166 S1: merge the mission field into the workspace-persistent
            # field — stamp workspace_id + expired_at on its points so they join
            # cross-mission recall while the mission-scoped view soft-archives.
            if field_inner is not None and hasattr(field_inner, "archive_into_workspace"):
                try:
                    await field_inner.archive_into_workspace(field_id, str(run.workspace_id))
                except Exception:
                    logger.warning(
                        "[Field] archive_into_workspace failed for run %s", run.id, exc_info=True,
                    )
            # Archive in place — data and field_id are retained so the
            # /field endpoint can still read this mission's field.
            run.config = {
                **cfg,
                "field_archived": True,
                "field_expired_at": datetime.now(timezone.utc).isoformat(),
            }
            db.flush()

    async def _cleanup_orphan_field_data(self, db: Session) -> None:
        """Delete field memory points whose ``field_id`` has no OrchestrationRun.

        Fields now live in a single shared Qdrant collection (``field_memory``)
        with ``field_id`` as a payload filter. Orphans appear when a run row
        is purged before its field was destroyed. One delete-by-filter call
        per orphan id is atomic and cheap.

        Also sweeps any leftover legacy per-mission ``field_<uuid>``
        collections from the pre-refactor layout. Safe to call repeatedly;
        becomes a no-op once they're all gone.
        """
        field = self._get_field()
        if not field:
            return

        # Resolve the inner adapter through the instrumentation wrapper.
        inner = getattr(field, "_inner", field)
        client = getattr(inner, "_client", None)
        if client is None:
            return

        try:
            from modules.context.adapters.vector_field import (
                SHARED_COLLECTION,
                VectorFieldSharedContext,
            )
        except Exception:
            return

        if not isinstance(inner, VectorFieldSharedContext):
            return

        referenced_ids = {
            row[0] for row in db.execute(
                text(
                    "SELECT config->>'field_id' FROM orchestration_runs "
                    "WHERE config->>'field_id' IS NOT NULL"
                )
            ).fetchall()
            if row[0]
        }

        # --- Pass 1: orphan points in the shared collection ---
        try:
            scrolled, _ = await client.scroll(
                collection_name=SHARED_COLLECTION,
                limit=10000,
                with_payload=["field_id"],
                with_vectors=False,
            )
            present_ids = {
                p.payload.get("field_id") for p in scrolled
                if p.payload and p.payload.get("field_id")
            }
            orphan_ids = present_ids - referenced_ids
            for fid in orphan_ids:
                try:
                    await inner.destroy_context(fid)
                except Exception:
                    logger.warning(
                        "[Coordinator] Failed to destroy orphan field %s",
                        fid, exc_info=True,
                    )
            if orphan_ids:
                logger.info(
                    "[Coordinator] Orphan cleanup: removed points for %d field_ids",
                    len(orphan_ids),
                )
        except Exception:
            logger.debug("[Coordinator] orphan-point sweep skipped", exc_info=True)

        # --- Pass 2: legacy per-mission collections (pre-refactor) ---
        try:
            collections_resp = await client.get_collections()
            legacy = [
                c.name for c in collections_resp.collections
                if c.name.startswith("field_") and c.name != SHARED_COLLECTION
            ]
            for name in legacy[:20]:  # throttle: 20 per tick
                try:
                    await client.delete_collection(name)
                    logger.info("[Coordinator] Dropped legacy field collection %s", name)
                except Exception:
                    logger.warning(
                        "[Coordinator] Failed to drop legacy %s", name, exc_info=True,
                    )
        except Exception:
            logger.debug("[Coordinator] legacy-collection sweep skipped", exc_info=True)

    async def _save_pending_output_documents(self, db: Session) -> None:
        """Route completed missions that have not been ingested through the
        synthesis flywheel (PRD-179 S2, F049).

        The exclusion of already-handled runs is done SQL-side, not by pulling a
        batch and filtering in Python: a run is a candidate only when it carries
        NONE of the three terminal markers — ``output_document_id`` (ingested),
        ``output_ingest`` (opted out / skipped), ``output_ingest_failed``
        (previous ingest errored). Ordering is ``created_at DESC`` so the newest
        completed missions ingest first. Together these stop the pre-fix
        starvation where an unordered ``LIMIT 3`` kept re-selecting the same
        already-done rows once more than three accumulated.
        """
        candidates = (
            db.query(OrchestrationRun)
            .filter(
                OrchestrationRun.state == RunState.COMPLETED.value,
                # SQL-side already-ingested / failure exclusion (JSONB ->> IS NULL).
                OrchestrationRun.config["output_document_id"].astext.is_(None),
                OrchestrationRun.config["output_ingest"].astext.is_(None),
                OrchestrationRun.config["output_ingest_failed"].astext.is_(None),
            )
            .order_by(OrchestrationRun.created_at.desc())
            .limit(Config.FLYWHEEL_INGEST_BATCH)
            .all()
        )
        for run in candidates:
            # _save_mission_output_as_document owns its own failure handling and
            # stamps the output_ingest_failed marker on error (so a poison run
            # drops out of the candidate set). The sweep stays a thin driver.
            await self._save_mission_output_as_document(db, run)

    # ------------------------------------------------------------------
    # Scheduler lifecycle
    # ------------------------------------------------------------------

    async def start(self, scheduler=None) -> None:
        """
        Register the coordinator tick on the shared scheduler.

        Args:
            scheduler: Shared APScheduler instance from UnifiedScheduler.
                       If None, creates a local scheduler (useful for tests).
        """
        if scheduler:
            self._scheduler = scheduler
            self._owns_scheduler = False
        else:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.jobstores.memory import MemoryJobStore

            self._scheduler = AsyncIOScheduler(
                jobstores={"default": MemoryJobStore()},
            )
            self._scheduler.start()
            self._owns_scheduler = True

        self._scheduler.add_job(
            self.tick,
            "interval",
            seconds=Config.COORDINATOR_TICK_INTERVAL_SECONDS,
            id="coordinator_tick",
            replace_existing=True,
            max_instances=1,
        )
        logger.info(
            "[Coordinator] Started tick loop (interval: %ds)",
            Config.COORDINATOR_TICK_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        """Stop the scheduler if this service owns it."""
        if self._scheduler and self._owns_scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        logger.info("[Coordinator] Service stopped")

    # ------------------------------------------------------------------
    # Tick (5s interval)
    # ------------------------------------------------------------------

    async def tick(self) -> Dict[str, Any]:
        """
        Main coordinator tick — called every 5s by the scheduler.

        Finds all active (running) missions, then for each:
          1. Dispatch phase: try to dispatch the next queued task
          2. Reconcile phase: check for stalls, completions, failures

        Returns a summary dict of what happened.
        """
        if self._tick_running:
            return {"status": "skipped", "reason": "already_running"}

        self._tick_running = True
        summary: Dict[str, Any] = {
            "status": "success",
            "runs_processed": 0,
            "dispatches": 0,
            "reconciliations": 0,
            "errors": [],
        }

        try:
            from core.database.database import SessionLocal

            db = SessionLocal()
            try:
                # Find active missions (running state only — not paused/awaiting)
                active_runs: List[OrchestrationRun] = (
                    db.query(OrchestrationRun)
                    .filter(
                        OrchestrationRun.state == RunState.RUNNING.value,
                    )
                    .all()
                )

                for run in active_runs:
                    try:
                        await self._process_run(db, run)
                        summary["runs_processed"] += 1
                        db.commit()
                    except Exception:
                        logger.error(
                            "Error processing run %s", run.id, exc_info=True,
                        )
                        db.rollback()
                        summary["errors"].append(str(run.id))

                # --- PRD-163 S5: async planning — run deferred planners ---
                try:
                    planned = await self._sweep_async_planning(db)
                    if planned:
                        summary["async_planned"] = planned
                except Exception:
                    logger.warning("[Coordinator] async planning sweep failed", exc_info=True)
                    db.rollback()

                # --- PRD-163 S3: approval countdowns — auto-proceed expired plans ---
                try:
                    n = self.check_approval_countdowns(db)
                    if n:
                        db.commit()
                        summary["countdown_auto_approved"] = n
                except Exception:
                    logger.warning("[Coordinator] approval countdown sweep failed", exc_info=True)
                    db.rollback()

                # --- PRD-200 S3: re-notify (and optionally expire) parked approvals ---
                try:
                    r = await self.check_approval_renotify(db)
                    if r:
                        db.commit()
                        summary["approval_renotified"] = r
                except Exception:
                    logger.warning("[Coordinator] approval re-notify sweep failed", exc_info=True)
                    db.rollback()

                # --- PRD-108: Clean up fields for terminal runs ---
                await self._cleanup_terminal_fields(db)
                db.commit()  # Persist field_id removal to stop destroy loop

                # --- PRD-108: Drop orphan field data (no mission row) ---
                # Atomic delete-by-filter inside the shared field_memory
                # collection; also sweeps any leftover legacy per-mission
                # collections from the pre-refactor layout.
                try:
                    await self._cleanup_orphan_field_data(db)
                except Exception:
                    logger.warning("[Coordinator] Orphan field cleanup failed", exc_info=True)

                # --- Save output docs for completed missions ---
                await self._save_pending_output_documents(db)
                db.commit()

                # --- PRD-166 S1 / PRD-178 S3: field compaction (throttled hourly) ---
                await self._maybe_compact_fields(db, summary)

                # --- Archive phase (throttled to once per hour) ---
                self._maybe_archive(db, summary)

            finally:
                db.close()

        except Exception:
            logger.error("Coordinator tick failed", exc_info=True)
            summary["status"] = "error"
        finally:
            self._tick_running = False

        return summary

    async def _process_run(
        self,
        db: Session,
        run: OrchestrationRun,
    ) -> None:
        """
        Process a single active mission run: dispatch phase + reconcile phase.
        """
        workspace_id = run.workspace_id

        # --- PRD-108: Ensure field exists (lazy create for manual-approve path) ---
        # Validates the underlying Qdrant collection still exists — a stale
        # field_id can survive in run.config if it was inherited from a parent
        # mission whose collection was destroyed by _cleanup_terminal_fields.
        existing_field_id = (run.config or {}).get("field_id")
        needs_field = not existing_field_id

        if existing_field_id and not needs_field:
            field = self._get_field()
            if field and hasattr(field, "context_exists"):
                try:
                    if not await field.context_exists(existing_field_id):
                        logger.warning(
                            "[PRD-108] Stale field_id %s on mission %s — collection missing, recreating",
                            existing_field_id, run.id,
                        )
                        # Clear the stale id so _create_mission_field assigns a fresh one
                        updated_config = {**(run.config or {})}
                        updated_config.pop("field_id", None)
                        run.config = updated_config
                        db.flush()
                        needs_field = True
                except Exception as e:
                    logger.debug("[PRD-108] field exists-check raised: %s", e)

        if needs_field:
            field_id = await self._create_mission_field(db, run)
            # Seed field with uploaded document references
            if field_id:
                attachments = (run.config or {}).get("attachments", [])
                if attachments:
                    field = self._get_field()
                    if field:
                        await self._seed_field_with_documents(
                            db, field, field_id, attachments, run.workspace_id,
                        )

        # Load roster agents for this workspace
        agents: List[Agent] = (
            db.query(Agent)
            .filter(
                and_(
                    Agent.workspace_id == workspace_id,
                    Agent.status == "active",
                )
            )
            .all()
        )

        # --- Dispatch phase (parallel via dispatch_ready) ---
        dispatch_results = MissionDispatcher.dispatch_ready(db, run, agents)

        # PRD-204 S4: the dispatcher's budget gate pauses silently (sync
        # code cannot await the dispatcher). This is that transition's
        # async seam: the run entered dispatch RUNNING and came out PAUSED
        # with a budget skip reason -- tell the user, once (the tick will
        # not re-process a PAUSED run, so this cannot repeat).
        if (
            RunState(run.state) == RunState.PAUSED
            and any(
                r.skipped_reason in ("budget_exceeded", "budget_critical_deferred")
                for r in dispatch_results
            )
        ):
            await notify_mission_budget_paused(db, run)

        # Collect successfully dispatched tasks for concurrent execution
        dispatched = [r for r in dispatch_results if r.dispatched]

        if dispatched:
            # Load tasks and perform all DB state changes SERIALLY on the
            # shared session before launching concurrent agent I/O.
            tasks_to_execute = []
            for result in dispatched:
                task = (
                    db.query(OrchestrationTask)
                    .filter(OrchestrationTask.id == result.task_id)
                    .first()
                )
                if task:
                    tasks_to_execute.append((task, result.agent_id))

            # --- Phase 1: DB prep (serial on shared session) ---
            # Transition tasks to RUNNING, load upstream deps, build prompts,
            # activate agents. All DB I/O must complete before parallel phase.
            prepared = []
            for task, agent_id in tasks_to_execute:
                try:
                    prep = await self._prepare_task(db, run, task, agent_id)
                    if prep is not None:
                        prepared.append(prep)
                except Exception as exc:
                    logger.error(
                        "Task %s preparation failed: %s",
                        task.id,
                        exc,
                        exc_info=True,
                    )

            # Commit Phase-1 prep before the parallel phase so the connection
            # is not idle-in-transaction for the whole asyncio.gather of agent
            # LLM calls (PRD-135 / W1-S9). RUNNING transitions and agent
            # activations become durable here — if the process dies mid-gather,
            # the durable-execution reaper (WS-C) recovers them rather than a
            # silent end-of-tick rollback.
            end_open_transaction(db)

            # --- Phase 2: Agent I/O (parallel via asyncio.gather) ---
            if prepared:
                agent_coros = [
                    self._run_agent_io(p["factory"], p["agent_id"], p["prompt"],
                                       p["task"], p["attachment_ids"],
                                       mode_caps=p["mode_caps"],
                                       agent_runtime=p.get("agent_runtime"),
                                       field_context=p.get("field_context"))
                    for p in prepared
                ]
                results = await asyncio.gather(*agent_coros, return_exceptions=True)

                # --- Phase 3: Record completions (serial on shared session) ---
                for prep, result in zip(prepared, results):
                    task = prep["task"]
                    agent_id = prep["agent_id"]
                    if isinstance(result, Exception):
                        logger.error(
                            "Task %s agent I/O failed: %s",
                            task.id,
                            result,
                            exc_info=result,
                        )
                        result = {"status": "error", "error": str(result)}
                    await self._record_task_result(db, run, task, agent_id, result)

        # --- Reconcile phase ---
        await MissionReconciler.reconcile(db, run)

        db.refresh(run)

        if RunState(run.state) == RunState.VERIFYING:
            await self._complete_verified_run(db, run)
            db.refresh(run)

        # --- PRD-164 S4: joiner checkpoint (bounded replanning) ---
        # After dispatch + reconcile, the progress ledger decides whether the
        # run is looping without forward progress; the joiner replans within
        # COORDINATOR_MAX_REPLANS or halts. Best-effort: a joiner error never
        # breaks the tick.
        if RunState(run.state) == RunState.RUNNING:
            try:
                await self._joiner_checkpoint(db, run)
            except Exception:
                logger.error(
                    "Joiner checkpoint failed for run %s", run.id, exc_info=True,
                )

        # PRD-142 W3-S11: missions primitive heartbeat at terminal boundary.
        # tick() only picks RunState.RUNNING runs, so a terminal state here is
        # always a fresh transition this tick — emit exactly once. COMPLETED →
        # green; FAILED / CANCELLED → down (the tile reflects the user-visible
        # outcome). Best-effort: the helper swallows any emit error so a
        # broken heartbeat writer cannot fail mission completion.
        if RunState(run.state) in TERMINAL_RUN_STATES:
            _emit_missions_primitive(
                run.workspace_id,
                success=RunState(run.state) == RunState.COMPLETED,
                detail=(
                    f"run={run.id} state={run.state} "
                    f"stop_reason={run.stop_reason or 'unspecified'}"
                ),
            )
            # PRD-227 US-002: narrate the run's terminal outcome into the launching
            # thread (run-level). Same once-per-run guarantee as the heartbeat above.
            _narrate_run_terminal(db, run)

    # ------------------------------------------------------------------
    # Joiner checkpoint — bounded replanning (PRD-164 S4)
    # ------------------------------------------------------------------

    async def _joiner_checkpoint(
        self,
        db: Session,
        run: OrchestrationRun,
    ) -> None:
        """LLMCompiler-style joiner: after each tick, decide CONTINUE / REPLAN
        / HALT from the run's progress ledger (Magentic-One stall counter).

        Churn without forward progress for COORDINATOR_STALL_LEDGER_LIMIT
        checks → REPLAN through the one existing ``replan_mission`` engine
        while ``replan_count`` is under COORDINATOR_MAX_REPLANS, else HALT
        (run FAILED, ``stop_reason='stalled'``). The ledger persists on
        ``run.config['progress_ledger']`` and every verdict is emitted as a
        ``run_stall_ledger`` event — the audit trail on the mission. No new
        planner algorithm lives here (PRD-164 non-goal).
        """
        tasks = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .all()
        )
        if not tasks:
            return

        snapshot = progress_ledger.snapshot_tasks(tasks)
        previous = (run.config or {}).get("progress_ledger")
        ledger, decision = progress_ledger.advance(
            previous,
            snapshot,
            stall_limit=Config.COORDINATOR_STALL_LEDGER_LIMIT,
            replan_count=run.replan_count or 0,
            max_replans=Config.COORDINATOR_MAX_REPLANS,
        )
        if ledger is not previous:
            # New dict on change only — idle ticks skip the JSONB write.
            run.config = {**(run.config or {}), "progress_ledger": ledger}

        if decision is progress_ledger.JoinerDecision.CONTINUE:
            return

        # Audit the verdict on the mission event stream.
        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_STALL_LEDGER,
            actor_type=ActorType.COORDINATOR,
            actor_id="joiner",
            payload={
                "decision": decision.value,
                "stall_streak": ledger.get("stall_streak"),
                "stall_limit": Config.COORDINATOR_STALL_LEDGER_LIMIT,
                "snapshot": snapshot,
                "replan_count": run.replan_count or 0,
                "max_replans": Config.COORDINATOR_MAX_REPLANS,
            },
        )
        logger.warning(
            "[Joiner] Mission %s verdict=%s (streak=%s, replans=%s/%s)",
            run.id, decision.value, ledger.get("stall_streak"),
            run.replan_count or 0, Config.COORDINATOR_MAX_REPLANS,
        )

        if decision is progress_ledger.JoinerDecision.REPLAN:
            try:
                await self.replan_mission(
                    db,
                    run.id,
                    actor_id="joiner",
                    notes=(
                        "Automatic replan: the progress ledger detected "
                        f"{ledger.get('stall_streak')} consecutive checks of "
                        "churn without forward progress (tasks retrying "
                        "without completing). Replace the looping approach."
                    ),
                    actor_type=ActorType.COORDINATOR,
                    trigger="stall_ledger",
                )
            except PlanValidationError:
                # replan_mission already failed the run + emitted the audit.
                logger.warning(
                    "[Joiner] Auto-replan validation failed for run %s",
                    run.id, exc_info=True,
                )
                return
            except Exception as exc:
                # A hard replan failure may not strand the loop — halt.
                logger.error(
                    "[Joiner] Auto-replan errored for run %s — halting",
                    run.id, exc_info=True,
                )
                await self._halt_stalled_run(
                    db, run, ledger, detail=f"auto-replan errored: {exc}",
                )
                return
            # Fresh plan gets a fresh window — rebaseline the ledger.
            run.config = {
                **(run.config or {}),
                "progress_ledger": progress_ledger.reset_after_replan(ledger),
            }
            return

        # HALT — replan budget exhausted.
        await self._halt_stalled_run(
            db, run, ledger, detail="replan budget exhausted",
        )

    async def _halt_stalled_run(
        self,
        db: Session,
        run: OrchestrationRun,
        ledger: Dict[str, Any],
        *,
        detail: str,
    ) -> None:
        """Joiner HALT: fail a looping mission with a named stop reason and
        record the failure in mission memory for PRD-159 recall."""
        reason = (
            "Joiner halt: no forward progress across "
            f"{ledger.get('stall_streak')} ledger checks; {detail}"
        )
        try:
            transition_run(
                db=db,
                run=run,
                new_state=RunState.FAILED,
                actor_type=ActorType.COORDINATOR,
                actor_id="joiner",
                reason=reason,
                stop_reason=StopReason.STALLED.value,
                stop_detail=reason,
            )
        except (ConflictError, InvalidTransitionError):
            logger.warning(
                "[Joiner] Could not halt run %s (state=%s)",
                run.id, run.state, exc_info=True,
            )
            return
        # PRD-204 S4: the user is told the mission failed (joiner halt path).
        await notify_mission_failed(db, run)
        await _store_mission_memory_safe(
            db, run.id, outcome="failed", failure_reason=reason,
        )

    # ------------------------------------------------------------------
    # Synthesis support (PRD-82C US-007)
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_upstream_outputs(
        db: Session,
        task: OrchestrationTask,
    ) -> List[Dict[str, Any]]:
        """
        Fetch outputs from all dependency tasks of the given task.

        Returns list of dicts with 'title', 'description', and 'output' keys,
        ordered by sequence_number.
        """
        upstream_deps = (
            db.query(OrchestrationTaskDependency)
            .filter(OrchestrationTaskDependency.task_id == task.id)
            .all()
        )
        if not upstream_deps:
            return []

        dep_task_ids = [d.depends_on_task_id for d in upstream_deps]
        dep_tasks = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.id.in_(dep_task_ids))
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        # Sanitize and truncate. Synthesis is the one consumer that still gets
        # full upstream outputs — its whole job is merging them (min_length
        # verification is 50% of the combined upstream), so the Q22 dispatch
        # digest deliberately does NOT apply here.
        _PER_OUTPUT_LIMIT = 8000
        _TOTAL_BUDGET = 30_000

        results: List[Dict[str, Any]] = []
        accumulated = 0
        for dt in dep_tasks:
            raw_output = dt.output or ""
            # Strip base64 blobs
            cleaned = _sanitize_for_field(raw_output)
            # Per-output truncation
            remaining = _TOTAL_BUDGET - accumulated
            if remaining <= 0:
                break
            limit = min(len(cleaned), _PER_OUTPUT_LIMIT, remaining)
            truncated = cleaned[:limit]
            if len(cleaned) > limit:
                truncated += "\n\n... (truncated)"
            accumulated += len(truncated)
            results.append({
                "title": dt.title,
                "description": (dt.description or "")[:500],
                "output": truncated,
            })
        return results

    @staticmethod
    def _collect_upstream_digest_rows(
        db: Session,
        task: OrchestrationTask,
    ) -> List[Dict[str, Any]]:
        """PRD-164 S4 (Q22): a task's immediate upstream-dependency outputs as
        field-shaped ``{key, value}`` rows for the dispatch digest.

        Keyed by the upstream task title — the same key the output was injected
        into the field under (``_inject_task_output_into_field``) — so the
        digest merge dedupes the field's echo of the same content. Values are
        sanitized and capped exactly like field injection; the per-task TOKEN
        budget is applied later by ``_attach_field_digest``.
        """
        upstream_deps = (
            db.query(OrchestrationTaskDependency)
            .filter(OrchestrationTaskDependency.task_id == task.id)
            .all()
        )
        if not upstream_deps:
            return []

        dep_task_ids = [d.depends_on_task_id for d in upstream_deps]
        dep_tasks = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.id.in_(dep_task_ids))
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )
        return [
            {
                "key": dt.title or f"task_{dt.sequence_number}",
                "value": _sanitize_for_field(str(dt.output))[:FIELD_VALUE_CAP_CHARS],
            }
            for dt in dep_tasks
            if dt.output
        ]

    @staticmethod
    def _build_synthesis_prompt(
        task: OrchestrationTask,
        upstream_outputs: List[Dict[str, Any]],
    ) -> str:
        """
        Build a specialised prompt for TaskType.SYNTHESIS tasks that merges
        parallel upstream outputs into a unified result.
        """
        parts = [
            f"# Synthesis Task: {task.title}",
            "",
            "You are synthesising the outputs of multiple upstream tasks into a "
            "single, unified result. Follow these rules:",
            "- Include ALL substantive content from every upstream output.",
            "- Resolve any contradictions by noting the discrepancy and choosing "
            "the better-supported position.",
            "- Write in a unified voice — this must read as one cohesive document, "
            "NOT a concatenation of separate pieces.",
            "- Preserve important details, data, and citations from each source.",
        ]

        if task.description:
            parts.append(f"\n## Task Description\n{task.description}")

        parts.append("\n## Upstream Outputs to Synthesise")
        for i, upstream in enumerate(upstream_outputs, 1):
            title = upstream.get("title", f"Task {i}")
            output = upstream.get("output", "(no output)")
            parts.append(f"\n### {i}. {title}\n{output}")

        if not upstream_outputs:
            parts.append(
                "\n*No upstream outputs available — produce the best result "
                "you can from the task description alone.*"
            )

        # PRD-108: Tell synthesis agent about the shared field
        field_id = (task.input_context or {}).get("field_id")
        if field_id:
            parts.append(
                "\n## Shared Mission Field\n"
                "You have access to the shared mission field. Use "
                "**platform_field_query** to search for additional findings, "
                "analysis, or context from other agents that may not appear in "
                "the upstream outputs above. Query the field for key topics "
                "before synthesising to ensure nothing is missed."
            )

        return "\n".join(parts)

    @staticmethod
    def _auto_synthesis_verification_criteria(
        task: OrchestrationTask,
        upstream_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate verification criteria for synthesis tasks.

        NOTE: We intentionally skip required_sections for synthesis —
        the LLM planner can't predict exact headings the agent will use,
        and exact-match checks cause PARTIAL downgrades that burn tokens
        on retries. The LLM judge already evaluates content quality.
        """
        criteria: List[Dict[str, Any]] = []

        # Minimum length — 50% of combined upstream
        combined_length = sum(len(u.get("output", "")) for u in upstream_outputs)
        min_length = max(combined_length // 2, 200)  # floor at 200 chars
        criteria.append({
            "type": "min_length",
            "value": min_length,
        })

        return criteria

    async def _prepare_task(
        self,
        db: Session,
        run: OrchestrationRun,
        task: OrchestrationTask,
        agent_id: int,
    ) -> Optional[Dict[str, Any]]:
        """DB prep phase for a task — runs serially on the shared session.

        Transitions ASSIGNED → RUNNING, loads upstream context, builds prompt,
        activates agent. Returns a dict with everything needed for the parallel
        agent I/O phase, or None if the task couldn't be prepared.
        """
        from modules.agents.factory.agent_factory import AgentFactory

        # Transition to RUNNING
        try:
            MissionDispatcher.record_task_running(db, task)
            db.flush()
        except (ConflictError, InvalidTransitionError):
            logger.warning(
                "Could not transition task %s to running", task.id,
                exc_info=True,
            )
            return None

        # PRD-164 S4 (Q22): upstream task outputs reach the agent as the
        # token-budgeted dispatch digest (PRD-166 field digest), NOT as raw
        # 8K-chars-per-task stuffing. Synthesis tasks are the one exception —
        # their job is merging full upstream content (_collect_upstream_outputs
        # below), so they skip the digest rows.
        is_synthesis = task.task_type == TaskType.SYNTHESIS.value
        upstream_rows: List[Dict[str, Any]] = (
            [] if is_synthesis
            else self._collect_upstream_digest_rows(db, task)
        )

        # PRD-108: Pass field_id so agent can query the shared field
        field_id = (run.config or {}).get("field_id")
        if field_id:
            task.input_context = {
                **(task.input_context or {}),
                "field_id": field_id,
            }
        # PRD-166 S3 + PRD-164 S4: pin the budgeted dispatch digest (immediate
        # upstream outputs merged ahead of semantic field hits) into the prompt
        # so the agent starts with accumulated knowledge instead of having to
        # query for it. Runs even without a field backend — upstream rows
        # still flow, under the same per-task budget.
        if upstream_rows or field_id:
            await self._attach_field_digest(
                db, run, task, field_id, agent_id, upstream_rows=upstream_rows,
            )

        # PRD-127: Get attachment_ids for this task
        task_attachment_ids: List[str] = []
        if hasattr(task, "attachment_ids") and task.attachment_ids:
            task_attachment_ids = task.attachment_ids
        else:
            task_attachment_ids = (run.config or {}).get("attachment_ids", [])

        # Build the prompt — synthesis tasks use a specialised prompt
        # (is_synthesis resolved above, before the dispatch-digest step).
        ctx = task.input_context or {}
        previous_output = ctx.get("previous_output")
        verification_feedback = ctx.get("verification_feedback")
        is_revision = bool(previous_output and verification_feedback)

        if is_synthesis:
            synthesis_upstream = self._collect_upstream_outputs(db, task)

            if is_revision:
                failures = verification_feedback.get("failures", [])
                reasoning = verification_feedback.get("reasoning", "Unknown")
                attempt = verification_feedback.get("attempt", "?")

                prompt_parts = [
                    f"# Revision Request: {task.title}",
                    f"\nYour previous synthesis (attempt {attempt}) needs revision. "
                    f"Do NOT rewrite from scratch — revise the content below to "
                    f"address the feedback while preserving everything that was good.",
                    f"\n## Issues to Fix\n{reasoning}",
                ]
                if failures:
                    prompt_parts.append(
                        "Failed checks: " + ", ".join(failures)
                    )
                prompt_parts.append(
                    f"\n## Your Previous Output (revise this)\n\n{previous_output}"
                )
                prompt = "\n".join(prompt_parts)
                logger.info(
                    "Built revision prompt for synthesis task %s (attempt %s)",
                    task.id, attempt,
                )
            else:
                prompt = self._build_synthesis_prompt(task, synthesis_upstream)

            if not task.verification_criteria:
                task.verification_criteria = (
                    self._auto_synthesis_verification_criteria(task, synthesis_upstream)
                )
                logger.info(
                    "Auto-set verification criteria for synthesis task %s: %s",
                    task.id,
                    task.verification_criteria,
                )
        else:
            prompt = MissionDispatcher.build_task_prompt(task)

        # Activate agent with power-mode overrides
        factory = AgentFactory(db_session=db)
        run_config = run.config or {}
        # An explicit run_config mode wins; otherwise inherit the workspace default
        # (workspace.settings['power_mode'], set via platform_set_power_mode / a
        # HARNESS power_mode_* prescription); falling back to 'standard'. (W4-S5.)
        power_mode = (
            run_config.get("power_mode")
            or _workspace_power_mode_default(run.workspace_id, db)
            or "standard"
        )
        mode_caps = _get_power_mode_caps(power_mode, db)

        force_tier = mode_caps.get("force_llm_tier")
        if force_tier:
            factory.active_agents.pop(agent_id, None)
            agent_runtime = await factory.activate_agent(
                agent_id, workspace_dir="/tmp/automatos_workspace",
                force_llm_tier=force_tier,
            )
            logger.info(
                "Task %s: power_mode=%s, force_llm_tier=%s for agent %d",
                task.id, power_mode, force_tier, agent_id,
            )
        else:
            agent_runtime = factory.active_agents.get(agent_id)
            if not agent_runtime:
                agent_runtime = await factory.activate_agent(agent_id, workspace_dir="/tmp/automatos_workspace")

        # The agent's max_tokens budget is resolved by AgentFactory from the
        # agent's own Max Output Tokens setting (then the model's registry
        # ceiling, then DEFAULT_MAX_OUTPUT_TOKENS). Power mode governs only the
        # LLM tier and tool-iteration count — never the token budget. Do not
        # re-introduce a power-mode token floor here: it silently capped large
        # mission writes regardless of what the agent was configured to produce.

        # Model selection is driven by power_mode + the agent's own configured
        # model. There is intentionally NO synthesis-specific override:
        #   light    → force_llm_tier=system_llm   (gemini-2.5-flash)
        #   standard → agent's own model           (e.g. DeepSeek for QUILL)
        #   max      → force_llm_tier=orchestrator_llm (Auto's premium model)
        # System LLM is reserved for codegraph / Mem0 / planner — never
        # auto-applied to mission tasks just because they're synthesis-typed.

        return {
            "task": task,
            "agent_id": agent_id,
            "agent_runtime": agent_runtime,
            "prompt": prompt,
            "factory": factory,
            "attachment_ids": task_attachment_ids,
            # Caps resolved here (serial DB path) so the concurrent I/O phase
            # never touches the DB — see _get_power_mode_caps / _run_agent_io.
            "mode_caps": mode_caps,
            # PRD-178 S1 (F020): bind the agent's field tools to THIS task's run,
            # resolved on the serial DB path. Threaded into execute_with_prompt →
            # the tool loop → PlatformActionExecutor so field_id no longer comes
            # from a `.first()` guess over concurrent running missions.
            "field_context": {
                "field_id": field_id,
                "mission_id": str(run.id),
            } if field_id else None,
        }

    async def _run_agent_io(
        self,
        factory: Any,
        agent_id: int,
        prompt: str,
        task: Any,
        attachment_ids: List[str],
        mode_caps: Optional[Dict[str, Any]] = None,
        agent_runtime: Optional[Any] = None,
        field_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute agent I/O — safe to run concurrently via asyncio.gather().

        No DB access here — only the LLM + tool loop. ``mode_caps`` is resolved
        upstream in _prepare_task (the serial DB phase) and passed in, so this
        concurrent path never reads system_settings.
        """
        caps = mode_caps or _POWER_MODE_DEFAULTS["standard"]
        max_iters = caps["max_tool_iterations"]
        # PRD-163 S5: per-power-mode timeout (falls back to the global default).
        task_timeout = caps.get("timeout_seconds") or Config.COORDINATOR_TASK_EXECUTION_TIMEOUT

        # Pass runtime directly when we have it so the factory cache can't
        # swap in a stale cached runtime under us mid-flight.
        agent_arg: Any = agent_runtime if agent_runtime is not None else agent_id

        try:
            result = await asyncio.wait_for(
                factory.execute_with_prompt(
                    agent=agent_arg,
                    prompt=prompt,
                    max_retries=0,
                    max_tool_iterations=max_iters,
                    attachment_ids=attachment_ids,
                    # PRD-178 S1 (F020): bind field tools to THIS task's run.
                    context=field_context,
                ),
                timeout=task_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Task %s execution timed out after %ds (agent=%d)",
                task.id,
                task_timeout,
                agent_id,
            )
            result = {
                "status": "error",
                "error": f"Execution timed out after {task_timeout}s",
            }
        except Exception as exc:
            logger.error(
                "execute_with_prompt failed for task %s: %s",
                task.id,
                exc,
                exc_info=True,
            )
            result = {"status": "error", "error": str(exc)}
        return result

    async def _record_task_result(
        self,
        db: Session,
        run: OrchestrationRun,
        task: OrchestrationTask,
        agent_id: int,
        result: Dict[str, Any],
    ) -> None:
        """Record task completion/failure — runs serially on shared session."""
        MissionDispatcher.record_task_completion(db, task, result)

        # PRD-131d Phase 2: capture permanent agent-error failures into memory.
        # record_task_completion transitions to FAILED only when retries are
        # exhausted; re-queued retries stay in QUEUED and should not fire here.
        try:
            if task.state == TaskState.FAILED.value:
                from core.services.mission_memory_service import MissionMemoryService
                await MissionMemoryService(db=db).store_task_failure(task=task)
        except Exception:
            logger.warning(
                "Task failure memory capture skipped for task %s",
                task.id, exc_info=True,
            )

        # PRD-128: dispatch mission_step_complete (default pref is 'silent'
        # so this is opt-in per workspace/user)
        try:
            step_agent = db.query(Agent).filter(Agent.id == agent_id).first()
            step_agent_name = step_agent.name if step_agent else f"agent-{agent_id}"
        except Exception:
            step_agent_name = f"agent-{agent_id}"
        step_output = result.get("output") or result.get("result") or ""
        await _dispatch_mission_event(
            db=db,
            run=run,
            event_type="mission_step_complete",
            title=f"Mission step: {task.title or task.id}",
            message=str(step_output)[:500] if step_output else None,
            agent_id=agent_id,
            agent_name=step_agent_name,
            status="ok" if result.get("status") == "success" else "error",
        )

        # PRD-227 US-002: narrate the task's terminal outcome into the launching
        # thread (task-level → throttled for large plans). Only fire on a settled
        # terminal state — a re-queued retry stays QUEUED and must not narrate.
        _tlabel = task.title or f"task {task.id}"
        if task.state == TaskState.COMPLETED.value:
            _narrate_mission(db, run, f"✓ Task complete: {_tlabel}", level="task", event="task_completed")
        elif task.state == TaskState.FAILED.value:
            _narrate_mission(db, run, f"✗ Task failed: {_tlabel}", level="task", event="task_failed")

        # PRD-108: Inject completed task output into shared field
        if result.get("status") == "success":
            db.refresh(task)
            await self._inject_task_output_into_field(run, task, agent_id)

        # Update run-level token tracking (PRD-82A Section 9)
        task_tokens = result.get("execution", {}).get("tokens_used", 0)
        if task_tokens:
            run.tokens_used = (run.tokens_used or 0) + task_tokens

            if (
                run.token_budget_estimate
                and run.tokens_used > run.token_budget_estimate * 1.5
            ):
                emit_event(
                    db=db,
                    run_id=run.id,
                    event_type=EventType.BUDGET_WARNING,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="coordinator",
                    payload={
                        # US-009 user-facing fields: tell the user what limit was
                        # hit and how to raise it, before any budget-driven pause.
                        "limit_type": "mission_token_budget",
                        "spent": run.tokens_used,
                        "limit": run.token_budget_estimate,
                        "message": (
                            f"This mission has used {run.tokens_used:,} tokens, over its "
                            f"estimated budget of {run.token_budget_estimate:,}. It will keep "
                            "running; an admin can raise the mission token budget or the "
                            "power-mode caps in Settings > Coordination."
                        ),
                        # Established diagnostic fields (parity with reconciler emit).
                        "tokens_used": run.tokens_used,
                        "token_budget_estimate": run.token_budget_estimate,
                        "ratio": round(run.tokens_used / run.token_budget_estimate, 2),
                    },
                )

    # ------------------------------------------------------------------
    # Lifecycle: create_mission
    # ------------------------------------------------------------------

    async def create_mission(
        self,
        db: Session,
        workspace_id: UUID,
        goal: str,
        created_by: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationRun:
        """
        Create a new mission: plan → create DB rows → board tasks → await approval.

        Args:
            db: SQLAlchemy session (caller manages transaction).
            workspace_id: Owning workspace UUID.
            goal: Natural-language goal string.
            created_by: Clerk user ID (e.g., 'user_xxx').
            config: Optional mission config overrides.

        Returns:
            The created OrchestrationRun.

        Raises:
            PlanValidationError: if planner cannot produce a valid plan.
        """
        mission_config = config or {}

        # Create the run in PENDING state
        run = OrchestrationRun(
            workspace_id=workspace_id,
            goal=goal,
            created_by=created_by,
            config=mission_config,
            state=RunState.PENDING.value,
            state_type="initial",
            max_retries=mission_config.get(
                "max_retries", Config.COORDINATOR_MAX_TASK_RETRIES,
            ),
            max_concurrent=1,  # Sequential only in 82A
        )
        db.add(run)
        db.flush()  # Get the run.id

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_CREATED,
            actor_type=ActorType.HUMAN,
            actor_id=created_by,
            payload={"goal": goal[:500]},
        )

        logger.info(
            "Mission created: run=%s workspace=%s goal='%s'",
            run.id,
            workspace_id,
            goal[:80],
        )

        # Transition to PLANNING
        transition_run(
            db=db,
            run=run,
            new_state=RunState.PLANNING,
            actor_type=ActorType.COORDINATOR,
            actor_id="coordinator",
        )

        # PRD-163 S5: async planning — return immediately in PLANNING; the
        # coordinator tick sweep runs the planner and the plan lands via a
        # mission_plan_ready notification. The default path stays synchronous
        # (the in-chat approval card is emitted from the create tool result,
        # which needs the plan inline).
        if mission_config.get("async_planning"):
            logger.info("Mission %s → async planning (deferred to coordinator tick)", run.id)
            return run

        return await self._run_planning(db, run)

    async def _run_planning(self, db: Session, run: OrchestrationRun) -> OrchestrationRun:
        """PRD-163 S5: the planning tail — load agents, decompose, persist, and
        evaluate approval. Shared by synchronous ``create_mission`` and the async
        planning tick sweep (``_sweep_async_planning``). On reaching
        awaiting_approval it dispatches a ``mission_plan_ready`` notification to
        the creating user (S1).

        Raises PlanValidationError (after transitioning the run to FAILED) if the
        planner cannot produce a valid plan.
        """
        workspace_id = run.workspace_id
        goal = run.goal
        mission_config = run.config or {}

        # Load roster agents
        agents: List[Agent] = (
            db.query(Agent)
            .filter(
                and_(
                    Agent.workspace_id == workspace_id,
                    Agent.status == "active",
                )
            )
            .all()
        )

        # Decompose goal into task DAG
        try:
            decomposition = await MissionPlanner.decompose(
                goal=goal,
                workspace_id=workspace_id,
                agents=agents,
                config=mission_config,
                db=db,  # PRD-164 S1: enables the planning context pack
            )
        except PlanValidationError:
            transition_run(
                db=db,
                run=run,
                new_state=RunState.FAILED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason="Plan validation failed after all retries",
                stop_reason="coordinator_error",
                stop_detail="Plan validation failed after all retries",
            )
            # PRD-204 S4: the user is told the mission failed (planning path).
            await notify_mission_failed(db, run)
            await _store_mission_memory_safe(
                db, run.id, outcome="failed",
                failure_reason="Plan validation failed after all retries",
            )
            raise

        # Store the plan + create task/dependency rows (PRD-163 S2: shared
        # with import_plan so an imported plan persists the exact given DAG).
        temp_id_to_task = self._persist_decomposition(db, run, decomposition)

        # PRD-164 S2: agent-match preview — rank candidates per task (Q21
        # blend, one embedding call per task) and persist the reasons on the
        # task rows + plan snapshot so the approval card can show them.
        planned_tasks = list(temp_id_to_task.values())
        try:
            signals_by_task = await asyncio.wait_for(
                AgentMatcher.compute_signals_for_tasks(
                    planned_tasks, agents, run.workspace_id,
                ),
                timeout=Config.AGENT_MATCH_SIGNAL_TIMEOUT_SECONDS
                * max(1, len(planned_tasks)),
            )
        except Exception:
            logger.warning(
                "Semantic match signals unavailable for run %s (lexical preview)",
                run.id, exc_info=True,
            )
            signals_by_task = {}
        self._annotate_match_previews(db, run, agents, planned_tasks, signals_by_task)

        # Create board tasks for kanban visibility
        try:
            create_mission_board_task(db, run)
            for task in temp_id_to_task.values():
                create_task_board_task(db, run, task)
        except Exception:
            logger.warning(
                "Failed to create board tasks for mission %s",
                run.id,
                exc_info=True,
            )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_PLAN_READY,
            actor_type=ActorType.COORDINATOR,
            actor_id="coordinator",
            payload={
                "task_count": len(decomposition.tasks),
                "token_estimate": decomposition.token_estimate,
            },
        )

        # PRD-163 S3: decide auto-approve vs await-approval via the workspace
        # approval policy ($ ceiling / full_auto + §12.3 gate). plan_only always
        # awaits (S2). A per-request auto_approve from chat is an explicit override.
        from core.services.approval_policy import evaluate_approval, ApprovalDecision

        estimated_cost = self._estimate_cost_usd(decomposition.token_estimate)
        if mission_config.get("plan_only"):
            decision = ApprovalDecision(
                auto_approve=False, reason="plan_only", policy="plan_only",
                ceiling=None, estimated_cost=estimated_cost, countdown_seconds=None,
            )
        else:
            decision = evaluate_approval(
                db, workspace_id, estimated_cost,
                override_auto_approve=bool(mission_config.get("auto_approve", False)),
            )

        if decision.auto_approve:
            transition_run(
                db=db,
                run=run,
                new_state=RunState.RUNNING,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason=f"Auto-approved: {decision.reason}",
            )
            # PRD-163 S3: distinct audit event with the policy + ceiling snapshot.
            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.RUN_AUTO_APPROVED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                payload=decision.audit_snapshot(),
            )
            self._queue_initial_tasks(db, run)
            await self._create_mission_field(db, run)
            logger.info("Mission %s auto-approved (%s) → running", run.id, decision.reason)
        else:
            # PRD-163 S3: countdown — store a deadline; the tick loop auto-proceeds
            # when it passes (cancelled by an explicit approve/reject in the meantime).
            if decision.countdown_seconds:
                # datetime/timezone/timedelta are module-level (see top imports);
                # a local re-import here would shadow them as function-locals and
                # UnboundLocalError the config stamp below on the no-countdown path.
                deadline = datetime.now(timezone.utc) + timedelta(seconds=decision.countdown_seconds)
                run.config = {
                    **(run.config or {}),
                    "approval_deadline_at": deadline.isoformat(),
                    "approval_countdown_seconds": decision.countdown_seconds,
                }
            transition_run(
                db=db,
                run=run,
                new_state=RunState.AWAITING_APPROVAL,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
            )
            # PRD-200 S3: stamp the notification baseline (so the re-notify sweep
            # can age this park) and the priced cost (so the approvals inbox card,
            # which composes the mission list, can show est. cost — the run row
            # carries no dollar field). Immutable merge preserves any countdown
            # deadline set above.
            run.config = {
                **(run.config or {}),
                "approval_last_notified_at": datetime.now(timezone.utc).isoformat(),
                "approval_estimated_cost_usd": round(decision.estimated_cost, 2),
            }
            logger.info("Mission %s → awaiting_approval (%s)", run.id, decision.reason)
            # PRD-163 S1/S5: tell the creating user the plan is ready for review.
            # This is the "plan lands via notification" half of async planning and
            # the mission_plan_ready notification S1 requires.
            task_count = len((run.plan or {}).get("tasks", []))
            await _dispatch_mission_event(
                db=db,
                run=run,
                event_type="mission_plan_ready",
                title=f"Mission plan ready: {(run.goal or 'Mission')[:120]}",
                message=(
                    f"{task_count} task(s) planned, est. ${decision.estimated_cost:.2f}. "
                    f"Review and approve to start execution."
                ),
                status="action_required",
            )

        return run

    def _estimate_cost_usd(self, token_estimate: int) -> float:
        """PRD-163 S3/S5: dollar cost for a token estimate — the currency the
        approval policy and budget ceilings are denominated in.

        PRD-192 S3 (F059 finish): priced through ``modules.policy.pricing`` —
        the ONE pricing source. A mission plan spans multiple agents/models, so
        no single model id exists at plan time; pricing applies its documented
        flat last-resort rate (the demoted coordinator constant, now owned by
        pricing.py alone — source-grep-guarded).
        """
        from modules.policy import pricing as _pricing

        return _pricing.price_total_tokens_usd(None, None, token_estimate)

    def check_approval_countdowns(self, db: Session, workspace_id: Optional[UUID] = None) -> int:
        """PRD-163 S3: auto-proceed any awaiting-approval mission whose countdown
        deadline has passed. Returns the number auto-approved. ``workspace_id=None``
        sweeps every workspace (the coordinator tick uses this). Cancelable: an
        explicit approve/reject moves the run out of awaiting_approval, so it is
        no longer a candidate here.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        q = db.query(OrchestrationRun).filter(
            OrchestrationRun.state == RunState.AWAITING_APPROVAL.value,
        )
        if workspace_id is not None:
            q = q.filter(OrchestrationRun.workspace_id == workspace_id)
        candidates = q.all()
        proceeded = 0
        for run in candidates:
            cfg = run.config or {}
            deadline_raw = cfg.get("approval_deadline_at")
            if not deadline_raw:
                continue
            try:
                deadline = datetime.fromisoformat(deadline_raw)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if now < deadline:
                continue
            transition_run(
                db=db, run=run, new_state=RunState.RUNNING,
                actor_type=ActorType.COORDINATOR, actor_id="coordinator",
                reason="Auto-approved: approval countdown elapsed",
            )
            emit_event(
                db=db, run_id=run.id, event_type=EventType.RUN_AUTO_APPROVED,
                actor_type=ActorType.COORDINATOR, actor_id="coordinator",
                payload={"auto_approved": True, "reason": "countdown_elapsed",
                         "deadline_at": deadline_raw},
            )
            self._queue_initial_tasks(db, run)
            proceeded += 1
            logger.info("Mission %s auto-proceeded (countdown elapsed)", run.id)
        return proceeded

    async def check_approval_renotify(
        self, db: Session, workspace_id: Optional[UUID] = None
    ) -> int:
        """PRD-200 S3: re-ping (and optionally expire) missions parked at
        awaiting_approval so a parked plan does not die after one notification.

        A parked run fires exactly one ``mission_plan_ready`` notification and is
        then invisible to the coordinator (the tick only processes RUNNING runs),
        so 47% of all missions ever created sit stranded at their approval gate.
        This sweep re-dispatches that notification every
        ``COORDINATOR_APPROVAL_RENOTIFY_SECONDS`` so the plan re-pings instead of
        dying. When ``COORDINATOR_APPROVAL_EXPIRY_ENABLED`` (default OFF — under
        the ``always_ask`` posture, terminating an unapproved plan is the
        operator's call, Q5), a plan older than
        ``COORDINATOR_APPROVAL_MAX_AGE_SECONDS`` is cancelled.

        Never touches a non-AWAITING run. ``workspace_id=None`` sweeps every
        workspace (the tick uses this). Returns the number of parked runs acted
        on (re-notified or expired), so the caller commits only when something
        changed.
        """
        def _aware(dt):
            if dt is None:
                return None
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

        def _parse_iso(raw):
            if not raw:
                return None
            try:
                return _aware(datetime.fromisoformat(raw))
            except (ValueError, TypeError):
                return None

        now = datetime.now(timezone.utc)
        renotify_after = Config.COORDINATOR_APPROVAL_RENOTIFY_SECONDS

        q = db.query(OrchestrationRun).filter(
            OrchestrationRun.state == RunState.AWAITING_APPROVAL.value,
        )
        if workspace_id is not None:
            q = q.filter(OrchestrationRun.workspace_id == workspace_id)
        candidates = q.all()

        acted = 0
        for run in candidates:
            cfg = run.config or {}

            # Optional expiry (OFF by default): cancel a plan that has sat
            # unapproved past the max age, measured from creation.
            if Config.COORDINATOR_APPROVAL_EXPIRY_ENABLED:
                created = _aware(run.created_at)
                if (
                    created is not None
                    and (now - created).total_seconds()
                    >= Config.COORDINATOR_APPROVAL_MAX_AGE_SECONDS
                ):
                    try:
                        transition_run(
                            db=db,
                            run=run,
                            new_state=RunState.CANCELLED,
                            actor_type=ActorType.COORDINATOR,
                            actor_id="coordinator",
                            reason="Approval expired: plan sat unapproved past the max age",
                            stop_reason="approval_expired",
                            stop_detail=(
                                f"No approval within "
                                f"{Config.COORDINATOR_APPROVAL_MAX_AGE_SECONDS}s"
                            ),
                        )
                        acted += 1
                        # PRD-227 US-002: narrate the approval-expiry cancel
                        # (out-of-tick sweep — the tick observer never sees it).
                        _narrate_run_terminal(db, run)
                        logger.info("Mission %s cancelled (approval expired)", run.id)
                    except ConflictError:
                        logger.warning(
                            "Conflict cancelling expired approval %s", run.id
                        )
                    continue

            # Re-notify baseline: last notification, else the parked run's last
            # write (a parked run is not otherwise touched, so updated_at ~= park
            # time), else creation.
            baseline = (
                _parse_iso(cfg.get("approval_last_notified_at"))
                or _aware(run.updated_at)
                or _aware(run.created_at)
            )
            if baseline is None:
                continue
            if (now - baseline).total_seconds() < renotify_after:
                continue

            task_count = len((run.plan or {}).get("tasks", []))
            cost = cfg.get("approval_estimated_cost_usd")
            cost_txt = f", est. ${cost:.2f}" if isinstance(cost, (int, float)) else ""
            await _dispatch_mission_event(
                db=db,
                run=run,
                event_type="mission_plan_ready",
                title=f"Reminder — mission plan awaiting approval: {(run.goal or 'Mission')[:110]}",
                message=(
                    f"{task_count} task(s) planned{cost_txt}. "
                    f"Still awaiting your review and approval."
                ),
                status="action_required",
            )
            run.config = {**cfg, "approval_last_notified_at": now.isoformat()}
            acted += 1

        return acted

    async def _sweep_async_planning(self, db: Session) -> int:
        """PRD-163 S5: run the planner for missions created with
        ``config.async_planning`` that are parked in PLANNING with no plan yet.
        ``create_mission`` returns those immediately; this sweep produces the
        plan and (via ``_run_planning``) fires the mission_plan_ready
        notification. Returns the number planned this tick.
        """
        candidates: List[OrchestrationRun] = (
            db.query(OrchestrationRun)
            .filter(OrchestrationRun.state == RunState.PLANNING.value)
            .all()
        )
        planned = 0
        for run in candidates:
            cfg = run.config or {}
            if not cfg.get("async_planning") or run.plan:
                continue  # not an async-planning run, or already planned
            try:
                await self._run_planning(db, run)
                db.commit()
                planned += 1
            except PlanValidationError:
                # _run_planning already transitioned the run to FAILED in-session.
                db.commit()
                logger.warning("[Coordinator] async planning failed for run %s", run.id)
            except Exception:
                logger.error(
                    "[Coordinator] async planning error for run %s", run.id, exc_info=True,
                )
                db.rollback()
        return planned

    # ------------------------------------------------------------------
    # Lifecycle: approve_plan
    # ------------------------------------------------------------------

    def _persist_decomposition(
        self,
        db: Session,
        run: OrchestrationRun,
        decomposition,
    ) -> Dict[str, OrchestrationTask]:
        """Write a decomposition (planner output OR an imported plan) to
        ``run.plan`` + OrchestrationTask / dependency rows. Returns the
        temp_id -> task map. PRD-163 S2: shared by create_mission and import_plan
        so an imported plan persists the EXACT given DAG (no re-decomposition)."""
        run.plan = {
            "tasks": [
                {
                    "temp_id": t.temp_id,
                    "title": t.title,
                    "description": t.description,
                    "agent_role": t.agent_role,
                    "sequence_number": t.sequence_number,
                    "task_type": t.task_type,
                    "complexity": getattr(t, "complexity", "moderate"),
                    "parallel_group": getattr(t, "parallel_group", None),
                }
                for t in decomposition.tasks
            ],
            "dependencies": [
                {"from": d.from_task_temp_id, "to": d.to_task_temp_id}
                for d in decomposition.dependencies
            ],
        }
        run.token_budget_estimate = decomposition.token_estimate
        run.max_concurrent = decomposition.max_concurrent
        if decomposition.template_used:
            run.config = {**(run.config or {}), "template_used": decomposition.template_used}

        temp_id_to_task: Dict[str, OrchestrationTask] = {}
        for planned in decomposition.tasks:
            task = OrchestrationTask(
                run_id=run.id,
                title=planned.title,
                description=planned.description,
                task_type=planned.task_type,
                sequence_number=planned.sequence_number,
                agent_role=planned.agent_role,
                state=TaskState.PENDING.value,
                state_type="initial",
                verification_criteria=planned.verification_criteria or None,
                input_context={"required_tools": planned.required_tools} if planned.required_tools else None,
                max_retries=run.max_retries,
                complexity=getattr(planned, "complexity", "moderate"),
                parallel_group=getattr(planned, "parallel_group", None),
                estimated_tokens=COMPLEXITY_TOKEN_BUDGET.get(
                    getattr(planned, "complexity", "moderate"), 4000
                ),
            )
            db.add(task)
            db.flush()  # Get task.id
            temp_id_to_task[planned.temp_id] = task

            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.TASK_CREATED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                task_id=task.id,
                payload={
                    "title": planned.title,
                    "sequence_number": planned.sequence_number,
                    "agent_role": planned.agent_role,
                },
            )

        for dep in decomposition.dependencies:
            from_task = temp_id_to_task.get(dep.from_task_temp_id)
            to_task = temp_id_to_task.get(dep.to_task_temp_id)
            if from_task and to_task:
                db.add(OrchestrationTaskDependency(
                    task_id=to_task.id,
                    depends_on_task_id=from_task.id,
                ))

        db.flush()
        return temp_id_to_task

    def _annotate_match_previews(
        self,
        db: Session,
        run: OrchestrationRun,
        agents: List[Agent],
        tasks: List[OrchestrationTask],
        signals_by_task: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """PRD-164 S2: rank candidate agents per planned task and persist the
        match preview — ``input_context['agent_match']`` on each task row and
        ``match_agent``/``match_reason`` mirrored into the plan snapshot (the
        approval card's source). Explicit agent overrides (PRD-163 S4) rank
        first by construction. Best-effort: a failure here never blocks
        mission creation — the dispatcher re-matches authoritatively anyway.
        """
        try:
            match_by_seq: Dict[int, Dict[str, Any]] = {}
            for task in tasks:
                input_context = task.input_context if isinstance(task.input_context, dict) else {}
                spec = {
                    "agent_role": task.agent_role,
                    "required_tools": input_context.get("required_tools", []),
                }
                ranked = AgentMatcher.rank(
                    db=db, task=task, agents=agents, task_spec=spec,
                    semantic=(signals_by_task or {}).get(task.id),
                )
                if not ranked:
                    continue
                annotation = {**build_match_annotation(ranked), "decided_at": "plan"}
                task.input_context = {**input_context, "agent_match": annotation}
                match_by_seq[int(task.sequence_number)] = annotation

            if match_by_seq:
                run.plan = annotate_plan_with_matches(run.plan, match_by_seq)
        except Exception:
            logger.warning(
                "Match preview annotation failed for run %s (non-fatal)",
                run.id, exc_info=True,
            )

    def import_plan(
        self,
        db: Session,
        workspace_id: UUID,
        goal: str,
        plan: Dict[str, Any],
        created_by: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationRun:
        """PRD-163 S2: create a mission from a pre-built plan WITHOUT re-running the
        planner. The given tasks/dependencies are persisted verbatim and the
        mission lands in awaiting_approval. Used by the plan-import endpoint and by
        Auto when a chat-approved plan should execute exactly as agreed (Q54)."""
        from modules.coordination.planner import (
            DecompositionResult,
            PlannedTask,
            PlannedDependency,
        )

        raw_tasks = plan.get("tasks") or []
        if not raw_tasks:
            raise ValueError("Imported plan has no tasks")

        planned_tasks: List[PlannedTask] = []
        for idx, t in enumerate(raw_tasks):
            temp_id = str(t.get("temp_id") or t.get("id") or f"t{idx + 1}")
            planned_tasks.append(PlannedTask(
                temp_id=temp_id,
                title=str(t.get("title") or f"Task {idx + 1}"),
                description=str(t.get("description") or ""),
                agent_role=str(t.get("agent_role") or "generalist"),
                sequence_number=int(t.get("sequence_number", idx + 1)),
                task_type=str(t.get("task_type") or "execution"),
                verification_criteria=t.get("verification_criteria") or [],
                required_tools=t.get("required_tools") or [],
                dependencies=[str(d) for d in (t.get("dependencies") or [])],
                complexity=str(t.get("complexity") or "moderate"),
                parallel_group=t.get("parallel_group"),
            ))

        valid_ids = {pt.temp_id for pt in planned_tasks}
        deps: List[PlannedDependency] = []
        for d in (plan.get("dependencies") or []):
            frm = str(d.get("from") or d.get("from_task_temp_id") or "")
            to = str(d.get("to") or d.get("to_task_temp_id") or "")
            if frm in valid_ids and to in valid_ids:
                deps.append(PlannedDependency(from_task_temp_id=frm, to_task_temp_id=to))

        token_estimate = int(plan.get("token_estimate") or sum(
            COMPLEXITY_TOKEN_BUDGET.get(pt.complexity, 4000) for pt in planned_tasks
        ))
        decomposition = DecompositionResult(
            tasks=planned_tasks,
            dependencies=deps,
            token_estimate=token_estimate,
            max_concurrent=int(plan.get("max_concurrent", 1)),
        )

        mission_config = {**(config or {}), "imported_plan": True}
        run = OrchestrationRun(
            workspace_id=workspace_id,
            goal=goal,
            created_by=created_by,
            config=mission_config,
            state=RunState.PENDING.value,
            state_type="initial",
            max_retries=mission_config.get("max_retries", Config.COORDINATOR_MAX_TASK_RETRIES),
            max_concurrent=decomposition.max_concurrent,
        )
        db.add(run)
        db.flush()

        emit_event(
            db=db, run_id=run.id, event_type=EventType.RUN_CREATED,
            actor_type=ActorType.HUMAN, actor_id=created_by,
            payload={"goal": goal[:500], "imported": True},
        )
        transition_run(
            db=db, run=run, new_state=RunState.PLANNING,
            actor_type=ActorType.COORDINATOR, actor_id="coordinator",
        )

        temp_id_to_task = self._persist_decomposition(db, run, decomposition)

        # PRD-164 S2: imported plans get the lexical match preview too (this
        # path is sync, so no semantic signals — the dispatcher blends them at
        # dispatch). An imported agent_role naming a roster agent is an
        # explicit override and is flagged as such on the preview.
        roster = (
            db.query(Agent)
            .filter(and_(
                Agent.workspace_id == workspace_id,
                Agent.status == "active",
            ))
            .all()
        )
        self._annotate_match_previews(
            db, run, roster, list(temp_id_to_task.values()), None,
        )

        emit_event(
            db=db, run_id=run.id, event_type=EventType.RUN_PLAN_READY,
            actor_type=ActorType.COORDINATOR, actor_id="coordinator",
            payload={"task_count": len(planned_tasks), "imported": True},
        )
        transition_run(
            db=db, run=run, new_state=RunState.AWAITING_APPROVAL,
            actor_type=ActorType.COORDINATOR, actor_id="coordinator",
        )
        logger.info("Imported plan -> mission %s (%d tasks) awaiting_approval", run.id, len(planned_tasks))
        return run

    def update_mission_plan(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
        task_edits: List[Dict[str, Any]],
    ) -> OrchestrationRun:
        """PRD-163 S4/Q57: apply approval-time task/agent edits to an
        awaiting-approval mission so they persist into execution.

        ``task_edits`` is a list of ``{task_id|temp_id|sequence_number,
        agent_role?, title?, description?}``. Edits mutate the OrchestrationTask
        rows the dispatcher will execute (so an edited ``agent_role`` actually
        changes who runs the task) and are mirrored into ``run.plan``. Field
        edits only — structural changes go through plan-import (Q54). Valid only
        while the mission is awaiting approval.
        """
        run = self._get_run(db, run_id)
        if run.state != RunState.AWAITING_APPROVAL.value:
            raise ValueError(
                f"Mission is in '{run.state}' state, expected 'awaiting_approval'"
            )

        tasks: List[OrchestrationTask] = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .all()
        )
        new_plan, fields_changed = apply_plan_task_edits(tasks, run.plan, task_edits)
        if fields_changed:
            run.plan = new_plan  # reassign so the JSON column is marked dirty
            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.RUN_PLAN_EDITED,
                actor_type=ActorType.HUMAN,
                actor_id=actor_id,
                payload={"fields_changed": fields_changed,
                         "edit_count": len(task_edits or [])},
            )
            logger.info(
                "Mission %s plan edited by %s (%d field(s))",
                run_id, actor_id, fields_changed,
            )
        return run

    def approve_plan(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
    ) -> OrchestrationRun:
        """
        Approve a mission plan and start execution.

        Transitions: awaiting_approval → running.
        Queues initial tasks (those with no dependencies) as QUEUED.
        Approval-time edits are applied beforehand via ``update_mission_plan``
        (PRD-163 S4/Q57), so this executes the plan exactly as shown.
        """
        run = self._get_run(db, run_id)

        transition_run(
            db=db,
            run=run,
            new_state=RunState.RUNNING,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_APPROVED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        self._queue_initial_tasks(db, run)

        # PRD-227 US-002: narrate the start into the launching thread (run-level).
        _n = _mission_task_count(run)
        _narrate_mission(
            db, run,
            f"Mission approved — starting {_n} task{'s' if _n != 1 else ''}: {(run.goal or '')[:120]}",
            level="run", event="run_started",
        )

        logger.info("Mission %s approved by %s → running", run_id, actor_id)
        return run

    # ------------------------------------------------------------------
    # Lifecycle: reject_plan
    # ------------------------------------------------------------------

    def reject_plan(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
        reason: str,
    ) -> OrchestrationRun:
        """Reject a mission plan. Transitions: awaiting_approval → failed."""
        run = self._get_run(db, run_id)

        transition_run(
            db=db,
            run=run,
            new_state=RunState.FAILED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            reason=reason,
            stop_reason="human_cancelled",
            stop_detail=f"Plan rejected: {reason}",
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_REJECTED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            payload={"reason": reason},
        )

        # Skip all pending tasks
        MissionReconciler._skip_remaining_tasks(
            db=db,
            run_id=run.id,
            reason=f"Plan rejected: {reason}",
        )

        VerificationService.clear_cache(run.id)
        logger.info("Mission %s rejected by %s: %s", run_id, actor_id, reason)
        return run

    # ------------------------------------------------------------------
    # Lifecycle: pause / resume / cancel
    # ------------------------------------------------------------------

    def pause_mission(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
    ) -> OrchestrationRun:
        """Pause a running mission. Running tasks continue but no new dispatches."""
        run = self._get_run(db, run_id)

        transition_run(
            db=db,
            run=run,
            new_state=RunState.PAUSED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_PAUSED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        logger.info("Mission %s paused by %s", run_id, actor_id)
        return run

    def resume_mission(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
    ) -> OrchestrationRun:
        """Resume a paused mission.

        If the mission was paused due to budget exceeded, auto-extend the
        budget by 25% so the dispatcher doesn't immediately re-pause.
        """
        run = self._get_run(db, run_id)

        # Auto-extend budget when tokens_used >= 80% of budget (prevents re-pause loop)
        budget = run.token_budget_estimate or 0
        used = run.tokens_used or 0
        if budget > 0 and used >= budget * 0.8:
            new_budget = int(used * 2.0)
            logger.info(
                "Mission %s: auto-extending budget %d → %d (tokens_used=%d)",
                run_id, budget, new_budget, used,
            )
            run.token_budget_estimate = new_budget

        transition_run(
            db=db,
            run=run,
            new_state=RunState.RUNNING,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_RESUMED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            payload={
                "budget_extended": budget > 0 and used >= budget,
                "old_budget": budget,
                "new_budget": run.token_budget_estimate,
                "tokens_used": used,
            },
        )

        logger.info("Mission %s resumed by %s", run_id, actor_id)
        return run

    def cancel_mission(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
    ) -> OrchestrationRun:
        """Cancel a mission. Running tasks continue to completion; no new dispatches."""
        run = self._get_run(db, run_id)

        transition_run(
            db=db,
            run=run,
            new_state=RunState.CANCELLED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            stop_reason="human_cancelled",
            stop_detail=f"Cancelled by user {actor_id}",
        )

        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_CANCELLED,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
        )

        # Skip all pending/queued tasks
        MissionReconciler._skip_remaining_tasks(
            db=db,
            run_id=run.id,
            reason="Mission cancelled",
        )

        VerificationService.clear_cache(run.id)
        # PRD-227 US-002: narrate the cancel into the launching thread (run-level).
        # cancel_mission is an out-of-tick API path, so the tick observer never
        # sees this run terminal — narrate here.
        _narrate_run_terminal(db, run)
        logger.info("Mission %s cancelled by %s", run_id, actor_id)
        return run

    # ------------------------------------------------------------------
    # Lifecycle: replan_mission (PRD-82B US-005)
    # ------------------------------------------------------------------

    async def replan_mission(
        self,
        db: Session,
        run_id: UUID,
        actor_id: str,
        notes: Optional[str] = None,
        *,
        actor_type: ActorType = ActorType.HUMAN,
        trigger: str = "human",
    ) -> OrchestrationRun:
        """
        Replan a mission by generating replacement tasks for the failed (or
        looping) subtree while preserving completed/verified tasks.

        Flow:
          1. Validate: 'failed' state (humans) or RUNNING via the joiner's
             stall-ledger trigger (PRD-164 S4); replan_count < max
          2. Transition → replanning
          3. Gather completed task outputs + identify the failed/looping task
          4. Call planner.replan() to generate replacement tasks
          5. Mark replaced tasks as skipped
          6. Insert new tasks + dependencies
          7. Transition replanning → running
          8. Queue initial new tasks

        Args:
            db: SQLAlchemy session (caller manages transaction).
            run_id: Mission run UUID.
            actor_id: Clerk user ID requesting the replan ('joiner' for the
                coordinator's stall-ledger path).
            notes: Optional guidance for the replanner.
            actor_type: HUMAN for the API/tool path (default), COORDINATOR
                when the PRD-164 S4 joiner replans automatically.
            trigger: 'human' (default) or 'stall_ledger' — recorded on the
                audit events; the stall trigger may replan a RUNNING mission.

        Returns:
            The updated OrchestrationRun.

        Raises:
            ValueError: if run not found or state/replan constraints violated.
            PlanValidationError: if planner cannot produce a valid replan.
        """
        run = self._get_run(db, run_id)

        # Validate state. Humans replan FAILED missions; the joiner's
        # stall-ledger trigger replans a RUNNING mission that is looping.
        allowed_states = (
            {RunState.FAILED, RunState.RUNNING}
            if trigger == "stall_ledger"
            else {RunState.FAILED}
        )
        if RunState(run.state) not in allowed_states:
            raise ValueError(
                f"Mission must be in 'failed' state to replan, "
                f"currently in '{run.state}'"
            )

        # Validate replan count
        current_replans = run.replan_count or 0
        max_replans = Config.COORDINATOR_MAX_REPLANS
        if current_replans >= max_replans:
            raise ValueError(
                f"Mission has been replanned {current_replans} times, "
                f"maximum is {max_replans}"
            )

        # Transition failed/running → replanning
        transition_run(
            db=db,
            run=run,
            new_state=RunState.REPLANNING,
            actor_type=actor_type,
            actor_id=actor_id,
            reason=(
                f"Replan requested via {trigger} "
                f"(attempt {current_replans + 1}/{max_replans})"
            ),
        )

        # Gather completed task outputs
        all_tasks: list[OrchestrationTask] = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run_id)
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        completed_outputs: list[dict] = []
        failed_task_title = "Unknown"
        failed_task_reason = "Unknown"

        for task in all_tasks:
            task_state = TaskState(task.state)
            if task_state == TaskState.VERIFIED:
                completed_outputs.append({
                    "task_id": str(task.id),
                    "title": task.title,
                    "output": task.output or "",
                })
            elif task_state == TaskState.FAILED:
                failed_task_title = task.title
                failed_task_reason = (
                    task.failure_detail
                    or task.failure_reason_code
                    or "Unknown failure"
                )

        # PRD-164 S4: a stall-ledger replan usually has no FAILED task — the
        # problem is a LOOP. Point the planner at the churning task instead.
        if failed_task_title == "Unknown" and trigger == "stall_ledger":
            looping = [
                t for t in all_tasks
                if TaskState(t.state) not in DONE_TASK_STATES
            ]
            if looping:
                loop_task = max(looping, key=lambda t: (t.attempt_number or 0))
                failed_task_title = loop_task.title
                failed_task_reason = (
                    f"Looping without forward progress: stuck in "
                    f"'{loop_task.state}' after {loop_task.attempt_number or 0} "
                    f"attempts (progress-ledger stall)"
                )

        # Load roster agents
        agents: list[Agent] = (
            db.query(Agent)
            .filter(
                and_(
                    Agent.workspace_id == run.workspace_id,
                    Agent.status == "active",
                )
            )
            .all()
        )

        # Call planner to generate replacement tasks
        try:
            decomposition = await MissionPlanner.replan(
                goal=run.goal,
                workspace_id=run.workspace_id,
                agents=agents,
                completed_outputs=completed_outputs,
                failed_task_title=failed_task_title,
                failed_task_reason=failed_task_reason,
                user_notes=notes,
                db=db,  # PRD-164 S1: enables the planning context pack
            )
        except PlanValidationError:
            # Replan failed — transition back to failed
            transition_run(
                db=db,
                run=run,
                new_state=RunState.FAILED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason="Replan validation failed after all retries",
                stop_reason="coordinator_error",
                stop_detail="Replan validation failed after all retries",
            )
            await _store_mission_memory_safe(
                db, run.id, outcome="failed",
                failure_reason="Replan validation failed after all retries",
            )
            raise

        # Mark replaced tasks as skipped. The human path (FAILED mission)
        # replaces failed + not-yet-started tasks; the stall-ledger path also
        # skips the mid-flight states — they ARE the loop being replaced, and
        # leaving them live would run the old plan alongside the new one.
        replaceable = {TaskState.FAILED, TaskState.PENDING, TaskState.QUEUED}
        if trigger == "stall_ledger":
            replaceable |= {
                TaskState.ASSIGNED,
                TaskState.RUNNING,
                TaskState.COMPLETED,
                TaskState.VERIFYING,
                TaskState.STALLED,
                TaskState.RETRYING,
            }
        for task in all_tasks:
            task_state = TaskState(task.state)
            if task_state in replaceable:
                task.failure_reason_code = "replaced_by_replan"
                task.failure_detail = f"Replaced during replan #{current_replans + 1}"
                try:
                    transition_task(
                        db=db,
                        task=task,
                        new_state=TaskState.SKIPPED,
                        actor_type=ActorType.COORDINATOR,
                        actor_id="coordinator",
                        reason=f"Replaced during replan #{current_replans + 1}",
                    )
                    sync_board_status(db, task)
                except (ConflictError, InvalidTransitionError):
                    logger.warning(
                        "Could not skip task %s during replan", task.id,
                        exc_info=True,
                    )

        # Compute new sequence numbers starting after existing max
        max_seq = max(
            (t.sequence_number for t in all_tasks),
            default=0,
        )

        # Insert new tasks
        temp_id_to_task: dict[str, OrchestrationTask] = {}
        for planned in decomposition.tasks:
            new_seq = max_seq + planned.sequence_number
            task = OrchestrationTask(
                run_id=run.id,
                title=planned.title,
                description=planned.description,
                task_type=planned.task_type,
                sequence_number=new_seq,
                agent_role=planned.agent_role,
                state=TaskState.PENDING.value,
                state_type="initial",
                verification_criteria=planned.verification_criteria or None,
                input_context=(
                    {"required_tools": planned.required_tools}
                    if planned.required_tools
                    else None
                ),
                max_retries=run.max_retries,
                complexity=getattr(planned, "complexity", "moderate"),
                parallel_group=getattr(planned, "parallel_group", None),
                estimated_tokens=COMPLEXITY_TOKEN_BUDGET.get(
                    getattr(planned, "complexity", "moderate"), 4000
                ),
            )
            db.add(task)
            db.flush()
            temp_id_to_task[planned.temp_id] = task

            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.TASK_CREATED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                task_id=task.id,
                payload={
                    "title": planned.title,
                    "sequence_number": new_seq,
                    "agent_role": planned.agent_role,
                    "replan": True,
                },
            )

        # Insert dependency edges for new tasks
        for dep in decomposition.dependencies:
            from_task = temp_id_to_task.get(dep.from_task_temp_id)
            to_task = temp_id_to_task.get(dep.to_task_temp_id)
            if from_task and to_task:
                dep_row = OrchestrationTaskDependency(
                    task_id=to_task.id,
                    depends_on_task_id=from_task.id,
                )
                db.add(dep_row)

        db.flush()

        # Create board tasks for new tasks
        try:
            for task in temp_id_to_task.values():
                create_task_board_task(db, run, task)
        except Exception:
            logger.warning(
                "Failed to create board tasks during replan for mission %s",
                run.id,
                exc_info=True,
            )

        # Update replan count
        run.replan_count = current_replans + 1

        # Update the plan JSONB with replan info
        run.plan = {
            **(run.plan or {}),
            f"replan_{current_replans + 1}": {
                "tasks": [
                    {
                        "temp_id": t.temp_id,
                        "title": t.title,
                        "description": t.description,
                        "agent_role": t.agent_role,
                        "sequence_number": t.sequence_number,
                        "task_type": t.task_type,
                    }
                    for t in decomposition.tasks
                ],
                "dependencies": [
                    {"from": d.from_task_temp_id, "to": d.to_task_temp_id}
                    for d in decomposition.dependencies
                ],
                "user_notes": notes,
            },
        }
        run.token_budget_estimate = (
            (run.token_budget_estimate or 0) + decomposition.token_estimate
        )

        # Emit replanned event
        emit_event(
            db=db,
            run_id=run.id,
            event_type=EventType.RUN_REPLANNED,
            actor_type=actor_type,
            actor_id=actor_id,
            payload={
                "replan_number": current_replans + 1,
                "new_task_count": len(decomposition.tasks),
                "token_estimate": decomposition.token_estimate,
                "user_notes": notes,
                "trigger": trigger,
            },
        )

        # Transition replanning → running
        transition_run(
            db=db,
            run=run,
            new_state=RunState.RUNNING,
            actor_type=ActorType.COORDINATOR,
            actor_id="coordinator",
            reason=f"Replan #{current_replans + 1} complete — resuming",
        )

        # Queue initial new tasks
        self._queue_initial_tasks(db, run)

        logger.info(
            "Mission %s replanned (#%d) by %s — %d new tasks",
            run_id,
            current_replans + 1,
            actor_id,
            len(decomposition.tasks),
        )

        return run

    # ------------------------------------------------------------------
    # Output summary builder (PRD-82A Section 6)
    # ------------------------------------------------------------------

    @staticmethod
    def build_output_summary(
        db: Session,
        run: OrchestrationRun,
    ) -> Dict[str, Any]:
        """
        Build the mission output summary from verified task results.

        This is a structured aggregation — no LLM call needed.
        Stored in run.output_summary JSONB.
        """
        tasks: List[OrchestrationTask] = (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        now = datetime.now(timezone.utc)
        started = run.started_at
        total_duration = (
            (now - started).total_seconds()
            if started
            else 0
        )

        verified_count = sum(
            1 for t in tasks if TaskState(t.state) == TaskState.VERIFIED
        )
        failed_count = sum(
            1 for t in tasks if TaskState(t.state) == TaskState.FAILED
        )

        task_summaries = []
        for task in tasks:
            # Get agent name if assigned
            agent_name = None
            if task.assigned_agent_id:
                agent = db.query(Agent).get(task.assigned_agent_id)
                if agent:
                    agent_name = agent.name

            output_excerpt = ""
            if task.output:
                output_excerpt = task.output[:500]

            task_summaries.append({
                "sequence": task.sequence_number,
                "title": task.title,
                "agent": agent_name,
                "verdict": (
                    "pass" if TaskState(task.state) == TaskState.VERIFIED
                    else "fail" if TaskState(task.state) == TaskState.FAILED
                    else TaskState(task.state).value
                ),
                "output_excerpt": output_excerpt,
                "tokens_used": task.tokens_used or 0,
            })

        config = run.config or {}
        summary: Dict[str, Any] = {
            "goal": run.goal,
            "tasks_completed": verified_count,
            "tasks_failed": failed_count,
            "total_duration_seconds": round(total_duration),
            "task_summaries": task_summaries,
            "generated_at": now.isoformat(),
        }

        # Include template and artifact metadata for frontend rendering
        template_used = config.get("template_used")
        if template_used:
            summary["template_used"] = template_used
        app_bundle_doc_id = config.get("app_bundle_document_id")
        if app_bundle_doc_id:
            summary["app_bundle_document_id"] = app_bundle_doc_id

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_run(db: Session, run_id: UUID) -> OrchestrationRun:
        """Load an OrchestrationRun by ID. Raises ValueError if not found."""
        run = db.query(OrchestrationRun).get(run_id)
        if run is None:
            raise ValueError(f"Mission run not found: {run_id}")
        return run

    @staticmethod
    def _queue_initial_tasks(
        db: Session,
        run: OrchestrationRun,
    ) -> int:
        """
        Queue tasks that have no dependencies (roots of the DAG).

        Transitions root tasks from PENDING → QUEUED so the dispatcher
        picks them up on the next tick.

        Returns count of tasks queued.
        """
        ready = DependencyResolver.get_ready_tasks(db, run.id)
        queued = 0

        for task in ready:
            try:
                transition_task(
                    db=db,
                    task=task,
                    new_state=TaskState.QUEUED,
                    actor_type=ActorType.COORDINATOR,
                    actor_id="coordinator",
                )
                sync_board_status(db, task)
                queued += 1
            except (ConflictError, InvalidTransitionError):
                logger.warning(
                    "Could not queue task %s", task.id, exc_info=True,
                )

        if queued:
            logger.info(
                "Queued %d initial tasks for mission %s", queued, run.id,
            )
        return queued

    @staticmethod
    def _requeue_task_with_feedback(
        db: Session,
        task: OrchestrationTask,
        feedback: str,
        actor_id: str,
    ) -> None:
        """
        Re-queue a verified/failed task with human feedback for retry.

        Injects feedback into task.input_context so the agent sees it
        on the next execution.
        """
        # Inject feedback — immutable replace for JSONB dirty detection
        task.input_context = {
            **(task.input_context or {}),
            "retry_feedback": feedback,
        }

        task.failure_reason_code = FailureReasonCode.VERIFICATION_REJECT.value
        task.attempt_number = (task.attempt_number or 0) + 1

        # Transition verified/failed → retrying → assigned pipeline
        try:
            transition_task(
                db=db,
                task=task,
                new_state=TaskState.RETRYING,
                actor_type=ActorType.HUMAN,
                actor_id=actor_id,
                reason=f"Human rejected with feedback: {feedback[:200]}",
            )
            transition_task(
                db=db,
                task=task,
                new_state=TaskState.ASSIGNED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason="Re-dispatching with human feedback",
            )
            sync_board_status(db, task)
        except (ConflictError, InvalidTransitionError):
            logger.warning(
                "Could not requeue task %s with feedback",
                task.id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Archival (PRD-82B US-009)
    # ------------------------------------------------------------------

    async def _maybe_compact_fields(self, db: Session, summary: Dict[str, Any]) -> None:
        """PRD-166 S1 / PRD-178 S3 (F063): retention/compaction for field memory —
        prune dead patterns so the shared Qdrant collection stays bounded as
        workspace fields compound across missions. Throttled to once per hour;
        errors never affect the rest of the tick. The prune decision is the
        unit-tested ``field_scoring.is_prunable``.

        F063 fix: the sweep is **workspace-scoped** (each workspace's points are
        compacted under its own filter, never a full unscoped re-scan) and
        **resumable** — the Qdrant scroll cursor is persisted per workspace so
        the next run continues where this one stopped instead of re-scanning
        compacted entries. ``next_offset=None`` from a completed pass clears the
        cursor so the following run starts fresh."""
        now = datetime.now(timezone.utc)
        if (
            self._last_field_compaction_at is not None
            and (now - self._last_field_compaction_at).total_seconds() < 3600
        ):
            return
        self._last_field_compaction_at = now

        field = self._get_field()
        inner = getattr(field, "_inner", field) if field else None
        if inner is None or not hasattr(inner, "compact"):
            return

        from modules.context.compaction_cursor import (
            load_compaction_cursor,
            save_compaction_cursor,
        )

        # Workspaces that have accumulated field data — scope compaction to each.
        try:
            workspace_ids = [
                str(row[0]) for row in db.execute(
                    text(
                        "SELECT DISTINCT workspace_id FROM orchestration_runs "
                        "WHERE config->>'field_id' IS NOT NULL"
                    )
                ).fetchall()
                if row[0] is not None
            ]
        except Exception:
            logger.warning("[Coordinator] Field compaction workspace scan failed", exc_info=True)
            return

        total_pruned = 0
        for ws_id in workspace_ids:
            try:
                cursor = load_compaction_cursor(db, ws_id)
                result = await inner.compact(workspace_id=ws_id, resume_offset=cursor)
                save_compaction_cursor(db, ws_id, result.next_offset)
                db.commit()
                total_pruned += result.pruned
            except Exception:
                db.rollback()
                logger.warning(
                    "[Coordinator] Field compaction failed for ws=%s", ws_id, exc_info=True,
                )
        if total_pruned:
            summary["field_pruned"] = total_pruned
            logger.info("[Coordinator] Field compaction pruned %d pattern(s)", total_pruned)

    def _maybe_archive(self, db: Session, summary: Dict[str, Any]) -> None:
        """
        Run archive_old_runs() if at least 1 hour since last archive attempt.

        Throttled to avoid running every 5s tick. Errors are logged and
        do not affect the rest of the tick.
        """
        now = datetime.now(timezone.utc)
        if (
            self._last_archive_at is not None
            and (now - self._last_archive_at).total_seconds() < 3600
        ):
            return

        self._last_archive_at = now

        try:
            archived = self.archive_old_runs(db)
            if archived > 0:
                db.commit()
                logger.info("[Coordinator] Archived %d old runs", archived)
                summary["archived"] = archived
        except Exception:
            db.rollback()
            logger.error(
                "[Coordinator] Archive failed", exc_info=True,
            )

    @staticmethod
    def archive_old_runs(
        db: Session,
        days: Optional[int] = None,
    ) -> int:
        """
        Archive terminal runs older than ``days`` (default from config).

        For each eligible run:
          1. Serialize run + tasks + events + dependencies to JSONB snapshot
          2. Insert into orchestration_archive
          3. Delete from active tables (CASCADE handles tasks/events/deps)

        Args:
            db: SQLAlchemy session (caller manages transaction).
            days: Retention period in days. Defaults to config value.

        Returns:
            Number of runs archived.
        """
        retention_days = days if days is not None else Config.COORDINATOR_ARCHIVE_AFTER_DAYS
        batch_size = Config.COORDINATOR_ARCHIVE_BATCH_SIZE
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        # Find terminal runs older than cutoff
        terminal_state_values = [s.value for s in TERMINAL_RUN_STATES]
        old_runs: List[OrchestrationRun] = (
            db.query(OrchestrationRun)
            .filter(
                OrchestrationRun.state.in_(terminal_state_values),
                OrchestrationRun.updated_at < cutoff,
            )
            .order_by(OrchestrationRun.updated_at.asc())
            .limit(batch_size)
            .all()
        )

        if not old_runs:
            return 0

        archived_count = 0
        for run in old_runs:
            run_id = run.id

            # Check not already archived (idempotent)
            existing = (
                db.query(OrchestrationArchive.id)
                .filter(OrchestrationArchive.original_run_id == run_id)
                .first()
            )
            if existing:
                # Already archived — just delete the active row
                db.delete(run)
                archived_count += 1
                continue

            # Load related data for snapshot
            tasks: List[OrchestrationTask] = (
                db.query(OrchestrationTask)
                .filter(OrchestrationTask.run_id == run_id)
                .order_by(OrchestrationTask.sequence_number)
                .all()
            )

            events: List[OrchestrationEvent] = (
                db.query(OrchestrationEvent)
                .filter(OrchestrationEvent.run_id == run_id)
                .order_by(OrchestrationEvent.created_at)
                .all()
            )

            task_ids = [t.id for t in tasks]
            dependencies: List[OrchestrationTaskDependency] = []
            if task_ids:
                dependencies = (
                    db.query(OrchestrationTaskDependency)
                    .filter(OrchestrationTaskDependency.task_id.in_(task_ids))
                    .all()
                )

            # Build JSONB snapshot
            archive_data = {
                "run": {
                    "id": str(run.id),
                    "workspace_id": str(run.workspace_id),
                    "goal": run.goal,
                    "plan": run.plan,
                    "config": run.config,
                    "state": run.state,
                    "state_type": run.state_type,
                    "created_by": run.created_by,
                    "assigned_coordinator_id": run.assigned_coordinator_id,
                    "output_summary": run.output_summary,
                    "token_budget_estimate": run.token_budget_estimate,
                    "tokens_used": run.tokens_used,
                    "max_retries": run.max_retries,
                    "max_concurrent": run.max_concurrent,
                    "replan_count": run.replan_count,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "updated_at": run.updated_at.isoformat() if run.updated_at else None,
                },
                "tasks": [
                    {
                        "id": str(t.id),
                        "title": t.title,
                        "description": t.description,
                        "task_type": t.task_type,
                        "sequence_number": t.sequence_number,
                        "agent_role": t.agent_role,
                        "state": t.state,
                        "assigned_agent_id": t.assigned_agent_id,
                        "verification_criteria": t.verification_criteria,
                        "input_context": t.input_context,
                        "output": t.output,
                        "output_metadata": t.output_metadata,
                        "failure_reason_code": t.failure_reason_code,
                        "failure_detail": t.failure_detail,
                        "attempt_number": t.attempt_number,
                        "tokens_used": t.tokens_used,
                        "started_at": t.started_at.isoformat() if t.started_at else None,
                        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                        "created_at": t.created_at.isoformat() if t.created_at else None,
                    }
                    for t in tasks
                ],
                "events": [
                    {
                        "id": str(e.id),
                        "task_id": str(e.task_id) if e.task_id else None,
                        "event_type": e.event_type,
                        "actor_type": e.actor_type,
                        "actor_id": e.actor_id,
                        "old_state": e.old_state,
                        "new_state": e.new_state,
                        "payload": e.payload,
                        "created_at": e.created_at.isoformat() if e.created_at else None,
                    }
                    for e in events
                ],
                "dependencies": [
                    {
                        "id": str(d.id),
                        "task_id": str(d.task_id),
                        "depends_on_task_id": str(d.depends_on_task_id),
                        "trigger_rule": d.trigger_rule,
                    }
                    for d in dependencies
                ],
            }

            # Insert archive row
            archive = OrchestrationArchive(
                original_run_id=run_id,
                goal=run.goal or "",
                state=run.state,
                workspace_id=run.workspace_id,
                created_by=run.created_by,
                created_at=run.created_at,
                completed_at=run.completed_at,
                archive_data=archive_data,
            )
            db.add(archive)

            # Delete from active tables (CASCADE handles tasks/events/deps)
            db.delete(run)
            archived_count += 1

        return archived_count

    async def _complete_verified_run(
        self,
        db: Session,
        run: OrchestrationRun,
    ) -> None:
        """
        Build output summary, run cross-task consistency check, and
        auto-complete the mission. Consistency check is informational only —
        results are stored in output_summary for the user to review at leisure.
        """
        try:
            summary = self.build_output_summary(db, run)
            run.output_summary = summary

            # --- Cross-task consistency verification (PRD-82B US-006) ---
            # Informational only — does NOT gate completion
            consistency_result = await self._run_consistency_check(db, run)
            if consistency_result is not None:
                summary["consistency"] = {
                    "passed": consistency_result.passed,
                    "score": consistency_result.score,
                    "reasoning": consistency_result.reasoning,
                    "issues": [
                        {
                            "task_ids": issue.task_ids,
                            "description": issue.description,
                            "severity": issue.severity,
                        }
                        for issue in consistency_result.issues
                    ],
                }
                run.output_summary = summary

                # Track token usage from consistency check
                if consistency_result.tokens_used > 0:
                    run.tokens_used = (run.tokens_used or 0) + consistency_result.tokens_used

            # Auto-complete — all tasks passed verification, work is done
            transition_run(
                db=db,
                run=run,
                new_state=RunState.COMPLETED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                reason="All tasks verified — mission complete",
                stop_reason="completed",
                stop_detail="All tasks verified successfully",
            )

            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.RUN_COMPLETED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                payload={
                    "tasks_completed": summary["tasks_completed"],
                    "total_duration_seconds": summary["total_duration_seconds"],
                    "consistency": summary.get("consistency"),
                },
            )

            # PRD-128: dispatch mission_complete on terminal transition
            # --- Mission cost summary (audit log) ---
            mission_cost = _log_mission_cost_summary(db, run)

            await _dispatch_mission_event(
                db=db,
                run=run,
                event_type="mission_complete",
                title=f"Mission complete: {(run.goal or 'Mission')[:120]}",
                message=(
                    f"{summary['tasks_completed']} tasks verified in "
                    f"{summary['total_duration_seconds']}s"
                    f" | est. cost ${mission_cost:.2f}"
                ),
                status="ok",
            )

            # Clean up verification cache
            VerificationService.clear_cache(run.id)

            logger.info(
                "Mission %s → completed (summary: %d tasks, %ds, consistency=%s)",
                run.id,
                summary["tasks_completed"],
                summary["total_duration_seconds"],
                "pass" if (consistency_result and consistency_result.passed) else "issues",
            )

            # PRD-131d Phase 1: persist mission summary to memory (success)
            await _store_mission_memory_safe(db, run.id, outcome="completed")

        except Exception:
            logger.error(
                "Failed to build output summary for run %s",
                run.id,
                exc_info=True,
            )

    async def _run_consistency_check(
        self,
        db: Session,
        run: OrchestrationRun,
    ) -> Optional[ConsistencyResult]:
        """
        Run cross-task consistency verification if enabled.

        Returns ConsistencyResult or None if disabled/skipped.
        """
        if not Config().COORDINATOR_CONSISTENCY_CHECK:
            logger.info(
                "Consistency check disabled for run %s (COORDINATOR_CONSISTENCY_CHECK=false)",
                run.id,
            )
            return None

        # Gather verified task outputs
        tasks: List[OrchestrationTask] = (
            db.query(OrchestrationTask)
            .filter(
                and_(
                    OrchestrationTask.run_id == run.id,
                    OrchestrationTask.state == TaskState.VERIFIED.value,
                )
            )
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        task_outputs = [
            {
                "task_id": str(t.id),
                "title": t.title or "",
                "output": t.output or "",
            }
            for t in tasks
        ]

        if len(task_outputs) < 2:
            return None

        verification_service = VerificationService()

        try:
            result = await verification_service.verify_cross_task_consistency(
                run_id=run.id,
                goal=run.goal or "",
                task_outputs=task_outputs,
            )

            # Emit consistency event
            emit_event(
                db=db,
                run_id=run.id,
                event_type=EventType.CONSISTENCY_CHECKED,
                actor_type=ActorType.COORDINATOR,
                actor_id="coordinator",
                payload={
                    "passed": result.passed,
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "issue_count": len(result.issues),
                    "high_severity_count": sum(
                        1 for i in result.issues if i.severity == "high"
                    ),
                    "tokens_used": result.tokens_used,
                },
            )

            return result

        except Exception:
            logger.error(
                "Consistency check failed for run %s — proceeding without",
                run.id,
                exc_info=True,
            )
            return None


# ---------------------------------------------------------------------------
# Singleton accessor (matches HeartbeatService pattern)
# ---------------------------------------------------------------------------

_coordinator_service: Optional[CoordinatorService] = None


def get_coordinator_service() -> CoordinatorService:
    """Get or create the singleton CoordinatorService."""
    global _coordinator_service
    if _coordinator_service is None:
        _coordinator_service = CoordinatorService()
    return _coordinator_service
