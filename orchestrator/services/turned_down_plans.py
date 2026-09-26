"""F171 (night 5, B34) — a plan the owner turned down informs the next one.

Mission e0633a0d took four plans. Each turn-down kept its reason (the run's
RUN_REJECTED event), but the next plan never saw it. Auto made a new mission
from its own one-line summary. The planner's chat context (the last five
messages, 500 characters each) no longer held the owner's list of changes.
The planner of a mission born in a chat now reads the plans that chat just
turned down: the owner's reason for each, and what each plan had.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models.orchestration import OrchestrationEvent, OrchestrationRun
from core.models.orchestration_enums import EventType

# The newest turn-downs of one conversation, back to its last plan that was not turned down.
TURNED_DOWN_MAX = 3
TURNED_DOWN_LOOKBACK = 10
REASON_MAX_CHARS = 2000
STEPS_SHOWN_MAX = 12
TURNED_DOWN_HEADER = "## Plans the owner turned down"
TURNED_DOWN_INTRO = (
    "The owner turned down the plan(s) below for this same conversation. Each reason is "
    "an instruction: the new plan must act on it, and must not repeat what was turned down."
)


@dataclass(frozen=True)
class TurnedDownPlan:
    run_id: str
    reason: str
    steps: Tuple[str, ...]


def _steps(plan: Any) -> Tuple[str, ...]:
    tasks = plan.get("tasks") if isinstance(plan, dict) else None
    lines: List[str] = []
    for task in tasks if isinstance(tasks, list) else []:
        if not isinstance(task, dict):
            continue
        who = task.get("match_agent") or task.get("agent_role") or "any agent"
        lines.append(f"{task.get('title') or 'Untitled step'} ({who})")
    return tuple(lines[:STEPS_SHOWN_MAX])


def turned_down_before(db: Session, run: OrchestrationRun) -> List[TurnedDownPlan]:
    """The plans ``run``'s conversation turned down just before it, newest first.
    Empty for a mission no chat started."""
    chat_id = (run.config or {}).get("origin_chat_id")
    if not chat_id:
        return []
    # A run planned right after it was created is flushed, not refreshed: its
    # server-set created_at is not on the object yet, and is this transaction's now().
    created = run.created_at if run.created_at is not None else func.now()
    earlier = (
        db.query(OrchestrationRun)
        .filter(
            OrchestrationRun.workspace_id == run.workspace_id,
            OrchestrationRun.id != run.id,
            OrchestrationRun.created_at <= created,
            OrchestrationRun.config["origin_chat_id"].astext == str(chat_id),
        )
        .order_by(OrchestrationRun.created_at.desc())
        .limit(TURNED_DOWN_LOOKBACK)
        .all()
    )
    if not earlier:
        return []
    reasons = {
        run_id: (payload or {}).get("reason")
        for run_id, payload in db.query(OrchestrationEvent.run_id, OrchestrationEvent.payload).filter(
            OrchestrationEvent.run_id.in_([r.id for r in earlier]),
            OrchestrationEvent.event_type == EventType.RUN_REJECTED.value,
        )
    }
    plans: List[TurnedDownPlan] = []
    for earlier_run in earlier:
        if earlier_run.id not in reasons:
            break  # the conversation's last plan that was not turned down
        reason = str(reasons[earlier_run.id] or "no reason given").strip()[:REASON_MAX_CHARS]
        plans.append(TurnedDownPlan(str(earlier_run.id), reason, _steps(earlier_run.plan)))
        if len(plans) >= TURNED_DOWN_MAX:
            break
    return plans


def turned_down_block(plans: Sequence[TurnedDownPlan]) -> str:
    """The planner prompt's section on them; empty when there are none."""
    if not plans:
        return ""
    parts = [TURNED_DOWN_HEADER, TURNED_DOWN_INTRO]
    for plan in plans:
        parts.append(f"\n### Turned down: plan {plan.run_id[:8]}")
        parts.append(f"The owner's reason:\n<owner_reason>\n{plan.reason}\n</owner_reason>")
        if plan.steps:
            parts.append("It had:\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(plan.steps, 1)))
    return "\n".join(parts) + "\n"
