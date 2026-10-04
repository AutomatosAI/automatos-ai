"""A brief Auto edits reaches the step it names, its card too, or is refused saying why (night 9: F308).

Night 9: "the Harbour Log intro is 85 to 95 words, not 100 to 150 - it's in the brand
voice doc" became platform_update_mission_plan {mission_id: 27, task_edits: [{task_id:
"harbour_log_intro", description: "… between 85 and 95 words …"}]} while #0027 waited
for approval. The plan and the step took it, but the step's card #1876 kept
"approximately 100-150 words": a mission's cards are made with its plan, and an edit
never reached them. Once a mission had started, every edit was refused with
"Mission is in 'running' state, expected 'awaiting_approval'".

Now (``plan_edits.reads_the_plan_edits`` calls this with the steps the edits name):
- an edit of a plan awaiting approval puts each step's new title and brief on its card;
- an edit of a mission that has started reaches each step it names that hasn't started
  (pending or queued): the step works from the new brief when it starts, and its card
  shows it;
- an edit of a step that has started or finished, or of a started step's agent, is
  refused, naming the step's state and the call that redoes it with the owner's words
  (platform_reject_mission with step and reason). Nothing is changed then.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
Steps = List[Tuple[Any, Optional[str]]]

PLAN_STATES = ("pending", "planning", "awaiting_approval")
NOT_STARTED = ("pending", "queued")
AGENT_FIELDS = ("agent_id", "agent_role")
BRIEF_FIELDS = ("title", "description")
STEP_CARD = "orchestration_task"
TITLE_CHARS = 500  # a card's title column

EDITED = ("{labels} of mission {mission} changed: each works from its new brief when it starts, and its card "
          "shows it.")
HAS_STARTED = "{label} has started ({state}), so its brief can't change under it."
BEING_STARTED = "{label} is being started right now, so its brief can't change under it."
AGENT_KEPT = "{label}'s agent can't change once its mission has started."
REDO = (" To have it redone with the owner's new words: platform_reject_mission {{mission_id: \"{mission}\", "
        "step: \"{label}\", reason: the owner's words}}.")
NOTHING_CHANGED = " Nothing was changed."


async def edits_reach_the_steps(handler: Handler, db: Session, workspace_id: Any, params: Dict[str, Any],
                                run: Any, named: List[Dict[str, Any]], steps: Steps) -> Dict[str, Any]:
    """platform_update_mission_plan with its edits named by step id (see the module)."""
    if not named:
        return await handler(db, workspace_id, params)
    if run.state not in PLAN_STATES:
        return edit_started_steps(db, workspace_id, run, named, steps)
    out = await handler(db, workspace_id, params)
    if isinstance(out, dict) and out.get("success"):
        _onto_the_cards(db, workspace_id, named)
    return out


def edit_started_steps(db: Session, workspace_id: Any, run: Any, named: List[Dict[str, Any]],
                       steps: Steps) -> Dict[str, Any]:
    """The edits of a started mission's steps: all of them land, on steps that haven't
    started, or none does and the answer says why for each."""
    from services.coordinator_service import apply_plan_task_edits

    by_id = {str(task.id): (task, number) for task, number in steps}
    mission = _mission(db, workspace_id, run)
    refused = [why for why in (_why_not(by_id[edit["task_id"]], edit, mission) for edit in named) if why]
    if not refused:
        refused = _being_started(db, [by_id[edit["task_id"]] for edit in named], mission)
    if refused:
        db.commit()  # nothing of the call changed; the rows it read under lock are let go
        return {"success": False, "mission_id": str(run.id), "error": " ".join(refused) + NOTHING_CHANGED}
    briefs = [{key: value for key, value in edit.items() if key == "task_id" or key in BRIEF_FIELDS} for edit in named]
    run.plan, _changed = apply_plan_task_edits([task for task, _ in steps], run.plan, briefs)
    _onto_the_cards(db, workspace_id, briefs)
    db.commit()
    labels = ", ".join(_label(*by_id[edit["task_id"]]) for edit in named)
    return {"success": True, "mission_id": str(run.id), "state": run.state,
            "message": EDITED.format(labels=labels[0].upper() + labels[1:], mission=mission)}


def _why_not(step: Tuple[Any, Optional[str]], edit: Dict[str, Any], mission: str) -> Optional[str]:
    """Why this edit can't land on a started mission's step, or None."""
    task, number = step
    label = _label(task, number)
    if any(edit.get(field) not in (None, "") for field in AGENT_FIELDS):
        return AGENT_KEPT.format(label=label[0].upper() + label[1:])
    if task.state not in NOT_STARTED:
        said = HAS_STARTED.format(label=label[0].upper() + label[1:], state=task.state)
        return said + REDO.format(mission=mission, label=number or task.sequence_number)
    return None


def _being_started(db: Session, steps: Steps, mission: str) -> List[str]:
    """Each step the dispatcher holds or has started since it was read: re-read under a
    row lock that never waits (F105), so a brief never changes under a starting step."""
    from core.models.orchestration import OrchestrationTask

    ids = [task.id for task, _ in steps]
    held = {task.id: task for task in db.query(OrchestrationTask).filter(OrchestrationTask.id.in_(ids))
            .with_for_update(skip_locked=True).populate_existing().all()}
    refused = []
    for task, number in steps:
        label = _label(task, number)
        if task.id not in held:
            refused.append(BEING_STARTED.format(label=label[0].upper() + label[1:]))
        elif held[task.id].state not in NOT_STARTED:
            refused.append(_why_not((held[task.id], number), {}, mission))
    return refused


def _onto_the_cards(db: Session, workspace_id: Any, edits: List[Dict[str, Any]]) -> None:
    """Each edited step's new title and brief on its card, where the board shows them."""
    from core.models.core import BoardTask

    for edit in edits:
        fields = {key: edit[key] for key in BRIEF_FIELDS if edit.get(key) not in (None, "")}
        if not fields:
            continue
        card = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == STEP_CARD,
                                          BoardTask.orchestration_task_id == UUID(str(edit["task_id"]))).first()
        if card is None:
            continue
        if "title" in fields:
            card.title = str(fields["title"])[:TITLE_CHARS]
        if "description" in fields:
            card.description = str(fields["description"])
    db.flush()


def _label(task: Any, number: Optional[str]) -> str:
    return f"step {number}" if number else f"step {task.sequence_number}"


def _mission(db: Session, workspace_id: Any, run: Any) -> str:
    from modules.tools.discovery.mission_step_verdicts import _mission_label

    return _mission_label(db, workspace_id, run)


__all__ = ["edit_started_steps", "edits_reach_the_steps"]
