"""Auto's plan edits land on the mission's real steps (night 8: F261, F282, F287).

Night 8, "I've updated the mission plan" came five times when nothing had changed.
Auto's edits named steps the plan doesn't have: a made-up id (4b6110f2-…, #0352),
"email_draft", "0410-1" and "0410-2", "0454-01", or 3 for #0428's email step; the
coordinator skips an edit it can't match, without a word, and the tool answered
"plan updated". Asked to "switch on the check for each step", Auto sent
{add_approval_gate: true}, {approval_required: true} and {wait_for_me: true} on each
step, and {"plan_updates": {"steps.*.approval_required": true}}: none of them is a
setting the plan reads, so #0410 and #0454 ran start to finish unseen.

- A step is named by its card's number (#0352.2, 0352.2, 352.2), by "0410-1", by its
  place in the plan, by its id, or by words of its title only it has; each edit is
  rewritten to the step's id. An edit that names no step is refused, listing them.
- An edit, or the call itself, that asks for a step to wait for the owner switches on
  the mission's check of every step (``check_each_step``), the one setting the mission
  reads, and the answer says so.
- Edits wrapped as {"changes": {...}} or {"updates": {...}} are unwrapped, and an
  agent named under assigned_agent_name, agent_name or agent is pinned by name.

Night 9 (F308): #0027's step 2 took "85 and 95 words" and its card #1876 kept "100-150":
an edit now reaches the step's card too, and a started mission's steps that haven't
started (``step_brief_edits``).
"""
from __future__ import annotations

import functools
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from modules.tools.discovery.step_brief_edits import edits_reach_the_steps

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

CHECK_EACH_STEP = "check_each_step"
STEP_KEYS = ("task_id", "temp_id", "sequence_number", "step", "step_number", "step_id", "id", "ref")
WRAPPERS = ("changes", "updates", "fields", "edits")
AGENT_KEYS = ("assigned_agent_name", "agent_name", "agent", "assignee")
# An edit's way of saying "this step waits for the owner".
WAIT_KEYS = ("add_approval_gate", "approval_gate", "approval_required", "requires_approval", "require_approval",
             "wait_for_me", "review_required", "requires_review", "check", CHECK_EACH_STEP, "pause_after",
             "pause_before", "human_review", "needs_approval")
FIELDS = ("agent_id", "agent_role", "title", "description")
FINISHED = ("completed", "failed", "cancelled")
NO_SUCH_STEP = ("No step {said} in this mission, so nothing was changed. Its steps: {steps}. Name each step by its "
                "number as the board shows it (e.g. {example}).")
CHECKS_ON = ("Every step of this mission now waits in Review for the owner's check before the next one starts "
             "(check_each_step is on).")
CHECKS_TOO_LATE = "This mission has finished, so there are no steps left to check; nothing was changed."


def reads_the_plan_edits(handler: Handler) -> Handler:
    """platform_update_mission_plan's edits, read as the steps they name (see the module)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        params = params or {}
        run = _run(db, workspace_id, params.get("mission_id"))
        if run is None:
            return await handler(db, workspace_id, params)
        edits = [_unwrapped(edit) for edit in params.get("task_edits") or [] if isinstance(edit, dict)]
        checks = _says_wait(params) or any(_says_wait(edit) for edit in edits)
        edits = [edit for edit in edits if any(edit.get(field) not in (None, "") for field in FIELDS)]
        steps = _steps(db, workspace_id, run)
        named, unknown = _named(edits, steps)
        if unknown:
            return {"success": False, "error": _no_such_step(unknown, steps)}
        if checks and str(run.state) in FINISHED:
            return {"success": False, "error": CHECKS_TOO_LATE}
        note = _check_each_step(db, run) if checks else None
        if not named and note:
            return {"success": True, "mission_id": str(run.id), "state": run.state, "message": note,
                    "checks_each_step": True}
        out = await edits_reach_the_steps(handler, db, workspace_id,  # F308: the step and its card, or why not
                                          {**params, "task_edits": named} if named else params, run, named, steps)
        return {**out, "check_note": note, "checks_each_step": True} if note and isinstance(out, dict) else out
    return wrapped


def _run(db: Session, workspace_id: Any, mission_id: Any) -> Any:
    from core.models.orchestration import OrchestrationRun

    try:
        run_id = UUID(str(mission_id))
    except (ValueError, TypeError):
        return None
    return db.query(OrchestrationRun).filter(OrchestrationRun.id == run_id,
                                             OrchestrationRun.workspace_id == workspace_id).first()


def _unwrapped(edit: Dict[str, Any]) -> Dict[str, Any]:
    """{"task_id": …, "changes": {"description": …}} as {"task_id": …, "description": …},
    with an agent's name where the plan reads it."""
    flat = {k: v for k, v in edit.items() if k not in WRAPPERS}
    for key in WRAPPERS:
        if isinstance(edit.get(key), dict):
            flat = {**edit[key], **flat}
    named = next((flat[key] for key in AGENT_KEYS if flat.get(key)), None)
    flat = {k: v for k, v in flat.items() if k not in AGENT_KEYS}
    return {**flat, "agent_role": named} if named and not flat.get("agent_role") else flat


def _says_wait(edit: Dict[str, Any]) -> bool:
    """Whether a call or an edit asks for a step to wait for the owner, in any key or
    in a {"plan_updates": {"steps.*.approval_required": true}} style map."""
    for key, value in edit.items():
        if any(word in str(key) for word in WAIT_KEYS) and value not in (False, None, "", "false", "no", 0):
            return True
        if isinstance(value, dict) and _says_wait(value):
            return True
    return False


def _steps(db: Session, workspace_id: Any, run: Any) -> List[Tuple[Any, Optional[str]]]:
    """The mission's steps in order, each with its card's number."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationTask
    from services.ticket_numbers import ticket_numbers

    tasks = (db.query(OrchestrationTask).filter(OrchestrationTask.run_id == run.id)
             .order_by(OrchestrationTask.sequence_number).all())
    cards = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id,
                                       BoardTask.orchestration_task_id.in_([t.id for t in tasks])).all() if tasks else []
    numbers = ticket_numbers(db, workspace_id, cards)
    by_task = {card.orchestration_task_id: numbers.get(card.id) for card in cards}
    return [(task, by_task.get(task.id)) for task in tasks]


def _named(edits: List[Dict[str, Any]], steps: List[Tuple[Any, Optional[str]]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Each edit with the id of the step it names, and what named no step."""
    named: List[Dict[str, Any]] = []
    unknown: List[str] = []
    for edit in edits:
        said = next((edit[key] for key in STEP_KEYS if edit.get(key) not in (None, "")), None)
        task = _step_named(said, steps)
        if task is None:
            unknown.append(repr(said) if said is not None else "(no step named)")
            continue
        fields = {k: v for k, v in edit.items() if k in FIELDS and v not in (None, "")}
        named.append({"task_id": str(task.id), **fields})
    return named, unknown


def _step_named(said: Any, steps: List[Tuple[Any, Optional[str]]]) -> Any:
    if said is None or isinstance(said, bool):
        return None
    text = str(said).strip()
    for task, number in steps:
        if text == str(task.id) or (number and _same_number(text, number)):
            return task
    place = _place(said, len(steps))
    if place is not None:
        return steps[place - 1][0]
    words = set(re.findall(r"[a-z]{3,}", text.lower().replace("_", " ")))
    titled = [task for task, _ in steps if words and words <= set(re.findall(r"[a-z]{3,}", (task.title or "").lower()))]
    return titled[0] if len(titled) == 1 else None


def _same_number(said: str, number: str) -> bool:
    """"#0352.2", "0352.2", "352.2" and "0352-2" all name step #0352.2."""
    match = re.fullmatch(r"#?0*(\d+)[.\-]0*(\d+)", said)
    target = re.fullmatch(r"#0*(\d+)\.(\d+)", number)
    return bool(match and target and match.groups() == target.groups())


def _place(said: Any, count: int) -> Optional[int]:
    """A step named by its place in the plan: 3, "3", "step 3"."""
    match = re.fullmatch(r"(?:step\s*)?(\d{1,2})", str(said).strip().lower())
    place = int(match.group(1)) if match else None
    return place if place and 1 <= place <= count else None


def _no_such_step(unknown: List[str], steps: List[Tuple[Any, Optional[str]]]) -> str:
    listed = "; ".join(f"{number or f'step {task.sequence_number}'} '{(task.title or '').strip()[:60]}'"
                       f"{f' ({task.agent_role})' if getattr(task, 'agent_role', None) else ''}"
                       for task, number in steps) or "none yet"
    example = next((number for _, number in steps if number), "the step's number")
    return NO_SUCH_STEP.format(said=", ".join(unknown), steps=listed, example=example)


def _check_each_step(db: Session, run: Any) -> str:
    """Switch on the mission's check of every step."""
    run.config = {**(run.config or {}), CHECK_EACH_STEP: True}
    db.commit()
    return CHECKS_ON


__all__ = ["CHECKS_ON", "reads_the_plan_edits"]
