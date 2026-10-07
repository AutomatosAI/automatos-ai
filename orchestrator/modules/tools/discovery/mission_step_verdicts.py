"""Auto's approve and reject on a mission that has started decide its step, as the board does (night 9: F308).

Night 9 (build 13), Auto in the owner's chat:
- "Step 1 of mission 27 (the green check) is fine - approve it with the note: Guji 118 kg
  is plenty, carry on." became platform_approve_mission {mission_id: 27}. #0027 was paused
  for the owner's check of step 1, and approving its plan moved it from paused to running,
  but the step stayed held: card #1875 sat in Review and steps 2-3 stayed queued until the
  owner pressed Approve on the board. Auto said step 1 was approved.
- "Reject it, then - send it back with: I need the three cafés and their kilos, or a plain
  line saying you can't." for step 1 of #0035 became platform_reject_mission {mission_id:
  35, reason: …}. Rejecting a plan cancels its mission whatever state it is in: #0035 was
  cancelled, step 2 skipped, #1888 left in Review, and Auto said it couldn't be brought back.
- "Yes, approve and run it - and remember each step stops for me before the next one
  starts." approved #0033 with no check of each step, and it ran through.

A mission's plan is approved or rejected only while it waits for that. Once it has started:
- approve is the board's Approve on the card of the step that waits for the owner's check
  (``api/board_tasks.approve_task``): the step is through, the owner's note is on its card,
  and the mission carries on;
- reject is the board's Reject on that step's card (``ticket_moves`` → ``send_back``): the
  owner's words go on it and the mission redoes the step, opening again when it had
  finished (F284). It never cancels the mission; platform_cancel_mission does that.
The step is the one the call names (``step``: its card's number, its place or its title;
or a step's card number as the mission), else the one step that waits for the owner.
When none, or several, can be told apart, nothing is done and the answer lists the steps.
An approval with no step waiting goes to the plan's approval as before.

The note on the card is the owner's own words: the call's, when they are the owner's, else
what the owner wrote after "note:" or "with:", else (a send-back) their latest message.
An approval of a plan whose owner's words ask for each step to stop switches the mission's
check of each step on first, and an approval's answer says whether the steps wait.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from modules.tools.discovery.owner_only import signed_by

APPROVES, REJECTS = "approve", "reject"
PLAN_STATES = ("pending", "planning", "awaiting_approval")
CANCELLED = "cancelled"
FINISHED = ("completed", "failed")
STEP_CARD, MISSION_CARD = "orchestration_task", "orchestration"
WAITING_STATE = "verifying"
NOTE_SHARE = 0.8
BY_AN_AGENT = "platform_tool"
TITLE_CHARS = 60
# What the owner wrote for the card: "approve it with the note: …", "send it back with: …".
NOTE_AFTER = re.compile(r"\b(?:note|notes|with|saying|feedback|correction|comment)\s*:\s*(.+)", re.I | re.S)

APPROVED = "{label} approved{noted}, as the board's Approve does: the step is through and mission {mission} {goes}."
WITH_NOTE = ", the owner's note kept on its card"
CARRIES_ON = "carries on"
STILL_WAITS = "is {state}"
SENT_BACK = ("{label} sent back with the owner's words, as the board's Reject does: mission {mission} redoes it"
             "{reopened}. The mission was not cancelled.")
REOPENED = ", and opened again to do so"
NOTHING_WAITS = ("Mission {mission} is {state}: its plan was approved already and no step waits for the owner's "
                 "check, so nothing was done. Its steps: {steps}.")
NO_STEP_NAMED = ("Mission {mission} has started, so rejecting its plan would cancel it: nothing was done. To have a "
                 "step redone, call this again with step (its number) and the owner's words as the reason; its "
                 "mission redoes it. Its steps: {steps}. Only to stop the whole mission: platform_cancel_mission.")
SEVERAL_WAIT = ("{count} steps of mission {mission} wait for the owner's check: {steps}. Nothing was done: call "
                "this again with step, the one the owner meant.")
NO_CARD = "Step {label} of mission {mission} has no card on the board, so nothing was done."


async def decided_on_the_step(db: Session, workspace_id: Any, run: Any, params: Dict[str, Any],
                              verdict: str) -> Optional[Dict[str, Any]]:
    """The answer when approve (``APPROVES``) or reject (``REJECTS``) on ``run`` is its
    step's to decide, None when it is the plan's (see the module). Before an approval of
    a plan, the owner's words may switch its check of each step on."""
    state = getattr(run, "state", None)
    if not isinstance(state, str) or state in PLAN_STATES or state == CANCELLED:
        if verdict == APPROVES and state == "awaiting_approval":
            _checks_as_the_owner_asks(db, workspace_id, run, params)
        return None
    step_card, refusal = _the_step(db, workspace_id, run, params.get("step"), verdict)
    if refusal:
        return {"success": False, "error": refusal}
    if step_card is None:
        return None
    said = _owners_words(db, workspace_id, params)
    if verdict == APPROVES:
        return await _approve(db, workspace_id, run, step_card, owners_note(said, params.get("note")), params)
    note = owners_note(said, params.get("reason") or params.get("note"), latest_otherwise=True)
    return _send_back(db, workspace_id, run, step_card, note, params)


def with_the_check_said(run: Any, out: Any) -> Any:
    """An approval's answer, saying whether the mission's steps wait for the owner's check."""
    from modules.coordination.owner_checks import checks_each_step
    from modules.tools.discovery.mission_asks import CHECKS_EACH_STEP, RUNS_UNCHECKED

    if not (isinstance(out, dict) and out.get("success")):
        return out
    checks = checks_each_step(getattr(run, "config", None))
    return {**out, "checks_each_step": checks,
            "message": f"{out.get('message', '')}{CHECKS_EACH_STEP if checks else RUNS_UNCHECKED}"}


def step_of_a_started_mission(db: Session, card: Any, run_id: Any) -> Optional[str]:
    """A step's number, for approving or rejecting it through its mission, when the
    mission has started; None while its plan waits (a step is not approved alone then)."""
    from core.models.orchestration import OrchestrationRun
    from services.ticket_numbers import ticket_number

    run = db.get(OrchestrationRun, run_id)
    if run is None or run.state in PLAN_STATES or run.state == CANCELLED:
        return None
    return ticket_number(db, card) or str(card.id)


def lets_held_steps_through(db: Session, workspace_id: Any, task_ids: Sequence[int], params: Dict[str, Any]) -> None:
    """Auto moving a held step's card to Done (platform_update_task_status, the call
    mission_refs names for approving a step) is the board's Approve on it too: the step
    is through and its mission carries on, where the move alone left both waiting."""
    from core.models.core import BoardTask
    from modules.coordination.owner_checks import let_through

    if not task_ids:
        return
    cards = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.id.in_(list(task_ids)),
                                       BoardTask.source_type == STEP_CARD).all()
    by = signed_by(params) or BY_AN_AGENT   # PRD-256 US-004: who clicked, never the owner on Auto's call alone
    if [card for card in cards if let_through(db, card, by=by)]:
        db.commit()


def owners_note(said: Sequence[str], given: Any, *, latest_otherwise: bool = False) -> str:
    """The words for the step's card: the call's when they are the owner's, else what
    the owner wrote after "note:" or "with:", else their latest message (a send-back's
    correction) or nothing. Outside a chat, the call's own."""
    from modules.tools.discovery.mission_create_checks import share_of_words

    given = " ".join(str(given or "").split())
    if not said:
        return given
    if given and share_of_words(given, said) >= NOTE_SHARE:
        return given
    match = NOTE_AFTER.search(said[0] or "")
    if match and match.group(1).strip():
        return match.group(1).strip()
    return str(said[0]).strip() if latest_otherwise else ""


def _owners_words(db: Session, workspace_id: Any, params: Dict[str, Any]) -> List[str]:
    from modules.tools.discovery.mission_owner_words import owners_words

    return owners_words(db, workspace_id, params)


def _checks_as_the_owner_asks(db: Session, workspace_id: Any, run: Any, params: Dict[str, Any]) -> None:
    """#0033: "approve and run it — and remember each step stops for me" switches the
    check of each step on before the plan is approved."""
    from modules.coordination.owner_checks import with_step_checks
    from modules.tools.discovery.mission_owner_words import asks_for_checks, autos_last_words

    said = _owners_words(db, workspace_id, params)
    if said and asks_for_checks(said, autos_last_words(db, workspace_id, params)):
        run.config = with_step_checks(run.config, on=True)
        db.flush()


def _the_step(db: Session, workspace_id: Any, run: Any, said: Any, verdict: str) -> Tuple[Any, Optional[str]]:
    """(the step's card, None), (None, why nothing was done), or (None, None) when an
    approval has no step to decide and goes to the plan's approval as before."""
    from modules.tools.discovery.plan_edits import _no_such_step, _step_named, _steps

    steps = _steps(db, workspace_id, run)
    if said not in (None, ""):
        task = _step_named(said, steps)
        if task is None:
            return None, _no_such_step([repr(said)], steps)
        return _card_or_refusal(db, workspace_id, run, task, steps)
    waiting = [(task, number) for task, number in steps if _waits(task)]
    if len(waiting) == 1:
        return _card_or_refusal(db, workspace_id, run, waiting[0][0], steps)
    if waiting:
        return None, SEVERAL_WAIT.format(count=len(waiting), mission=_mission_label(db, workspace_id, run),
                                         steps=_listed(waiting))
    if verdict == REJECTS:
        return None, NO_STEP_NAMED.format(mission=_mission_label(db, workspace_id, run), steps=_listed(steps))
    if run.state in FINISHED:
        return None, NOTHING_WAITS.format(mission=_mission_label(db, workspace_id, run), state=run.state,
                                          steps=_listed(steps))
    return None, None


def _card_or_refusal(db: Session, workspace_id: Any, run: Any, task: Any,
                     steps: List[Tuple[Any, Optional[str]]]) -> Tuple[Any, Optional[str]]:
    from core.models.core import BoardTask

    card = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == STEP_CARD,
                                      BoardTask.orchestration_task_id == task.id).first()
    if card is not None:
        return card, None
    label = next((number for t, number in steps if t.id == task.id and number), f"{task.sequence_number}")
    return None, NO_CARD.format(label=label, mission=_mission_label(db, workspace_id, run))


def _waits(task: Any) -> bool:
    """A step held for the owner's check (``owner_checks.holds_for_the_owner``)."""
    from modules.coordination.owner_checks import WAITING_KEY

    return task.state == WAITING_STATE and bool((task.input_context or {}).get(WAITING_KEY))


def _listed(steps: List[Tuple[Any, Optional[str]]]) -> str:
    return "; ".join(f"{number or f'step {task.sequence_number}'} '{(task.title or '').strip()[:TITLE_CHARS]}' "
                     f"({task.state})" for task, number in steps) or "none yet"


def _mission_label(db: Session, workspace_id: Any, run: Any) -> str:
    """The mission's card number (#0027), or its id when it has no card."""
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_number

    card = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == MISSION_CARD,
                                      BoardTask.orchestration_run_id == run.id).first()
    return (ticket_number(db, card) if card is not None else None) or str(run.id)


def _person(params: Dict[str, Any]) -> Optional[str]:
    """Who decides: the person the executor named for this mission call (``_created_by``)."""
    return params.get("_created_by") or params.get("_user_id")


class _OwnersWords:
    """What the board's Approve reads from its request: the owner's note."""

    def __init__(self, note: str):
        self._body = {"note": note or None}

    async def json(self) -> Dict[str, Any]:
        return self._body


async def _approve(db: Session, workspace_id: Any, run: Any, card: Any, note: str,
                   params: Dict[str, Any]) -> Dict[str, Any]:
    """The board's Approve on the step's card, word for word (approve_task)."""
    from fastapi import HTTPException

    from api.board_tasks import approve_task
    from services.ticket_numbers import ticket_label

    label, mission = ticket_label(card), _mission_label(db, workspace_id, run)
    ctx = SimpleNamespace(workspace_id=workspace_id, user_id=_person(params))
    try:
        await approve_task(card.id, _OwnersWords(note), ctx=ctx, db=db)
    except HTTPException as refused:
        return {"success": False, "task_id": card.id, "error": f"{refused.detail} Nothing was done."}
    db.refresh(run)
    goes = CARRIES_ON if run.state == "running" else STILL_WAITS.format(state=run.state)
    return {"success": True, "mission_id": str(run.id), "state": run.state, "task_id": card.id,
            "message": APPROVED.format(label=label[0].upper() + label[1:], noted=WITH_NOTE if note else "",
                                       mission=mission, goes=goes)}


def _send_back(db: Session, workspace_id: Any, run: Any, card: Any, note: str,
               params: Dict[str, Any]) -> Dict[str, Any]:
    """The board's Reject on the step's card (ticket_moves → send_back)."""
    from modules.tools.discovery.ticket_moves import _send_back as boards_reject
    from services.ticket_numbers import ticket_label

    label, mission, finished = ticket_label(card), _mission_label(db, workspace_id, run), run.state in FINISHED
    out = boards_reject(db, workspace_id, card, {"note": note, "_user_id": _person(params)})
    if not out.get("success"):
        return {**out, "error": f"{out.get('error')} Nothing was done."}
    db.refresh(run)
    return {"success": True, "mission_id": str(run.id), "state": run.state, "task_id": card.id,
            "message": SENT_BACK.format(label=label[0].upper() + label[1:], mission=mission,
                                        reopened=REOPENED if finished else "")}


__all__ = ["APPROVES", "REJECTS", "decided_on_the_step", "lets_held_steps_through", "owners_note",
           "step_of_a_started_mission", "with_the_check_said"]
