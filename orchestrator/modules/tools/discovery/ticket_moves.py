"""F259 (night 7) and F278 (night 7b): Auto moves a ticket only where the board would.

PRD-252 R6 makes a drag on the board do what the matching button does. Auto's
``platform_update_task_status`` kept its own path, so Auto could still:

- file a ticket nobody had worked on as Done or Review (#0044, #0045): finished,
  with nothing on the card;
- move a running ticket to Review or Done (#0094, #0111): the run went on, billed,
  and its answer never reached the card;
- send a ticket back by moving it to Assigned (#0181, night 7b): the card re-ran
  its OLD brief, and the draft it was sent back with vanished (no history, no note).

``keeps_the_board_rules`` wraps the tool. The first two are refused as tool errors in
the board's own words (``services/board_drag_rules.move_refusal``), so Auto can tell
the owner which button to press. A ticket in Review or Done moved to Assigned is the
board's Reject (``api/board_tasks.send_back``): the draft goes on record, the
owner's ``note`` is the correction its redo works from, and the redo starts. Review →
Done stays Auto's approval (F235 files the deliverable); its ``note`` is kept as the
board's Approve keeps one. The words Auto reaches for ("approved", "send back") mean
those two moves. A bulk call refuses only the tickets the board would, and moves the
rest.

Night 9b (F318): a move of an answered ticket to In progress with a note re-ran its old
brief, and the note was dropped (only a move to Done kept one): the owner's words were
nowhere on the card, and times_sent_back stayed 0. A note on that move is the owner's
correction, so it is the board's Reject too. F319: a send-back's answer says so
(``sent_back``), which is what backs Auto's "I've sent it back".
"""
from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

SEND_BACK = "assigned"
RERUN = "in_progress"
ANSWERED = ("review", "done")
BY_AN_AGENT = "platform_tool"
AN_AGENTS_NOTE_BY = "an agent"
MAX_NOTE_CHARS = 1000  # an operator note on the board (api/board_tasks.MAX_TASK_NOTE_CHARS)
SENT_BACK_WITH_A_NOTE = "Sent back: its draft is kept in the ticket's history, and the redo works from your note."
SENT_BACK_WITHOUT_ONE = ("Sent back: its draft is kept in the ticket's history. With no note the agent "
                         "only knows to try again; add one to say what to fix.")


def keeps_the_board_rules(handler: Handler) -> Handler:
    """Wrap update_board_task_status: a move the board refuses is a tool error in the
    board's words, a send-back is the board's Reject, and the owner's note is kept.
    Any other move is the handler's."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from modules.tools.discovery.handlers_board_tasks import MAX_BULK_TASK_IDS

        params = _in_the_boards_words(params)
        listed = _listed(params)
        if listed and len(listed) > MAX_BULK_TASK_IDS:  # refused whole by the handler, nothing moved first
            return await handler(db, workspace_id, params)
        decided = _the_boards_way(db, workspace_id, listed or [params.get("task_id")], params)
        if not decided:
            result = await handler(db, workspace_id, params)
            _keep_the_note(db, workspace_id, params, result)
            return result
        if listed is None:
            return next(iter(decided.values()))
        return await _the_rest(handler, db, workspace_id, params, listed, decided)
    return wrapped


async def _the_rest(handler: Handler, db: Session, workspace_id: Any, params: Dict[str, Any],
                    listed: List[Any], decided: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """A bulk call with tickets already answered the board's way: the handler moves
    the rest, and the answer counts them all."""
    rest = [ref for ref in listed if _as_id(ref) not in decided]
    result = await handler(db, workspace_id, {**params, "task_ids": rest}) if rest else {}
    _keep_the_note(db, workspace_id, params, result)
    return _bulk_answer(params.get("status"), len(listed), decided, result)


def _listed(params: Dict[str, Any]) -> Optional[List[Any]]:
    ids = params.get("task_ids")
    return ids if isinstance(ids, list) and ids else None


def _in_the_boards_words(params: Dict[str, Any]) -> Dict[str, Any]:
    """The call with its status said the board's way: "approved" is Done, "send back" Assigned."""
    from modules.tools.execution.call_effects import STATUS_WORDS   # night 7b: the words Auto reaches for

    status = params.get("status")
    meant = STATUS_WORDS.get(status.strip().lower()) if isinstance(status, str) else None
    return {**params, "status": meant} if meant else params


def _the_boards_way(db: Session, workspace_id: Any, refs: List[Any], params: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """The tickets this call moves as the board would, by id, with each one's answer:
    a refusal in the board's words, or a send-back through the board's Reject. The
    rest are the handler's to move."""
    new_status = params.get("status")
    answers: Dict[int, Dict[str, Any]] = {}
    if not isinstance(new_status, str):
        return answers
    for task in _tickets(db, workspace_id, refs):
        if task.status == new_status:
            continue
        if _sends_it_back(task, new_status, params):
            answers[task.id] = _send_back(db, workspace_id, task, params)
            continue
        refusal = _refusal(db, task, new_status)
        if refusal:
            answers[task.id] = _refused(task, refusal)
    return answers


def _sends_it_back(task: Any, new_status: str, params: Dict[str, Any]) -> bool:
    """An answered ticket moved to Assigned, or (F318) to In progress with a note: the
    board's Reject, the note its correction."""
    if task.status not in ANSWERED:
        return False
    return new_status == SEND_BACK or (new_status == RERUN and bool(str(params.get("note") or "").strip()))


def _tickets(db: Session, workspace_id: Any, refs: List[Any]) -> List[Any]:
    """This workspace's tickets among ``refs``; one not found is the handler's to report."""
    from core.models.core import BoardTask

    ids = [i for i in (_as_id(ref) for ref in refs) if i is not None]
    if not ids:
        return []
    return db.query(BoardTask).filter(BoardTask.id.in_(ids), BoardTask.workspace_id == workspace_id).all()


def _refusal(db: Session, task: Any, new_status: str) -> Optional[str]:
    """``move_refusal`` for this ticket: a running ticket waits or is cancelled, and a
    finished column needs work on the card."""
    from api.board_tasks import MISSION_TICKET_TYPES, _running_now
    from services.board_drag_rules import move_refusal

    return move_refusal(task, new_status, running=_running_now(db, task),
                        mission_ticket=getattr(task, "source_type", None) in MISSION_TICKET_TYPES)


def _send_back(db: Session, workspace_id: Any, task: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The board's Reject for a ticket Auto sends back (F278), the owner's note as the
    correction. A mission's own card is decided on the mission's page, as on the board."""
    from fastapi import HTTPException

    from api.board_tasks import MISSION_CARD_SOURCE, MISSION_CARD_VERDICT, send_back
    from modules.tools.execution.call_effects import SENT_BACK_SAID
    from services.board_consent import actor_from_user_id

    if getattr(task, "source_type", None) == MISSION_CARD_SOURCE:
        return _refused(task, MISSION_CARD_VERDICT)
    driver = params.get("_user_id")
    ctx = SimpleNamespace(workspace_id=workspace_id, user_id=driver)
    try:
        note = send_back(db, ctx, task, params.get("note"),
                         by=actor_from_user_id(driver) if driver else BY_AN_AGENT)
    except HTTPException as refused:
        return _refused(task, str(refused.detail))
    return {"success": True, "task_id": task.id, "status": task.status, "triggered_execution": True,
            SENT_BACK_SAID: True, "message": SENT_BACK_WITH_A_NOTE if note else SENT_BACK_WITHOUT_ONE}


def _keep_the_note(db: Session, workspace_id: Any, params: Dict[str, Any], result: Any) -> None:
    """The note a move carried, on each ticket the handler moved: an approval's note on
    one moved to Done (as the board's Approve keeps it), a plain note on any other.
    The owner's when a person drives the call, an agent's otherwise."""
    from modules.tools.discovery.handlers_board_task_done import moved_to_done

    note = str(params.get("note") or "").strip()[:MAX_NOTE_CHARS]
    moved = moved_to_done(result) if note and isinstance(result, dict) else []
    if not moved:
        return
    from services.cli_host_service import append_session_note
    from services.ticket_verdict import OPERATOR_NOTE_BY, keep_approval_note

    owners = bool(params.get("_user_id"))
    for task_id in moved:
        if owners and params.get("status") == "done":
            keep_approval_note(db, task_id=task_id, workspace_id=workspace_id, note=note)
        else:
            append_session_note(db, task_id=task_id, workspace_id=workspace_id, note=note,
                                by=OPERATOR_NOTE_BY if owners else AN_AGENTS_NOTE_BY)
    db.commit()


def _refused(task: Any, why: str) -> Dict[str, Any]:
    return {"success": False, "task_id": task.id, "error": why}


def _as_id(ref: Any) -> Optional[int]:
    if isinstance(ref, bool):
        return None
    if isinstance(ref, int):
        return ref
    return int(ref) if isinstance(ref, str) and ref.strip().isdigit() else None


def _bulk_answer(status: Any, requested: int, decided: Dict[int, Dict[str, Any]], tool: Any) -> Dict[str, Any]:
    """The bulk answer's shape (update_board_task_status), counting the tickets moved
    the board's way here with the handler's."""
    tool = tool if isinstance(tool, dict) else {}
    updated = [tid for tid, answer in decided.items() if answer.get("success")] + list(tool.get("updated") or [])
    failed = [{"task_id": tid, "error": answer["error"]} for tid, answer in decided.items()
              if not answer.get("success")] + list(tool.get("failed") or [])
    if tool and tool.get("success") is False and not isinstance(tool.get("updated"), list):
        failed.append({"task_id": None, "error": tool.get("error")})
    return {"success": not failed, "partial": bool(updated) and bool(failed), "status": status,
            "requested": requested, "updated_count": len(updated), "updated": updated, "failed": failed}


__all__ = ["keeps_the_board_rules"]
