"""``platform_update_task`` with a status moves the card, its note kept (F309, night 9).

Night 9, iteration 1: "Please approve card 1866 with this note: Good, this is the figure
I'll use. Ignore the June review from now on." Auto's first call was
platform_update_task {task_id: 1866, notes: "…", status: "done"}. The edit tool took no
status: the call was refused before it ran ("this action changes a ticket's details, not
its status … Any other move is platform_update_task_status's"). Auto then split the
approval in two: platform_update_task {note} (a plain note) and
platform_update_task_status {status: "done"} with no note. The move carried no note, so
the approval's own note ("Approved: …", ``services.ticket_verdict.keep_approval_note``, which the owner's lessons
read, F249) was never written, and the owner found their words on none of the card's
fields they checked (#1866). In iteration 2 Auto made the one call,
platform_update_task_status {status: "done", note}, and the note was kept (#1879).

So platform_update_task takes a status (``actions_board_tasks``), and it is that move:
any field it edits is edited first (the board's way, ``ticket_edits.briefs_like_the_board``),
then the card moves through platform_update_task_status with the call's note, which keeps
it as that move does (``notes`` reaches ``note`` through the executor's aliases).

A new brief on a card already worked on is the board's Re-brief, which sends the card back
itself. PRD-256 FX-013 (night 12, A440): that call with a status was refused, while the
owner's correction was urgent. The description now re-briefs the card and the status is
dropped: the answer says so (``status_ignored``, which the receipt shows as "status ignored:
a re-brief sends the card back by itself"), and the move it would have made is never
recorded as made (``call_effects.done_effects``).

The fix-wave review (P256-FIX-RVW-10): a closing status (done, cancelled) beside that new
brief asked the owner to "approve (move to Done)" and the click re-briefed the card and
sent it back. The drop applies only to a status that does not close the card; a closing
one is refused before any card (``rebrief_that_closes``, which ``owner_only.platform_ask``
reads), naming both choices, as before FX-013.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

from sqlalchemy.orm import Session

from modules.tools.discovery.owner_only import CLICKED_BY, closing_status

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

STATUS = "status"
NOTE = "note"
DESCRIPTION = "description"
# The fields platform_update_task edits; a status call with none of them only moves the card.
EDITED = ("title", DESCRIPTION, "priority", "review_mode", "tags")
STATUS_IGNORED = ("The status was ignored: a new brief on a card its agent has already worked on is the board's "
                  "Re-brief, which sends the card back to its agent by itself.")
EDIT_ACTION = "platform_update_task"
CLOSE_OR_REBRIEF = ("A new brief on a card its agent has already worked on is the board's Re-brief, which sends "
                    "the card back to its agent: it cannot also close the card. Nothing was done. Choose one: "
                    "re-brief it (send the description without a status), or close it (send the status "
                    "without a description).")


def moves_the_card(params: Dict[str, Any]) -> bool:
    """Whether an edit call carries a status to move its card to."""
    return isinstance(params.get(STATUS), str) and bool(params[STATUS].strip())


async def edited_then_moved(edit: Handler, db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The call's edits through ``edit`` (none when it has none), then the card's move
    through platform_update_task_status, with the call's note. The first refusal is
    the answer, and nothing after it is done."""
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status
    from modules.tools.discovery.ticket_edits import _worked_ticket

    if _rebriefs(params) and _worked_ticket(db, workspace_id, params.get("task_id")):
        if closing_status(params):
            return _close_or_rebrief()
        return await rebriefed_without_its_status(edit, db, workspace_id, params)
    edits = {k: v for k, v in params.items() if k not in (STATUS, NOTE)}
    edited = await edit(db, workspace_id, edits) if any(edits.get(f) is not None for f in EDITED) else {}
    if edited and edited.get("success") is not True:
        return edited
    moved = await update_board_task_status(db, workspace_id, _the_move(db, workspace_id, params))
    return {**moved, "updated": edited["updated"]} if edited.get("updated") else moved


def rebrief_that_closes(db: Session, workspace_id: Any, action: str,
                        params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The refusal for platform_update_task with a new brief and a closing status on a
    worked card, read before any card is raised (P256-FIX-RVW-10), else None. The card is
    read by its number as the ticket tools read it (``ticket_id_named``)."""
    from modules.tools.discovery.ticket_edits import _worked_ticket
    from services.ticket_refs import ticket_id_named

    if action != EDIT_ACTION or not _rebriefs(params) or closing_status(params) is None:
        return None
    task_id, _ = ticket_id_named(db, workspace_id, params.get("task_id"))
    return _close_or_rebrief() if task_id is not None and _worked_ticket(db, workspace_id, task_id) else None


def _rebriefs(params: Dict[str, Any]) -> bool:
    """Whether the call carries a new brief."""
    return bool(str(params.get(DESCRIPTION) or "").strip())


def _close_or_rebrief() -> Dict[str, Any]:
    """The refusal naming both choices: nothing was done."""
    return {"success": False, "error": CLOSE_OR_REBRIEF}


async def rebriefed_without_its_status(edit: Handler, db: Session, workspace_id: Any,
                                      params: Dict[str, Any]) -> Dict[str, Any]:
    """The board's Re-brief through ``edit`` with the call's status dropped, and an answer
    that says it was (FX-013); a refused re-brief is the answer as it came."""
    from modules.tools.execution.call_effects import SENT_BACK_SAID, STATUS_IGNORED_SAID

    out = await edit(db, workspace_id, {k: v for k, v in params.items() if k != STATUS})
    if not isinstance(out, dict) or out.get("success") is not True:
        return out
    message = " ".join(part for part in (str(out.get("message") or ""), STATUS_IGNORED) if part)
    return {**out, "message": message, SENT_BACK_SAID: True, STATUS_IGNORED_SAID: True}


def _the_move(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The status call: the card by its number, as the board says it (an id may be
    another card's number), its status, the note, and who drives the call."""
    from modules.tools.discovery.ticket_edits import _ticket
    from services.ticket_numbers import ticket_number

    task = _ticket(db, workspace_id, params.get("task_id"))
    move = {"task_id": (ticket_number(db, task) or task.id) if task is not None else params.get("task_id"),
            STATUS: params[STATUS].strip()}
    note = str(params.get(NOTE) or "").strip()
    extra = {NOTE: note} if note else {}
    driver = {key: params[key] for key in ("_user_id", CLICKED_BY) if params.get(key)}
    return {**move, **extra, **driver}


__all__ = ["CLOSE_OR_REBRIEF", "STATUS_IGNORED", "edited_then_moved", "moves_the_card", "rebrief_that_closes",
           "rebriefed_without_its_status"]
