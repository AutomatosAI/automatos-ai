"""``platform_update_task`` in a chat a person drives (F241, night 7b).

"Approve #0198 with this note: Right, 204 bags a sack" ended with the owner's note on
#0198 credited to "an agent". Auto writes a note in a chat because the person behind
the chat said it, so it is theirs: the board says "you", as it does for a note the
owner types on the card. A note from an agent's own lane (a heartbeat, a ticket, a
playbook step), where no person drives the call, still says "an agent".

``_user_id`` is the server-injected driver (platform executor, OPERATOR_CONSENT_ACTIONS),
never a model argument.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

NOTE = "note"
BY_THE_PERSON = "you"
# update_board_task's answer when a call carries nothing it changes.
NOTHING_TO_CHANGE = "Nothing to change"


def notes_say_who_asked(handler: Handler) -> Handler:
    """A note in a call a person drives is written as theirs; anything else is the handler's."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        note = str((params or {}).get(NOTE) or "").strip()
        if not note or not params.get("_user_id"):
            return await handler(db, workspace_id, params)
        out = await handler(db, workspace_id, {k: v for k, v in params.items() if k != NOTE})
        only_the_note = isinstance(out, dict) and str(out.get("error") or "").startswith(NOTHING_TO_CHANGE)
        if not isinstance(out, dict) or (out.get("success") is not True and not only_the_note):
            return out
        _write_note(db, workspace_id, params["task_id"], note)
        updated = {**(out.get("updated") or {}), NOTE: "added"}
        return {"success": True, "task_id": out.get("task_id") or params["task_id"], "updated": updated}
    return wrapped


def _write_note(db: Session, workspace_id: Any, task_id: Any, note: str) -> None:
    from api.board_tasks import MAX_TASK_NOTE_CHARS
    from services.cli_host_service import append_session_note

    append_session_note(db, task_id=int(task_id), workspace_id=workspace_id,
                        note=note[:MAX_TASK_NOTE_CHARS], by=BY_THE_PERSON)
    db.commit()


__all__ = ["BY_THE_PERSON", "notes_say_who_asked"]
