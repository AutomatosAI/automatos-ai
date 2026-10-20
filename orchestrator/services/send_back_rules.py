"""P256-FIX-N17-2 (F444, night 17): the board's Reject's rule, in one place.

Night 17: "turn #1767 down" raised an approval card for a send-back, the owner clicked
it, and the click found the Reject's 422 ("Cannot reject a task with no assigned
agent"): the grant was given back and the card never moved, in silence. The Reject's
preconditions lived inside ``api.board_tasks.send_back`` alone, so the card Auto raises
in a person's chat (``modules/tools/discovery/send_back_checks``) could not read them.

This is that rule, read without writing anything: the board's Reject raises it, with
the same codes and the same words as before, and the chat's card reads it before it
asks the owner anything.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from services.run_redo import redo_refusal, takes_its_own_redo

# F092 (night 3): a finished ticket can be sent back, and a re-run says what it replaces.
SENDABLE_BACK = ("review", "done")
NOT_SENDABLE = "Only a ticket in review or done can be sent back (currently: {status})"
NO_AGENT = "Cannot reject a task with no assigned agent"
UNPROCESSABLE = 422
CONFLICT = 409


def send_back_refusal(db: Any, task: Any) -> Optional[Tuple[int, str]]:
    """(the HTTP code, why) the board's Reject refuses ``task``, before anything changes;
    None when it can be sent back. ``task`` may be read as a PATCH would leave it
    (``board_drag_rules.as_patched``: an agent the same call gives it counts)."""
    if task.status not in SENDABLE_BACK:
        return UNPROCESSABLE, NOT_SENDABLE.format(status=task.status)
    refused = redo_refusal(db, task)  # F243: a redo that can't run on this card
    if refused:
        return CONFLICT, refused
    if not task.assigned_agent_id and not takes_its_own_redo(task):
        return UNPROCESSABLE, NO_AGENT
    return None


__all__ = ["NO_AGENT", "SENDABLE_BACK", "send_back_refusal"]
