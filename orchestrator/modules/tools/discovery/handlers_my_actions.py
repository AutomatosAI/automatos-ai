"""platform_my_actions (P256-FIX-N14-3, night 14, F417): what Auto did, and what the owner decided.

Night 14 Auto told the owner it had only read things, after a night of writes, and found 1 of 8
clicks when asked "what did I decide about #…". This read answers both from the platform's own
records: the asking person's chats' write receipts (``my_actions_reads``), done, still waiting
or asked and since decided, and their clicks on approval cards (``my_actions_clicks``), each card
by its board number. ``ticket`` narrows them to one card. The asking person is server-injected
(``_driving_user_id``): a call with no person behind it (a widget visitor, a headless agent) reads
nothing.
"""
from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy.orm import Session

from .my_actions_clicks import DECIDED, clicks_in, settled
from .my_actions_reads import DONE, WAITING, Found, matching, number_key, writes_in
from .my_actions_window import Window, when_said, window_of

MAX_LISTED = 60  # per list: the newest are kept, the counts say how many there were
TITLE = "what I did and what you decided, {said}"
NO_PERSON = ("This read is for the person in the chat, and no person is behind this call: "
             "nothing was read.")
BAD_TICKET = "ticket must be a card's board number, like #1437; got {said!r}"
NOTHING = ("Nothing {said} (from {start} to {end} UTC){about}: no writes of mine, done or waiting, "
           "and no clicks of yours.")
FOUND = ("{said} (from {start} to {end} UTC){about}: {done} done, {waiting} waiting for your click, "
         "{decided} asked and since decided, {clicks} decided by you.")
ABOUT = " about card {ticket}"
CUT = " Not everything is listed: ask about a shorter window or one card."
EARLIER = ("These are earlier turns' work, not this reply's: give each its time as listed ('at 03:04 on 13 Oct "
           "I created …') and name each card by its number.")


async def my_actions(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Auto's done and waiting writes and the person's clicks in a window, by board number."""
    user_id = params.get("_driving_user_id")
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        return {"success": False, "error": NO_PERSON}
    window, wrong = window_of(params)
    if window is None:
        return {"success": False, "error": wrong}
    ticket = params.get("ticket")
    key = number_key(ticket) if ticket not in (None, "") else None
    if ticket not in (None, "") and key is None:
        return {"success": False, "error": BAD_TICKET.format(said=ticket)}
    writes, clicks = writes_in(db, workspace_id, user_id, window), clicks_in(db, workspace_id, user_id, window)
    found = Found(matching(settled(db, workspace_id, writes.items, window), key), writes.capped)
    return _answer(window, ticket if key else None, found, Found(matching(clicks.items, key), clicks.capped))


def _answer(window: Window, ticket: Any, writes: Found, clicks: Found) -> Dict[str, Any]:
    lists = {status: [item for item in writes.items if item["status"] == status] for status in (DONE, WAITING, DECIDED)}
    lists["clicks"] = clicks.items
    counts = {"done": len(lists[DONE]), "waiting": len(lists[WAITING]), "decided": len(lists[DECIDED]),
              "clicks": len(clicks.items)}
    cut = writes.capped or clicks.capped or any(len(listed) > MAX_LISTED for listed in lists.values())
    said = {"said": window.said, "start": when_said(window.start), "end": when_said(window.end),
            "about": ABOUT.format(ticket=ticket) if ticket else ""}
    message = (FOUND.format(**said, **counts) if any(counts.values()) else NOTHING.format(**said)) + (CUT if cut else "")
    return {"success": True, "title": TITLE.format(said=window.said),
            "window": {"from": window.start.isoformat(), "to": window.end.isoformat(), "said": window.said},
            "done": _newest(lists[DONE]), "waiting": _newest(lists[WAITING]), "decided": _newest(lists[DECIDED]),
            "clicks": _newest(lists["clicks"]), "counts": counts, "truncated": cut,
            "message": message[0].upper() + message[1:], "note": EARLIER if any(counts.values()) else None}


def _newest(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return items[-MAX_LISTED:]


__all__ = ["my_actions"]
