"""platform_my_actions (P256-FIX-N14-3, night 14, F417): what Auto did, and what the owner decided.

Night 14 Auto told the owner it had only read things, after a night of writes, and found 1 of 8
clicks when asked "what did I decide about #…". This read answers both from the platform's own
records: the asking person's chats' write receipts (``my_actions_reads``), done, still waiting
or asked and since decided, and their clicks on approval cards (``my_actions_clicks``), each card
by its board number. ``ticket`` narrows them to one card. P256-FIX-N15-4 (F425): they come as one
short line each (``my_actions_lines``), fitted to what a chat turn keeps of a tool's answer, and
what does not fit is counted, with the ``since`` that reads on from it. The asking person is server-injected
(``_driving_user_id``): a call with no person behind it (a widget visitor, a headless agent) reads
nothing.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.context_guard import count_tokens

from .my_actions_clicks import DECIDED, clicks_in, settled
from .my_actions_lines import Listed, by_time, record_lines, record_text
from .my_actions_reads import DONE, WAITING, Item, iso_utc, matching, number_key, writes_in
from .my_actions_window import Window, when_said, window_of

# What a chat turn keeps of one tool's answer is 2,000 tokens (consumers/chatbot/service.py
# _TOOL_RESULT_TOKEN_BUDGET); this answer stays under it, with room for the formatter's heading.
ANSWER_TOKENS = 1900
TITLE = "what I did and what you decided, {said}"
NO_PERSON = ("This read is for the person in the chat, and no person is behind this call: "
             "nothing was read.")
BAD_TICKET = "ticket must be a card's board number, like #1437; got {said!r}"
NOTHING = ("Nothing {said} (from {start} to {end} UTC){about}: no writes of mine, done or waiting, "
           "and no clicks of yours.")
FOUND = ("{said} (from {start} to {end} UTC){about}: {done} done, {waiting} waiting for your click, "
         "{decided} asked and since decided, {clicks} decided by you.")
ABOUT = " about card {ticket}"
LEFT_OUT = (" The record lists them oldest first; {count} more, from {when} UTC on, are not in it: call "
            "platform_my_actions again with since={since} for those.")
UNREAD = (" The record lists them oldest first; more, from {when} UTC on, are not in it: call "
          "platform_my_actions again with since={since} for those.")
EARLIER = ("Earlier turns' work, not this reply's: one change per line at its time (UTC, the day above it); "
           "a line can hold a run ('23:01 card made #1588, 23:05 #1589'). Give each its time ('at 23:01 on "
           "13 Oct I made card #1588') and its card's number.")
WIDEST_SINCE = "2026-10-13T23:55:00.000000+00:00"   # the longest a left-out note's since is, for measuring


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
    horizon = min((rest for rest in (writes.rest_from, clicks.rest_from) if rest is not None), default=None)
    done = _before(matching(settled(db, workspace_id, writes.items, window), key), horizon)
    return _answer(window, ticket if key else None, done, _before(matching(clicks.items, key), horizon), horizon)


def _answer(window: Window, ticket: Any, writes: List[Item], clicks: List[Item],
            horizon: Optional[datetime]) -> Dict[str, Any]:
    counts = {"done": _count(writes, DONE), "waiting": _count(writes, WAITING), "decided": _count(writes, DECIDED),
              "clicks": len(clicks)}
    said = {"said": window.said, "start": when_said(window.start), "end": when_said(window.end),
            "about": ABOUT.format(ticket=ticket) if ticket else ""}
    found = FOUND.format(**said, **counts) if any(counts.values()) else NOTHING.format(**said)
    answer = {"success": True, "title": TITLE.format(said=window.said), "counts": counts,
              "note": EARLIER if any(counts.values()) else None}
    widest = _left_out_note(Listed([], len(writes) + len(clicks), WIDEST_SINCE), None)
    room = ANSWER_TOKENS - count_tokens(json.dumps({**answer, "message": found + widest, "record": "",
                                                   "left_out": {"count": 0, "since": WIDEST_SINCE},
                                                   "truncated": True}, default=str, indent=2))
    listed = record_lines(by_time(writes, clicks), room)
    if listed.next_since is None and horizon is not None:
        listed = Listed(listed.lines, 0, iso_utc(horizon))
    message = found + (_left_out_note(listed, horizon) if listed.next_since else "")
    return {**answer, "message": message[0].upper() + message[1:], "record": record_text(listed),
            "left_out": {"count": listed.left_out, "since": listed.next_since} if listed.next_since else None,
            "truncated": listed.next_since is not None}


def _left_out_note(listed: Listed, horizon: Optional[datetime]) -> str:
    """How many items the record leaves out, from when, and the ``since`` that reads them."""
    when = when_said(datetime.fromisoformat(listed.next_since))
    if horizon is not None:
        return UNREAD.format(when=when, since=listed.next_since)
    return LEFT_OUT.format(count=listed.left_out, when=when, since=listed.next_since)


def _before(items: List[Item], horizon: Optional[datetime]) -> List[Item]:
    """The items read before the first row a cap left out (all of them when none was)."""
    if horizon is None:
        return items
    return [item for item in items if datetime.fromisoformat(item["at"]) < horizon]


def _count(items: List[Item], status: str) -> int:
    return sum(1 for item in items if item["status"] == status)


__all__ = ["my_actions"]
