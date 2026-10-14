"""P256-FIX-N14-3 (night 14, F417): what Auto did, read back from the saved receipts.

Night 14, 04:43: "Tonight I've only read things. I haven't sent, created, changed or scheduled
anything", after Auto had made an agent, a skill, five cards, a Gmail draft, a memory and a
playbook run that night. Each turn's receipts were saved with its message (PRD-256 US-001, the
``receipts`` part), and nothing read them across chats.

``writes_in`` reads the done and waiting write receipts of the asking person's own chats in the
window, in this workspace only, never another person's chat. P256-FIX-N15-4 (F425): it reads only the
turns that hold a write, oldest first, so a cap leaves out the newest and the answer can say where
the rest begins (``Found.rest_from``) and how to read on. Each names its card by the board
number (``#workspace_seq``): by the receipt's subject, else by the card its link opens; a row id
is never shown (P256-FIX-N13-7). The clicks are ``my_actions_clicks``'s.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set

from .my_actions_window import Window, when_said

Item = Dict[str, Any]

RECEIPTS_PART, WRITE, DONE, WAITING = "receipts", "write", "done", "waiting"  # consumers/chatbot/receipts.py
ASSISTANT = "assistant"
MAX_MESSAGES = 500   # the window's assistant turns holding a write that are read, oldest first
HOLDS_A_WRITE = [{"type": RECEIPTS_PART, RECEIPTS_PART: [{"kind": WRITE}]}]   # JSONB containment
# Each item's time as history: "at 03:04 on 13 Oct" (claims_backed reads "at 03:04" as a past time,
# so a recap the owner asked for is not taken for this reply's own claims).
AT = "at {when}"
CARD_LINK_ID = re.compile(r"[?&]task_id=(\d+)")  # receipts.CARD_LINK
NUMBER_KEY = re.compile(r"^#?0*(\d+)(?:\.(\d+))?$")


class Found(NamedTuple):
    """What a read found, oldest first, and, when its cap cut the newest rows off, when they begin."""

    items: List[Item]
    rest_from: Optional[datetime] = None


def number_key(ref: Any) -> Optional[str]:
    """'#0042', '0042', 42 → '42'; a mission step '#0051.3' → '51.3'; None when ``ref`` is no number."""
    found = NUMBER_KEY.match(str(ref).strip()) if ref not in (None, "") and not isinstance(ref, bool) else None
    if found is None:
        return None
    return f"{found.group(1)}.{found.group(2)}" if found.group(2) else found.group(1)


def matching(items: Iterable[Item], key: Optional[str]) -> List[Item]:
    """The items that name the card ``key`` (``number_key``); all of them when ``key`` is None."""
    items = list(items)
    if key is None:
        return items
    return [item for item in items if key in _keys(item.get("numbers") or [item.get("number")])]


def writes_in(db: Any, workspace_id: Any, user_id: int, window: Window) -> Found:
    """The done and waiting writes of ``user_id``'s chats in ``window``, oldest first."""
    from core.models.core import Chat, Message

    rows = (db.query(Message.parts, Message.created_at, Chat.id, Chat.title)
            .join(Chat, Chat.id == Message.chat_id)
            .filter(Message.workspace_id == workspace_id, Chat.workspace_id == workspace_id,
                    Chat.user_id == user_id, Message.role == ASSISTANT, Message.parts.contains(HOLDS_A_WRITE),
                    Message.created_at >= _naive(window.start), Message.created_at < _naive(window.end))
            .order_by(Message.created_at.asc(), Message.id.asc()).limit(MAX_MESSAGES + 1).all())
    items = [item for row in rows[:MAX_MESSAGES] for item in _writes_of(row)]
    rest_from = rows[MAX_MESSAGES][1] if len(rows) > MAX_MESSAGES else None
    return Found(_numbered(db, workspace_id, items), _utc(rest_from) if rest_from is not None else None)


def at_said(moment: datetime) -> str:
    """'at 03:04 on 13 Oct' (UTC)."""
    return AT.format(when=when_said(moment))


def iso_utc(moment: Optional[datetime]) -> Optional[str]:
    """In UTC, so the items sort by it ("oldest first") whatever zone the row came back in."""
    if moment is None:
        return None
    return (moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)).isoformat()


def _writes_of(row: Any) -> List[Item]:
    parts, at, chat_id, title = row
    receipts = [r for part in (parts if isinstance(parts, list) else []) if isinstance(part, dict)
                and part.get("type") == RECEIPTS_PART for r in (part.get(RECEIPTS_PART) or [])]
    return [_write(receipt, at, chat_id, title) for receipt in receipts
            if isinstance(receipt, dict) and receipt.get("kind") == WRITE and receipt.get("status") in (DONE, WAITING)]


def _write(receipt: Dict[str, Any], at: datetime, chat_id: Any, title: Any) -> Item:
    subject = str(receipt.get("subject") or "")
    linked = CARD_LINK_ID.search(str(receipt.get("link") or ""))
    return {"at": iso_utc(at), "when": at_said(at), "status": receipt.get("status"), "what": receipt.get("effect"),
            "subject": subject, "action": receipt.get("action"),
            "number": subject if NUMBER_KEY.match(subject) else None,
            "task_id": int(linked.group(1)) if linked else None, "chat": title, "chat_id": str(chat_id)}


def _numbered(db: Any, workspace_id: Any, items: List[Item]) -> List[Item]:
    """Each item that names no number but links a card, with that card's board number."""
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_numbers

    ids = {item["task_id"] for item in items if item["number"] is None and item["task_id"] is not None}
    if not ids:
        return [_without_row_id(item) for item in items]
    tasks = db.query(BoardTask).filter(BoardTask.id.in_(ids), BoardTask.workspace_id == workspace_id).all()
    numbers = ticket_numbers(db, workspace_id, tasks)
    return [_without_row_id({**item, "number": item["number"] or numbers.get(item["task_id"])}) for item in items]


def _without_row_id(item: Item) -> Item:
    """A row id is never shown (P256-FIX-N13-7): the board number stands for the card."""
    return {key: value for key, value in item.items() if key != "task_id"}


def _keys(numbers: Sequence[Any]) -> Set[str]:
    return {key for key in (number_key(number) for number in numbers) if key}


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def _naive(moment: datetime) -> datetime:
    """``messages.created_at`` is stored without a zone, in UTC."""
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


__all__ = ["DONE", "WAITING", "Found", "Item", "at_said", "iso_utc", "matching", "number_key", "writes_in"]
