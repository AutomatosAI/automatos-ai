"""PRD-256 FX-009 (night 12, B3, M5, E6): what the owner's click ran reaches the next turn.

After a click, the resume (api/approval_grants ``_resume_tool_call``) stored only whether
the call succeeded, and the chat history the next turn reads holds text parts only
(consumers/chatbot/prompt_analyzer): the mission the click created never reached the
model, which guessed its id again on the next turn and failed.

Now the resume keeps the result's ids and titles on the grant (``executed_result.ids``:
the mission's id, its card's number, a ticket's id and title), and the chat the ask came
from gets one line, "Your click ran: <what the card asked> → <ids>" (or why it did not go
through), posted as a text message through the one background write path
(``chat_messenger``), so the next turn's history carries it. Only a chat's ask is told;
an agent's or a board run's has no chat to tell.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

# The result keys that name what the call made or changed, in the order the line reads them.
ID_KEYS = ("mission_id", "task_id", "agent_id", "document_id", "execution_id", "playbook_id", "post_id", "id")
NUMBER_KEY = "number"
TITLE_KEYS = ("title", "name", "goal")
# Result objects that carry the row the call made or changed (get_mission's "mission", a ticket's "task").
ROWS = ("mission", "task", "agent", "execution", "post", "document")
ROW_ID = "id"
MAX_IDS = 6
TITLE_CHARS = 120
ERROR_CHARS = 500
ORIGIN = "owner_click"
LINK_TYPE = "approval"
CHAT_LANE = "chat"
LINE = "Your click ran: {act} → {ids}"
NOT_DONE = "Your click ran: {act} → it did not go through: {error}"
NO_IDS = "done"
NO_REASON = "no reason given"
IDS = "ids"


def executed_summary(raw: Any) -> Dict[str, Any]:
    """The grant's ``executed_result`` for a resumed call's result: whether it ran, why
    not, and (FX-009) the ids and titles of what it made or changed."""
    raw = raw if isinstance(raw, dict) else {}
    ok = bool(raw.get("success"))
    summary: Dict[str, Any] = {
        "success": ok,
        "error": (str(raw.get("error"))[:ERROR_CHARS] if (not ok and raw.get("error")) else None),
        "requires_confirmation": bool(raw.get("requires_confirmation")),
        "executed_at": _now(),
    }
    ids = result_ids(raw) if ok else {}
    return {**summary, IDS: ids} if ids else summary


def failed_summary(error: Any) -> Dict[str, Any]:
    """The grant's ``executed_result`` for a resume that did not run: why, never a success."""
    return {"success": False, "error": str(error)[:ERROR_CHARS], "requires_confirmation": False,
            "executed_at": _now()}


def result_ids(raw: Dict[str, Any]) -> Dict[str, str]:
    """The ids, number and title a result names, from the result and the rows it carries:
    {"mission_id": "3f2…", "number": "#0992", "goal": "Plan the spring menu"}."""
    found: Dict[str, str] = {}
    rows = [("", raw), *((name, raw[name]) for name in ROWS if isinstance(raw.get(name), dict))]
    for prefix, row in rows:
        for key, value in _named_in(prefix, row):
            found.setdefault(key, value)
    return dict(list(found.items())[:MAX_IDS])


def said_in_the_chat(db: Any, grant: Any) -> None:
    """Tell the chat the ask came from what its click ran (one text line the next turn
    reads). Never raises: the click has already run, and its grant is committed."""
    details = grant.details if isinstance(getattr(grant, "details", None), dict) else {}
    result = details.get("executed_result")
    chat_id = details.get("conversation_id")
    if not (isinstance(result, dict) and "success" in result and details.get("lane") == CHAT_LANE
            and _is_uuid(chat_id)):
        return
    from services.chat_messenger import deliver_background_message

    caller = details.get("caller_context") if isinstance(details.get("caller_context"), dict) else {}
    deliver_background_message(db, workspace_id=grant.workspace_id, text=click_line(grant, result),
                               source={"origin": ORIGIN, "grant_id": grant.id}, chat_id=str(chat_id),
                               clerk_user_id=caller.get("user_id"), link_type=LINK_TYPE, link_id=str(grant.id))


def click_line(grant: Any, result: Dict[str, Any]) -> str:
    """"Your click ran: approve a mission's plan 'Spring menu' (mission #0992) → mission_id 3f2…"."""
    act = _act(grant)
    if not result.get("success"):
        return NOT_DONE.format(act=act, error=result.get("error") or NO_REASON)
    ids = result.get(IDS) if isinstance(result.get(IDS), dict) else {}
    return LINE.format(act=act, ids=", ".join(f"{key} {value}" for key, value in ids.items()) or NO_IDS)


def _act(grant: Any) -> str:
    """What the card asked, from its question's first line ("Approve a mission's plan …:"),
    else the action's name."""
    asked = str(getattr(grant, "question_md", None) or "").strip()
    head = asked.splitlines()[0].rstrip(":.").strip() if asked else ""
    if not head:
        return str(getattr(grant, "tool_name", None) or "the call")
    return f"{head[:1].lower()}{head[1:]}"


def _named_in(prefix: str, row: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """Each id, number and title ``row`` holds; a row's own ``id`` is "<row>_id"."""
    for key in (*ID_KEYS, NUMBER_KEY, *TITLE_KEYS):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, UUID)) or value == "":
            continue
        named = f"{prefix}_{ROW_ID}" if prefix and key == ROW_ID else key
        yield named, str(value)[:TITLE_CHARS]


def _is_uuid(value: Any) -> bool:
    try:
        UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["click_line", "executed_summary", "failed_summary", "result_ids", "said_in_the_chat"]
