"""platform_search_chat_history: the asking person's own chats, by words or by window.

P256-FIX-N15-4 (night 15, F425): the person's own chats in this workspace (the server's
``_driving_user_id``, as GET /api/chat/search reads the signed-in user's), a message matching
when it holds every word of the query, in any order.

P256-FIX-N17-11 (nights 16-17, F443): "what did I ask you first tonight?" was answered seven
times with a recent ask (night 17: 18:27, then the repeats at 18:31) while the night's first was
17:03. No tool listed the person's own messages oldest first: the search needed a query, gave the
newest hits of both roles plus a second ``formatted`` copy of them, and the chat keeps the head
of a tool's first 2,000 tokens. Now:
- with no query, a ``window`` (my_actions_window's names: 'tonight' runs from 18:00 UTC) or a
  ``since`` (ISO, which overrides it) lists every message in it; neither, and nothing is read;
- ``order`` newest (the default) or oldest first, ``role`` user or assistant;
- one entry per message (its time UTC, chat title, role, up to 300 characters); a message
  after 30 minutes or more with nothing says so first (``gap``);
- the answer is fitted to what a chat turn keeps of a tool's answer (``ANSWER_TOKENS``, as
  handlers_my_actions does), and what it leaves out is counted, with the ``since`` that reads
  on (oldest first) or the time it stops ``before`` (newest first).
A call with no person behind it (a widget visitor, a headless agent) reads nothing.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.context_guard import count_tokens

from .actions_search import ORDERS, ROLES
from .handlers_search import NO_PERSON_CHATS, _bounded, chat_search_terms
from .my_actions_window import WINDOWS, Window, when_said, window_of

logger = logging.getLogger(__name__)

NEWEST, OLDEST = ORDERS
CHAT_SEARCH_DAYS, CHAT_SEARCH_MAX_DAYS = 30, 365   # a query's look-back with no window or since
CHAT_SEARCH_LIMIT, CHAT_SEARCH_MAX_LIMIT = 20, 100
CHAT_SNIPPET_CHARS = 300
GAP_MINUTES = 30
# What a chat turn keeps of one tool's answer is 2,000 tokens (consumers/chatbot/tool_cut.py);
# this answer stays under it, with room for the formatter's heading (as handlers_my_actions).
ANSWER_TOKENS = 1900
DATE = "%Y-%m-%d %H:%M UTC"
GAP = "after {hours}{minutes} min with nothing"
HOURS = "{hours} h "
NOTHING_TO_READ = ("Say what to read: a query (the words to find), or a window ({windows}) or a since time "
                   "(ISO, UTC) to list the messages in. Nothing was read.")
BAD_ORDER = "order must be newest or oldest; got {said!r}. Nothing was read."
BAD_ROLE = "role must be user or assistant; got {said!r}. Nothing was read."
LOOKBACK = "in the last {days} days"
WHOSE = {"user": "messages you sent", "assistant": "replies of mine", None: "messages"}
FOUND = "{count} of the {matched} {whose}{matching} {said} (from {start} to {end} UTC), {order} first."
NOTHING_FOUND = "No {whose}{matching} {said} (from {start} to {end} UTC)."
MATCHING = " holding '{query}'"
LEFT_SINCE = (" {count} more, from {when} UTC on, are not in it: call platform_search_chat_history again with "
              "since={since} and the same order and role for those.")
LEFT_BEFORE = (" {count} older ones, before {when} UTC, are not in it: read the window oldest first, or with a "
               "since, for those.")
CHAT_SEARCH_FAILED = "The chat history search could not run, so nothing was read; the detail is in the server log."

# The person's own messages in this workspace, in the window, holding every term (none: all of them).
_OLDEST_FIRST_SQL = """
    SELECT m.id, m.role, m.parts, m.created_at, c.title AS chat_title, COUNT(*) OVER () AS matched
    FROM messages m
    JOIN chats c ON c.id = m.chat_id
    WHERE c.user_id = :user_id
      AND c.workspace_id = CAST(:workspace_id AS uuid)
      AND m.workspace_id = CAST(:workspace_id AS uuid)
      AND m.created_at >= :since AND m.created_at < :until
      AND (CAST(:role AS text) IS NULL OR m.role = CAST(:role AS text))
      AND NOT EXISTS (
          SELECT 1 FROM unnest(CAST(:terms AS text[])) AS term
          WHERE NOT EXISTS (
              SELECT 1 FROM jsonb_array_elements(
                  CASE WHEN jsonb_typeof(m.parts) = 'array' THEN m.parts ELSE '[]'::jsonb END) AS p
              WHERE p->>'text' ILIKE '%' || term || '%'
          )
      )
    ORDER BY m.created_at ASC, m.id ASC
    LIMIT :lim
"""
_NEWEST_FIRST_SQL = """
    SELECT m.id, m.role, m.parts, m.created_at, c.title AS chat_title, COUNT(*) OVER () AS matched
    FROM messages m
    JOIN chats c ON c.id = m.chat_id
    WHERE c.user_id = :user_id
      AND c.workspace_id = CAST(:workspace_id AS uuid)
      AND m.workspace_id = CAST(:workspace_id AS uuid)
      AND m.created_at >= :since AND m.created_at < :until
      AND (CAST(:role AS text) IS NULL OR m.role = CAST(:role AS text))
      AND NOT EXISTS (
          SELECT 1 FROM unnest(CAST(:terms AS text[])) AS term
          WHERE NOT EXISTS (
              SELECT 1 FROM jsonb_array_elements(
                  CASE WHEN jsonb_typeof(m.parts) = 'array' THEN m.parts ELSE '[]'::jsonb END) AS p
              WHERE p->>'text' ILIKE '%' || term || '%'
          )
      )
    ORDER BY m.created_at DESC, m.id DESC
    LIMIT :lim
"""
_SQL = {OLDEST: _OLDEST_FIRST_SQL, NEWEST: _NEWEST_FIRST_SQL}


class Read(NamedTuple):
    """What a call asks to read: its words, window, order, role and limit."""

    query: str
    terms: List[str]
    window: Window
    order: str
    role: Optional[str]
    limit: int


class Entry(NamedTuple):
    """One message as read: its time, and the entry the answer lists."""

    at: datetime
    shown: Dict[str, Any]


async def search_chat_history(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """The asking person's own messages in this workspace: those holding every word of ``query``,
    or every one in a window, newest or oldest first, fitted to what the chat keeps."""
    user_id = params.get("_driving_user_id")
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        return {"success": False, "error": NO_PERSON_CHATS}
    read, wrong = read_asked(params)
    if read is None:
        return {"success": False, "error": wrong}
    try:
        with db.begin_nested():   # a failed read rolls back to here, not the turn's whole session
            rows = db.execute(text(_SQL[read.order]), {
                "user_id": user_id, "workspace_id": str(workspace_id), "since": _naive(read.window.start),
                "until": _naive(read.window.end), "role": read.role, "terms": read.terms, "lim": read.limit + 1,
            }).fetchall()
    except Exception:
        logger.exception("[PlatformExecutor] Chat history read failed")
        return {"success": False, "error": CHAT_SEARCH_FAILED}
    matched = int(rows[0].matched) if rows else 0
    return answer(read, entries(rows, read.order, gaps=not read.terms), matched)


def read_asked(params: Dict[str, Any], now: Optional[datetime] = None) -> Tuple[Optional[Read], Optional[str]]:
    """The read ``params`` ask for, or None and why it cannot be done (in words)."""
    query = str(params.get("query") or "").strip()
    terms = chat_search_terms(query)
    order = str(params.get("order") or NEWEST).strip().lower()
    if order not in ORDERS:
        return None, BAD_ORDER.format(said=params.get("order"))
    role = params.get("role")
    role = str(role).strip().lower() if role not in (None, "") else None
    if role is not None and role not in ROLES:
        return None, BAD_ROLE.format(said=params.get("role"))
    window, wrong = _window(params, bool(terms), now)
    if window is None:
        return None, wrong
    limit = max(_bounded(params.get("limit"), CHAT_SEARCH_LIMIT, CHAT_SEARCH_MAX_LIMIT), 1)
    return Read(query, terms, window, order, role, limit), None


def _window(params: Dict[str, Any], has_terms: bool, now: Optional[datetime]) -> Tuple[Optional[Window], Optional[str]]:
    """``since`` or ``window`` (my_actions_window); with neither, a query's ``days``; else nothing."""
    if params.get("since") not in (None, "") or params.get("window") not in (None, ""):
        return window_of({"since": params.get("since"), "window": params.get("window")}, now)
    if not has_terms:
        return None, NOTHING_TO_READ.format(windows=", ".join(WINDOWS))
    end = now or datetime.now(timezone.utc)
    days = max(_bounded(params.get("days"), CHAT_SEARCH_DAYS, CHAT_SEARCH_MAX_DAYS), 1)
    return Window(end - timedelta(days=days), end, LOOKBACK.format(days=days)), None


def entries(rows: List[Any], order: str, *, gaps: bool) -> List[Entry]:
    """Each message as an entry, in ``order``; with ``gaps``, a message after ``GAP_MINUTES`` or
    more with nothing (in the messages read) carries that gap first."""
    listed = [Entry(_utc(row.created_at), _shown(row)) for row in rows]
    chronological = listed if order == OLDEST else listed[::-1]
    marked = [chronological[0]] if chronological else []
    for before, entry in zip(chronological, chronological[1:]):
        silent = entry.at - before.at
        marked.append(Entry(entry.at, {"gap": gap_said(silent), **entry.shown})
                      if gaps and silent >= timedelta(minutes=GAP_MINUTES) else entry)
    return marked if order == OLDEST else marked[::-1]


def gap_said(silent: timedelta) -> str:
    """'after 2 h 0 min with nothing', 'after 45 min with nothing'."""
    hours, minutes = divmod(int(silent.total_seconds()) // 60, 60)
    return GAP.format(hours=HOURS.format(hours=hours) if hours else "", minutes=minutes)


def answer(read: Read, listed: List[Entry], matched: int) -> Dict[str, Any]:
    """The answer with as many entries as fit ``ANSWER_TOKENS``; the rest counted, with how to read them."""
    low, high = 0, min(read.limit, len(listed))
    while low < high:   # the most entries whose whole answer fits
        middle = (low + high + 1) // 2
        if count_tokens(_measured(_answer(read, listed, middle, matched))) <= ANSWER_TOKENS:
            low = middle
        else:
            high = middle - 1
    return _answer(read, listed, low, matched)


def _answer(read: Read, listed: List[Entry], kept: int, matched: int) -> Dict[str, Any]:
    window = read.window
    found = (FOUND if matched else NOTHING_FOUND).format(
        count=kept, matched=matched, whose=WHOSE[read.role], said=window.said, order=read.order,
        matching=MATCHING.format(query=read.query) if read.terms else "",
        start=when_said(window.start), end=when_said(window.end))
    left, note = _left_out(read, listed, kept, matched)
    return {"success": True, "query": read.query or None, "window": window.said, "order": read.order,
            "role": read.role, "total": kept, "message": found + note,
            "results": [entry.shown for entry in listed[:kept]], "left_out": left}


def _left_out(read: Read, listed: List[Entry], kept: int, matched: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """How many matched messages the answer leaves out, and the since (oldest first) or the time
    it stops before (newest first) that reads them; (None, '') when none is."""
    count = matched - kept
    if count <= 0:
        return None, ""
    if read.order == OLDEST:
        since = listed[kept].at if kept < len(listed) else read.window.end
        return ({"count": count, "since": since.isoformat()},
                LEFT_SINCE.format(count=count, when=when_said(since), since=since.isoformat()))
    before = listed[kept - 1].at if kept else read.window.end
    return {"count": count, "before": before.isoformat()}, LEFT_BEFORE.format(count=count, when=when_said(before))


def _measured(answer_: Dict[str, Any]) -> str:
    """The answer as the result formatter gives it to the model (its JSON, success left out)."""
    return json.dumps({key: value for key, value in answer_.items() if key != "success"}, default=str, indent=2)


def _shown(row: Any) -> Dict[str, Any]:
    parts = row.parts if isinstance(row.parts, list) else []
    said = " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))
    return {"date": _utc(row.created_at).strftime(DATE), "chat_title": row.chat_title, "role": row.role,
            "content": said[:CHAT_SNIPPET_CHARS]}


def _utc(moment: datetime) -> datetime:
    """A naive time is the database's own, UTC."""
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def _naive(moment: datetime) -> datetime:
    """The database's naive UTC."""
    return _utc(moment).replace(tzinfo=None)


__all__ = ["ANSWER_TOKENS", "GAP_MINUTES", "NEWEST", "OLDEST", "answer", "entries", "gap_said", "read_asked",
           "search_chat_history"]
