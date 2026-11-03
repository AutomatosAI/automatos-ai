"""PRD-268 US-C11c (§6.3, FR-9, FR-14, K7, K9, K10) — Auto's follow-up tools.

``platform_schedule_follow_up`` keeps Auto's "I'll check again at 15:10";
``platform_list_follow_ups`` and ``platform_cancel_follow_up`` read and cancel the person's
own. The rows, the lock and the caps are ``services.proactive.follow_ups``'.

Who a call is for comes from the server only: ``follow_up_context.carries_the_follow_up``
hands each tool ``_follow_up`` (the driving person, their chat and the turn's language),
and ``exec_platform`` the calling agent as ``_agent_id``. Each tool refuses, with a sentence
the model reads (``services.proactive.follow_up_strings``) and an ``error_code``, a call:

* made for nobody, or for someone who does not belong to the workspace (``no_person``);
* from an agent that is not the workspace's Auto (``not_auto``).

Scheduling also refuses (K10) a call outside the person's own chat (``no_conversation``), a
workspace whose ``effective_mode`` is not live (``not_live``: a follow-up that will not
post must not be promised), a workspace whose background work only the trial could pay
for (``no_background``: the scheduler skips such a run, ``services.credits.routing
.background_allowed``; a check that cannot be read refuses too), a paused person
(``paused``), a follow-up about a Mission, a
playbook run or a ticket, or whose reason names a ticket by its board number
(``use_a_watch``: the watcher supervises those), a reason or a time it cannot read, and
any cap (``follow_up_cap``, with the cap's name and value). The check-in lane never runs
these tools (``check_in_lane.CHAT_ONLY_ACTIONS``), so no check-in and no follow-up's own
turn schedules another (§12-Q9).

Every tool runs on a session of its own, on a worker thread: their reads are synchronous,
the workspace's lock may wait, and none commits the chat turn's session.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.proactive.follow_up_caps import (
    ABOUT_KINDS, WATCHED_KINDS, CapBroken, FollowUpCaps, FollowUpTimeRefused, follow_up_time, time_broken,
)
from modules.tools.discovery.follow_up_context import FOLLOW_UP_PARAM
from services.proactive import follow_ups
from services.proactive.follow_up_strings import cap_key, locale_of, say

logger = logging.getLogger(__name__)

__all__ = ["cancel_follow_up", "list_follow_ups", "names_a_ticket", "schedule_follow_up"]

NO_PERSON, NO_CONVERSATION, NOT_AUTO = "no_person", "no_conversation", "not_auto"
NOT_LIVE, NO_BACKGROUND, PAUSED, USE_A_WATCH = "not_live", "no_background", "paused", "use_a_watch"
NO_REASON, LONG_REASON, BAD_ABOUT, BAD_STATUS, BAD_ID = "no_reason", "long_reason", "bad_about", "bad_status", "bad_id"
CAP_CODE = "follow_up_cap"
ALREADY = {follow_ups.FIRED: "already_fired", follow_ups.CANCELLED: "already_cancelled",
           follow_ups.MISSED: "already_missed"}
ABOUT_ID_CHARS = 64
STATUS_ALL = "all"
# A follow-up's number: ASCII digits, as many as a row id can have.
_FOLLOW_UP_NUMBER = re.compile(r"#?(\d{1,18})", re.ASCII)
# Postgres refuses a NUL in text; a reason never carries one.
NUL = "\x00"
# A ticket named by its board number, as services.ticket_numbers reads one: "#0042", "#0051.3",
# or "ticket 42".
_TICKET_MENTION = re.compile(r"(?<![\w#])(#\d+(?:\.\d+)?)(?!\w)|\bticket\s+(#?\d+(?:\.\d+)?)(?!\w)", re.IGNORECASE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _refused(locale: str, code: str, **values: Any) -> Dict[str, Any]:
    return {"success": False, "error": say(locale, code, **values), "error_code": code}


def _own_session(db: Session) -> Session:
    """A session of its own on the chat turn's database."""
    return Session(bind=db.get_bind())


def names_a_ticket(reason: str) -> bool:
    """Whether ``reason`` names a ticket by its board number ("#0042", "ticket 42")."""
    from services.ticket_numbers import is_bare_ref, is_number_ref

    for match in _TICKET_MENTION.finditer(reason):
        token = match.group(1) or match.group(2)
        if is_number_ref(token) or is_bare_ref(token):
            return True
    return False


def _own_chat(db: Session, workspace_id: UUID, user_id: int, chat_id: Any) -> Optional[UUID]:
    """The turn's chat when it is the person's own chat in this workspace, else None."""
    try:
        chat = UUID(str(chat_id))
    except (TypeError, ValueError):
        return None
    row = db.execute(text("SELECT 1 FROM chats WHERE id = CAST(:id AS uuid) AND workspace_id = CAST(:ws AS uuid) "
                          "AND user_id = :user"), {"id": str(chat), "ws": str(workspace_id), "user": user_id}).first()
    return chat if row is not None else None


def _who_refusal(db: Session, workspace_id: UUID, caller: Mapping[str, Any], agent_id: Any) -> Optional[str]:
    """``no_person`` when the call is made for nobody in the workspace; ``not_auto`` when the
    calling agent is not its Auto; None when it is Auto, for a person who belongs here (any
    role, the local operator too: ``services.push.recipients``)."""
    from services.push.recipients import AUTO_MESSAGE, recipients

    user_id = caller.get("user_id")
    if user_id is None or not recipients(db, workspace_id, AUTO_MESSAGE, int(user_id)):
        return NO_PERSON
    auto = follow_ups.auto_agent_id(db, workspace_id)
    if auto is None or isinstance(agent_id, bool) or agent_id != auto:
        return NOT_AUTO
    return None


def _background_allowed(db: Session, workspace_id: UUID) -> bool:
    """Whether something other than the trial can pay for the follow-up's run: the scheduler
    skips it otherwise (``scheduled_task_steps.skipped_for_credits``). A check that cannot be
    read says no, as ``gates._credits_refusal`` refuses a check-in."""
    from services.credits.routing import background_allowed

    try:
        with db.begin_nested():
            return bool(background_allowed(db, workspace_id))
    except Exception:  # noqa: BLE001 — a check that did not complete refuses the follow-up
        logger.exception("[follow-up] background_allowed could not be read for %s: schedule refused", workspace_id)
        return False


def _schedule_who_refusal(db: Session, workspace_id: UUID, caller: Mapping[str, Any], agent_id: Any) -> Optional[str]:
    """K10: Auto, in the person's own chat, in a live workspace whose background work more
    than the trial pays for, for a person not paused."""
    from services.proactive.switches import MODE_LIVE, effective_mode, person_paused

    refusal = _who_refusal(db, workspace_id, caller, agent_id)
    if refusal is not None:
        return refusal
    if _own_chat(db, workspace_id, int(caller["user_id"]), caller.get("chat_id")) is None:
        return NO_CONVERSATION
    if effective_mode(db, workspace_id) != MODE_LIVE:
        return NOT_LIVE
    if not _background_allowed(db, workspace_id):
        return NO_BACKGROUND
    return PAUSED if person_paused(db, workspace_id, int(caller["user_id"])) else None


def _about(value: Any) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """``about`` as it is kept (``{kind, id}``), or why it is refused."""
    if value in (None, {}, ""):
        return None, None
    if not isinstance(value, dict) or value.get("kind") not in ABOUT_KINDS:
        return None, BAD_ABOUT
    ident = " ".join(str(value.get("id") or "").replace(NUL, "").split())[:ABOUT_ID_CHARS]
    if value["kind"] in WATCHED_KINDS or names_a_ticket(ident):
        return None, USE_A_WATCH
    return ({"kind": value["kind"], "id": ident} if ident else {"kind": value["kind"]}), None


def _reason(value: Any, caps: FollowUpCaps) -> Tuple[str, Optional[str]]:
    """The reason, trimmed, or why it is refused: none, too long, or naming a ticket."""
    reason = value.replace(NUL, "").strip() if isinstance(value, str) else ""
    if not reason:
        return "", NO_REASON
    if len(reason) > caps.reason_chars:
        return "", LONG_REASON
    return (reason, USE_A_WATCH) if names_a_ticket(reason) else (reason, None)


def _cap_refused(locale: str, broken: CapBroken) -> Dict[str, Any]:
    return {"success": False, "error": say(locale, cap_key(broken.cap), limit=broken.limit), "error_code": CAP_CODE,
            "cap": broken.cap, "limit": broken.limit}


def _request(params: Mapping[str, Any], workspace_id: UUID, caps: FollowUpCaps, now: datetime,
             chat: UUID) -> Tuple[Optional[follow_ups.FollowUpRequest], Optional[Dict[str, Any]]]:
    """The follow-up ``params`` ask for, or the refusal its arguments or the time caps give."""
    caller, locale = params[FOLLOW_UP_PARAM], locale_of(params[FOLLOW_UP_PARAM])
    about, refusal = _about(params.get("about"))
    reason, reason_refusal = _reason(params.get("reason"), caps)
    if refusal or reason_refusal:
        return None, _refused(locale, refusal or reason_refusal, limit=caps.reason_chars)
    try:
        at = follow_up_time(params.get("in_minutes"), params.get("at"), now)
    except FollowUpTimeRefused as refused:
        return None, _refused(locale, refused.code)
    broken = time_broken(caps, at, now)
    if broken is not None:
        return None, _cap_refused(locale, broken)
    return follow_ups.FollowUpRequest(workspace_id=workspace_id, user_id=int(caller["user_id"]), chat_id=chat,
                                      agent_id=int(params["_agent_id"]), at=at, reason=reason, about=about), None


def _schedule(db: Session, workspace_id: UUID, params: Mapping[str, Any], now: datetime) -> Dict[str, Any]:
    """Every K10 check, then the follow-up, on ``db`` (a session of its own)."""
    caller = params[FOLLOW_UP_PARAM]
    locale = locale_of(caller)
    refusal = _schedule_who_refusal(db, workspace_id, caller, params.get("_agent_id"))
    if refusal is not None:
        logger.info("[follow-up] schedule refused in %s: %s", workspace_id, refusal)
        return _refused(locale, refusal)
    caps = follow_ups.platform_caps()
    # The person's own chat: _schedule_who_refusal has just read it so.
    request, refused = _request(params, workspace_id, caps, now, UUID(str(caller["chat_id"])))
    if refused is not None:
        return refused
    scheduled = follow_ups.schedule_follow_up(db, request, caps, now)
    if isinstance(scheduled, CapBroken):
        return _cap_refused(locale, scheduled)
    at = request.at.isoformat()
    return {"success": True, "follow_up_id": scheduled, "at": at,
            "message": say(locale, "scheduled", follow_up_id=scheduled, at=at)}


def _on_own_session(db: Session, work: Any, *args: Any) -> Dict[str, Any]:
    """``work(session, *args)`` on a session of its own, closed after (it runs on a worker thread)."""
    session = _own_session(db)
    try:
        return work(session, *args)
    finally:
        session.close()


def _caller(params: Any) -> Optional[Mapping[str, Any]]:
    caller = params.get(FOLLOW_UP_PARAM) if isinstance(params, dict) else None
    return caller if isinstance(caller, dict) and caller.get("user_id") is not None else None


async def schedule_follow_up(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule a follow-up for the person whose chat turn this is, or say why not."""
    params = params or {}
    caller = _caller(params)
    if caller is None:
        return _refused(locale_of(params.get(FOLLOW_UP_PARAM)), NO_PERSON)
    return await asyncio.to_thread(_on_own_session, db, _schedule, UUID(str(workspace_id)), params, _now())


def _listed(db: Session, workspace_id: UUID, params: Mapping[str, Any]) -> Dict[str, Any]:
    caller = params[FOLLOW_UP_PARAM]
    locale, status = locale_of(caller), params.get("status") or follow_ups.PENDING
    if status not in (follow_ups.PENDING, STATUS_ALL):
        return _refused(locale, BAD_STATUS)
    refusal = _who_refusal(db, workspace_id, caller, params.get("_agent_id"))
    if refusal is not None:
        return _refused(locale, refusal)
    found = follow_ups.list_follow_ups(db, workspace_id, int(caller["user_id"]),
                                       only_pending=status == follow_ups.PENDING)
    return {"success": True, "follow_ups": [item.to_dict() for item in found], "count": len(found),
            "note": say(locale, "listed")}


async def list_follow_ups(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The person's own follow-ups (pending by default; ``status`` 'all' for every one). Reads only."""
    params = params or {}
    if _caller(params) is None:
        return _refused(locale_of(params.get(FOLLOW_UP_PARAM)), NO_PERSON)
    return await asyncio.to_thread(_on_own_session, db, _listed, UUID(str(workspace_id)), params)


def _follow_up_id(value: Any) -> Optional[int]:
    """``follow_up_id`` as a row id: a whole number, or its ASCII digits (with a '#') as text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 < value < 10 ** 18 else None
    match = _FOLLOW_UP_NUMBER.fullmatch(str(value or "").strip())
    return int(match.group(1)) if match else None


def _cancel(db: Session, workspace_id: UUID, params: Mapping[str, Any], follow_up_id: int) -> Dict[str, Any]:
    caller = params[FOLLOW_UP_PARAM]
    locale = locale_of(caller)
    refusal = _who_refusal(db, workspace_id, caller, params.get("_agent_id"))
    if refusal is not None:
        return _refused(locale, refusal)
    outcome, cancelled = follow_ups.cancel_follow_up(db, workspace_id, int(caller["user_id"]), follow_up_id)
    if cancelled is None:  # nothing cancelled now: already fired, cancelled or missed (follow_ups._not_cancelled)
        code = ALREADY.get(outcome, follow_ups.NOT_FOUND)
        return _refused(locale, code, follow_up_id=follow_up_id)
    return {"success": True, "follow_up_id": follow_up_id, "status": follow_ups.CANCELLED,
            "message": say(locale, "cancelled", follow_up_id=follow_up_id)}


async def cancel_follow_up(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Cancel the person's own pending follow-up, or say why not."""
    params = params or {}
    caller = _caller(params)
    if caller is None:
        return _refused(locale_of(params.get(FOLLOW_UP_PARAM)), NO_PERSON)
    follow_up_id = _follow_up_id(params.get("follow_up_id"))
    if follow_up_id is None:
        return _refused(locale_of(caller), BAD_ID)
    return await asyncio.to_thread(_on_own_session, db, _cancel, UUID(str(workspace_id)), params, follow_up_id)
