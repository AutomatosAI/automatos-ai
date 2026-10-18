"""P256-FIX-N17-3 (F439, nights 16–17): a call's card is never a stale one from another chat.

Night 16: five run-nows of playbook #81, from four chats, each got "Card raised", and each was
#2607, raised the night before in another chat: ``issue_tool_grant`` reused any pending row of
the same call (workspace, action, params), whatever its chat, turn or age, and nothing moves a
call's card to expired. Night 17 reused it 13 hours after it expired, and its click answered
"it did not go through: no reason given". The reused row kept the first chat's details, so the
click's line and its resume went to that chat.

Before the pending row of the same call is reused (``reusable``):

- a row past its ``expires_at`` is closed (``system:expired``) and a fresh card is raised;
- a chat turn (``conversation_id`` and ``turn_id``: only the chat's ``build_tool_caller_context``
  sets a turn) that finds a card raised in ANOTHER chat of the same person closes it
  (``system:superseded``) and raises a fresh one in this chat, so the click's line and its resume
  go here; another person's card is theirs, and is met as already waiting;
- a row an agent's send waits on (``agent_sends.AGENT_SEND``: a session ticket is parked on it)
  is reused as today, and so is any row an agent's, a board run's or a playbook step's call finds.

A close that matches no row was decided a moment before: a click (the row is reused, as today)
or another turn's close (a fresh card is raised; the closed one could never be clicked). A chat
turn that reuses a card raised on an earlier turn has its ask marked ``already_waiting`` (the
card's id and when it was raised), so the model is told it is that card, not a new one
(``card_raised``).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

EXPIRED_BY = "system:expired"
SUPERSEDED_BY = "system:superseded"
CHAT_LANE = "chat"
ALREADY_WAITING = "already_waiting"
DRIVING_USER = "driving_user_id"


def on_a_chat_turn(caller_context: Any) -> bool:
    """Whether the call comes from a person's chat turn (a conversation and a turn)."""
    ctx = caller_context if isinstance(caller_context, dict) else {}
    return bool(ctx.get("conversation_id")) and bool(ctx.get("turn_id"))


def the_pending_card(db: Any, workspace_id: Any, subject_id: str, caller_context: Any) -> Optional[Any]:
    """The pending card of this call (``find_pending_grant``) to reuse, or None to raise a fresh one."""
    from core.models.approval_grants import SUBJECT_TOOL_CALL
    from core.services.approval_grants import find_pending_grant

    existing = find_pending_grant(db, workspace_id, subject_type=SUBJECT_TOOL_CALL, subject_id=subject_id)
    return reusable(db, workspace_id, existing, caller_context)


def reusable(db: Any, workspace_id: Any, existing: Any, caller_context: Any,
             now: Optional[datetime] = None) -> Optional[Any]:
    """The pending row to reuse as this call's card, or None when a fresh card is raised (the
    row lapsed, or a chat turn found its person's card from another chat: it is closed here).
    Caller owns the txn."""
    if existing is None:
        return None
    closing = _closed_as(existing, caller_context, now)
    if closing is None:
        return existing
    from core.services.approval_grants import expire_pending_grants

    closed = expire_pending_grants(db, workspace_id, [existing.id], revoked_by=closing, now=now)
    _forget(db, existing)
    if closed != 1:
        return _after_the_race(db, existing)
    logger.info("[stale_cards] grant %s closed (%s): a fresh card is raised", existing.id, closing)
    return None


def _after_the_race(db: Any, existing: Any) -> Optional[Any]:
    """The close matched nothing: the row was decided a moment before. A click's yes stands as
    the card, as today; a row another turn closed (or the owner declined) is no card to click."""
    from core.models.approval_grants import GrantStatus

    refresh = getattr(db, "refresh", None)
    if callable(refresh):
        refresh(existing)
    stands = (GrantStatus.PENDING.value, GrantStatus.GRANTED.value)
    return existing if getattr(existing, "status", None) in stands else None


def with_already_waiting(ask: Dict[str, Any], grant: Any, caller_context: Any) -> Dict[str, Any]:
    """The ask, marked ``already_waiting`` ({grant_id, since: when it was raised, ISO UTC}) when
    a chat turn reuses a card that no earlier call of this turn raised; else the ask unchanged."""
    since = getattr(grant, "requested_at", None)
    if not on_a_chat_turn(caller_context) or not isinstance(since, datetime):
        return ask
    if str(_details(grant).get("turn_id") or "") == str(caller_context.get("turn_id")):
        return ask
    from modules.tools.discovery.my_actions_reads import iso_utc

    return {**ask, ALREADY_WAITING: {"grant_id": grant.id, "since": iso_utc(since)}}


def _closed_as(existing: Any, caller_context: Any, now: Optional[datetime]) -> Optional[str]:
    """Why the row is closed rather than reused (its ``revoked_by``), or None to reuse it."""
    from core.services.approval_grants import is_lapsed
    from modules.tools.discovery.agent_sends import send_marker

    if send_marker(existing) is not None:
        return None
    if is_lapsed(existing, now=now):
        return EXPIRED_BY
    return SUPERSEDED_BY if _from_another_chat(existing, caller_context) else None


def _from_another_chat(existing: Any, caller_context: Any) -> bool:
    """A chat turn found a card raised on the chat lane of a different conversation, by the same
    person (a card that names another person is left to them)."""
    if not on_a_chat_turn(caller_context):
        return False
    details = _details(existing)
    raised_in = details.get("conversation_id")
    if details.get("lane") != CHAT_LANE or not raised_in or str(raised_in) == str(caller_context["conversation_id"]):
        return False
    snapshot = details.get("caller_context") if isinstance(details.get("caller_context"), dict) else {}
    theirs, mine = snapshot.get(DRIVING_USER), caller_context.get(DRIVING_USER)
    return theirs is None or mine is None or str(theirs) == str(mine)


def _details(grant: Any) -> Dict[str, Any]:
    details = getattr(grant, "details", None)
    return details if isinstance(details, dict) else {}


def _forget(db: Any, grant: Any) -> None:
    """The close is one UPDATE (``synchronize_session=False``): the session's copy of the row
    is expired so the next read sees it closed."""
    expire = getattr(db, "expire", None)
    if callable(expire):
        expire(grant)


__all__ = ["ALREADY_WAITING", "EXPIRED_BY", "SUPERSEDED_BY", "on_a_chat_turn", "reusable", "the_pending_card",
           "with_already_waiting"]
