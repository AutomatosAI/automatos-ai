"""P256-FIX-N14-3 (night 14, F417): the owner's clicks, and what became of Auto's asks.

Night 14, "what did I decide about #…" found 1 click of 8. The grants table holds every click:

- ``clicks_in``: the asking person's own clicks on a call's approval card (a ``tool_call`` grant,
  as ``click_memory`` remembers them; a question has its own path), approved or declined, in this
  workspace, with what came of each and the card's board number: the tickets its call names
  (``click_memory.numbers_of_the_card``), else the number its card's words show.
- ``settled``: a write receipt saved as ``waiting`` stays so in its message; the card it raised
  may since have been decided. Each waiting item is read against its grant (the same action, in
  the same chat, asked no later than the message was saved, the newest): still pending, it waits;
  otherwise it is ``decided``, with the decision.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, List, Optional

from .my_actions_reads import WAITING, Found, Item, at_said, iso_utc
from .my_actions_window import Window

MAX_CLICKS = 200
MAX_ASKS = 200       # the newest call cards read to settle the window's waiting writes
APPROVED, DECLINED, LAPSED = "approved", "declined", "lapsed"
DECIDED = "decided"
USER_ACTOR = "user:"
NUMBER_SAID = re.compile(r"#\d+(?:\.\d+)?")
# '#' and the most digits a board number has (services.ticket_numbers.MAX_SEARCHED_DIGITS).
MAX_NUMBER_CHARS = 10


def clicks_in(db: Any, workspace_id: Any, user_id: int, window: Window) -> Found:
    """``user_id``'s own clicks on call cards in ``window`` (approved or declined), oldest first; a cap
    leaves out the newest (P256-FIX-N15-4), and ``rest_from`` says when they begin."""
    from sqlalchemy import and_, func, or_

    from core.models.approval_grants import KIND_QUESTION, SUBJECT_TOOL_CALL, ApprovalGrant, GrantStatus

    refs = _actor_refs(db, user_id)
    said_yes = and_(ApprovalGrant.granted_by.in_(refs), ApprovalGrant.granted_at >= window.start,
                    ApprovalGrant.granted_at < window.end)
    said_no = and_(ApprovalGrant.status == GrantStatus.DENIED.value, ApprovalGrant.revoked_by.in_(refs),
                   ApprovalGrant.revoked_at >= window.start, ApprovalGrant.revoked_at < window.end)
    grants = (db.query(ApprovalGrant)
              .filter(ApprovalGrant.workspace_id == workspace_id, ApprovalGrant.subject_type == SUBJECT_TOOL_CALL,
                      ApprovalGrant.kind != KIND_QUESTION, or_(said_yes, said_no))
              .order_by(func.coalesce(ApprovalGrant.granted_at, ApprovalGrant.revoked_at).asc(), ApprovalGrant.id.asc())
              .limit(MAX_CLICKS + 1).all())
    clicks = sorted((_click(db, grant) for grant in grants[:MAX_CLICKS]), key=lambda item: item["at"])
    rest = grants[MAX_CLICKS] if len(grants) > MAX_CLICKS else None
    return Found(clicks, _utc(rest.granted_at or rest.revoked_at) if rest is not None else None)


def settled(db: Any, workspace_id: Any, items: List[Item], window: Window) -> List[Item]:
    """``items`` with each waiting write whose card has since been decided marked ``decided``."""
    from core.models.approval_grants import KIND_QUESTION, SUBJECT_TOOL_CALL, ApprovalGrant

    actions = {item["action"] for item in items if item["status"] == WAITING}
    if not actions:
        return items
    grants = (db.query(ApprovalGrant)
              .filter(ApprovalGrant.workspace_id == workspace_id, ApprovalGrant.subject_type == SUBJECT_TOOL_CALL,
                      ApprovalGrant.kind != KIND_QUESTION, ApprovalGrant.tool_name.in_(actions),
                      ApprovalGrant.requested_at < window.end)
              .order_by(ApprovalGrant.requested_at.desc()).limit(MAX_ASKS).all())
    now = datetime.now(timezone.utc)
    return [_as_it_stands(item, grants, now) if item["status"] == WAITING else item for item in items]


def decision_of(grant: Any, now: datetime) -> Optional[str]:
    """APPROVED, DECLINED, or LAPSED (it expired, or was withdrawn, with no yes); None while it waits."""
    from core.models.approval_grants import GrantStatus

    if grant.status == GrantStatus.DENIED.value:
        return DECLINED
    if grant.granted_at is not None:
        return APPROVED  # a single-use yes is retired by its own run (revoked, system:consumed)
    expires = _utc(grant.expires_at) if grant.expires_at is not None else None
    if grant.status == GrantStatus.PENDING.value and (expires is None or expires > now):
        return None
    return LAPSED


def _as_it_stands(item: Item, grants: List[Any], now: datetime) -> Item:
    saved = datetime.fromisoformat(item["at"])
    grant = next((g for g in grants if g.tool_name == item["action"] and _chat_of(g) == item["chat_id"]
                  and g.requested_at is not None and _utc(g.requested_at) <= saved), None)
    decision = decision_of(grant, now) if grant is not None else None
    return {**item, "status": DECIDED, "decision": decision} if decision else item


def _click(db: Any, grant: Any) -> Item:
    from services.click_memory import NOTHING_RAN, card_words, numbers_of_the_card, outcome_of

    approved = decision_of(grant, datetime.now(timezone.utc)) == APPROVED
    at = grant.granted_at if approved else grant.revoked_at
    card = card_words(grant)
    details = grant.details if isinstance(grant.details, dict) else {}
    numbers = list(dict.fromkeys([*numbers_of_the_card(db, grant), *board_numbers_said(db, grant.workspace_id, card)]))
    outcome = outcome_of(details.get("executed_result")) if approved else NOTHING_RAN
    return {"at": iso_utc(at), "when": at_said(at), "decision": APPROVED if approved else DECLINED,
            "card": card, "numbers": numbers, "outcome": outcome or None, "grant_id": grant.id}


def board_numbers_said(db: Any, workspace_id: Any, card: str) -> List[str]:
    """The '#N' a card's words show that are a ticket's board number in this workspace.
    P256-FIX-N15-9 (F429): a card also writes an agent's, a document's or another card's own id
    with a '#' ('agent #12', 'send card #1607'), and none of those is a ticket's number."""
    from core.models.core import BoardTask
    from services.ticket_numbers import resolve_ticket_ref

    said = NUMBER_SAID.findall(card)
    whole = {int(n[1:]) for n in said if "." not in n and len(n) <= MAX_NUMBER_CHARS}
    known = {seq for (seq,) in db.query(BoardTask.workspace_seq).filter(
        BoardTask.workspace_id == workspace_id, BoardTask.workspace_seq.in_(whole)).all()} if whole else set()

    def is_a_card(number: str) -> bool:   # a mission step's (#0051.3) is read one at a time: they are rare
        if "." in number:
            return resolve_ticket_ref(db, workspace_id, number) is not None
        return int(number[1:]) in known

    return [n for n in said if is_a_card(n)]


def _actor_refs(db: Any, user_id: int) -> List[str]:
    """How the grants table names this person: 'user:<users.id>', and 'user:<Clerk id>' (the hosted
    edition's approval routes record the Clerk id)."""
    from core.models.core import User

    clerk = db.query(User.clerk_user_id).filter(User.id == user_id).scalar()
    return [f"{USER_ACTOR}{ref}" for ref in (user_id, clerk) if ref not in (None, "")]


def _chat_of(grant: Any) -> Optional[str]:
    details = grant.details if isinstance(grant.details, dict) else {}
    chat = details.get("conversation_id")
    return str(chat) if chat else None


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


__all__ = ["APPROVED", "DECIDED", "DECLINED", "LAPSED", "clicks_in", "decision_of", "settled"]
