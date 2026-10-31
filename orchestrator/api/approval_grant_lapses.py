"""P256-FIX-N17-3 (F439, night 17): a click on a card that can no longer run says why, and runs nothing.

Night 17 clicked #2607 thirteen hours after it expired. /grant checked only that it was pending,
flipped it to granted and resumed the call; the call's gate found the yes expired (``is_authorising``),
asked again, and the chat was told "Your click ran: … it did not go through: no reason given".

``the_card_can_run`` is /grant's status check: a card that is not pending is refused as before
("Grant is not pending"), one the platform replaced by a newer card for the same call
(``stale_cards.SUPERSEDED_BY``) or closed on its expiry says so, and a pending call's card past its
``expires_at`` is closed (``EXPIRED``, ``system:expired``), committed before the refusal (``get_db``
rolls back what is left), and refused with its time. Nothing is resumed and no decision is
remembered. A board ticket's card (its run re-queues under the grant) and an agent's send card
(a session ticket is parked on it) keep today's path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException

from core.models.approval_grants import GrantStatus, SUBJECT_TOOL_CALL
from core.services.approval_grants import expire_pending_grants, is_lapsed
from modules.tools.discovery.agent_sends import send_marker
from modules.tools.discovery.my_actions_reads import at_said
from modules.tools.execution.stale_cards import EXPIRED_BY, SUPERSEDED_BY

REFUSED = 422
NOT_PENDING = "Grant is not pending (status: {status})"
EXPIRED = "This card expired {at}; nothing ran. Ask Auto again for a fresh card."
NO_TIME = "earlier"
REPLACED = ("Grant is not pending: it was replaced by a newer card for the same request; nothing ran. "
            "Approve the newer card.")


def the_card_can_run(db: Any, grant: Any, now: Optional[datetime] = None) -> None:
    """Raise the 422 for a card a click cannot run: not pending, replaced, or past its expiry."""
    if grant.status != GrantStatus.PENDING.value:
        raise HTTPException(status_code=REFUSED, detail=_not_pending(grant))
    if not _lapses_here(grant, now):
        return
    if expire_pending_grants(db, grant.workspace_id, [grant.id], revoked_by=EXPIRED_BY, now=now) != 1:
        db.refresh(grant)   # another click decided it a moment before: say what it is now
        raise HTTPException(status_code=REFUSED, detail=_not_pending(grant))
    db.commit()
    raise HTTPException(status_code=REFUSED, detail=EXPIRED.format(at=_expired_at(grant)))


def _not_pending(grant: Any) -> str:
    """Why a decided card runs nothing, in the owner's words where the platform closed it."""
    if grant.status == GrantStatus.EXPIRED.value and grant.revoked_by == SUPERSEDED_BY:
        return REPLACED
    if grant.status == GrantStatus.EXPIRED.value and grant.revoked_by == EXPIRED_BY:
        return EXPIRED.format(at=_expired_at(grant))
    return NOT_PENDING.format(status=grant.status)


def _lapses_here(grant: Any, now: Optional[datetime]) -> bool:
    """A call's card past its expiry, that is neither a board ticket's nor an agent's send."""
    if grant.subject_type != SUBJECT_TOOL_CALL or not is_lapsed(grant, now=now):
        return False
    details = grant.details if isinstance(grant.details, dict) else {}
    return details.get("board_task_id") is None and send_marker(grant) is None


def _expired_at(grant: Any) -> str:
    """'at 05:12 on 15 Oct' (UTC); 'earlier' for a row that keeps no expiry time."""
    exp = getattr(grant, "expires_at", None)
    if exp is None:
        return NO_TIME
    return at_said(exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc))


__all__ = ["EXPIRED", "REPLACED", "the_card_can_run"]
