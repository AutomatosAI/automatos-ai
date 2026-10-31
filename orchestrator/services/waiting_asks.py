"""P256-FIX-N17-3 (F439, nights 16–17): the asks still waiting for one person's click.

"Did the stock check run?" (night 16 rows 168 and 347; night 17 row 460, two minutes after the
run-now raised card #2828) was answered from finished board cards: the turn read the team's
findings (cards in review or done) and the documents, never the person's own pending approvals,
so the run that waited for their click was never named.

``waiting_asks`` reads them: this workspace's call cards (``tool_call`` approvals) still pending
and not expired, raised in the last day for this person (the ``driving_user_id`` the card kept
from the chat turn that raised it), newest first, at most five; each as its card's words and
when it was raised ("at 04:12 on 14 Oct"). Another person's cards, questions and board tickets'
approvals are not theirs to click here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

LOOKBACK = timedelta(hours=24)
MAX_ASKS = 5
DRIVING_USER = ("caller_context", "driving_user_id")   # tool_grants._grant_details' snapshot


def waiting_asks(db: Any, workspace_id: Any, user_id: Any, now: datetime) -> List[Dict[str, str]]:
    """[{card, at}] for each call card of ``user_id`` waiting for their click, newest first."""
    from core.models.approval_grants import KIND_APPROVAL, SUBJECT_TOOL_CALL, ApprovalGrant, GrantStatus
    from modules.tools.discovery.my_actions_reads import at_said
    from services.click_memory import card_words

    rows = (db.query(ApprovalGrant)
            .filter(ApprovalGrant.workspace_id == workspace_id,
                    ApprovalGrant.subject_type == SUBJECT_TOOL_CALL,
                    ApprovalGrant.kind == KIND_APPROVAL,
                    ApprovalGrant.status == GrantStatus.PENDING.value,
                    ApprovalGrant.expires_at > now,
                    ApprovalGrant.requested_at >= now - LOOKBACK,
                    ApprovalGrant.details[DRIVING_USER].astext == str(user_id))
            .order_by(ApprovalGrant.requested_at.desc())
            .limit(MAX_ASKS)
            .all())
    return [{"card": card_words(row), "at": at_said(_utc(row.requested_at))} for row in rows]


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


__all__ = ["LOOKBACK", "MAX_ASKS", "waiting_asks"]
