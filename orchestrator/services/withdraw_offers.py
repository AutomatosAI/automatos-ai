"""F091-C2 (night 3): a "no" in chat offers to withdraw Auto's pending request.

Night 3: the owner told Auto "no" in chat, but Auto's delete card (#600) stayed
pending for its whole 24 h — the words in the conversation never reached the
card. When the owner's message reads as a refusal and this conversation still
has a gated request of Auto's waiting on a yes, the reply now carries a card
offering to withdraw it (one click denies the grant). Nothing is withdrawn
without that click: a "no" to something else must not cancel a request.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_REFUSAL = re.compile(
    r"^\s*(?:no\b|nope\b|nah\b|don'?t\b|do not\b|stop\b|cancel\b|never ?mind\b|leave it\b|"
    r"not now\b|keep it\b|forget it\b|hold on\b|wait\b)",
    re.IGNORECASE,
)
MAX_OFFERED = 3


def is_refusal(text: Optional[str]) -> bool:
    """A message that opens with a refusal ("No — keep it", "Don't delete that")."""
    return bool(_REFUSAL.match(text or ""))


def pending_requests(db: Any, workspace_id: Any, conversation_id: str) -> List[Dict[str, Any]]:
    """This conversation's gated requests still waiting on a yes, newest first."""
    from core.models.approval_grants import (
        KIND_APPROVAL, SUBJECT_TOOL_CALL, ApprovalGrant, GrantStatus,
    )

    rows = (
        db.query(ApprovalGrant)
        .filter(
            ApprovalGrant.workspace_id == workspace_id,
            ApprovalGrant.subject_type == SUBJECT_TOOL_CALL,
            ApprovalGrant.kind == KIND_APPROVAL,
            ApprovalGrant.status == GrantStatus.PENDING.value,
            ApprovalGrant.details["conversation_id"].astext == str(conversation_id),
        )
        .order_by(ApprovalGrant.requested_at.desc())
        .limit(MAX_OFFERED)
        .all()
    )
    return [{"grant_id": g.id, "action": g.tool_name, "reason": g.reason} for g in rows]


def withdraw_offer(db: Any, workspace_id: Any, conversation_id: Optional[str],
                   user_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The offer to put on this reply, or None."""
    if not conversation_id or not is_refusal(user_text):
        return None
    requests = pending_requests(db, workspace_id, conversation_id)
    if not requests:
        return None
    return {
        "message": "You said no — withdraw the request still waiting for your yes?",
        "requests": requests,
    }
